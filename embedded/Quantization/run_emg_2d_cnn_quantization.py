#!/usr/bin/env python3
"""Quantize the four CBEB EMG 2D-CNN SavedModels and record host CPU numbers.

Uses the existing dynamic-range path in
``functions/Embedded_Model_Convertion.py`` (``tf.lite.Optimize.DEFAULT``:
int8 weights, float activations) and a second conversion with no
optimizations (float32 weights and activations, ``*.float.tflite``).
Full-integer INT8 is not produced here: the raw EMG windows are not mounted
on this host, so there is no representative dataset, and dynamic-range
accuracy on real EMG has not been shown to pass.

Latency is model-only (no EMG preprocess), on the testbench CPU. It is not
board, edge, or prosthesis latency. The comparable columns are Keras
``model(x, training=False)``, float TFLite ``Interpreter.invoke``, and
dynamic-range TFLite ``Interpreter.invoke``. ``Model.predict`` is recorded
separately because its overhead confounds the runtime comparison.

Accuracy here is numerical agreement between Keras logits and TFLite logits
on synthetic inputs, plus a clean-window subset through
``Pipeline_deployment_v2`` when the dataset directory is actually present.
EMG corruption accuracy is a different script
(``run_emg_tflite_corruption.py``).

Does not retrain and does not modify the SavedModels.
"""

import argparse
import csv
import gc
import os
import sys
import time

# This script's numbers are host-CPU model latency. Hide GPUs before TF loads.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(QUANT_DIR, "..", ".."))
FUNCTIONS_DIR = os.path.join(QUANT_DIR, "functions")
PIPELINE_DIR = os.path.join(
    PACKAGE_ROOT, "EMG", "Experiments", "Deep_learning", "Train"
)
TRAININGS_DIR = os.path.join(PIPELINE_DIR, "trainings")
DEFAULT_OUT_DIR = os.path.join(PACKAGE_ROOT, "results", "quantization")

if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

from Embedded_Model import Embedded_Model  # noqa: E402
from Embedded_Model_Convertion import (  # noqa: E402
    convert_saved_model_to_float_tflite,
    convert_saved_model_to_tflite,
)

import tensorflow as tf  # noqa: E402

PAPER_MODELS = (
    {
        "dataset": "65hd",
        "dataset_key": "65hd",
        "norm": "bychannel",
        "name": "EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_bychannel_adamW",
    },
    {
        "dataset": "65hd",
        "dataset_key": "65hd",
        "norm": "bysample",
        "name": "EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_bysample_adamW",
    },
    {
        "dataset": "hyser",
        "dataset_key": "hyser",
        "norm": "bychannel",
        "name": "EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_bychannel_adamW",
    },
    {
        "dataset": "hyser",
        "dataset_key": "hyser",
        "norm": "bysample",
        "name": "EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_bysample_adamW",
    },
)

DATA_ROOT = os.path.abspath(os.environ.get("CBEB_DATA_ROOT", os.path.join(PACKAGE_ROOT, "data")))
DATASET_RELATIVE = {
    "65hd": os.path.join(DATA_ROOT, "65_HDsEMG"),
    "hyser": os.path.join(DATA_ROOT, "HYSER", "files", "hd-semg", "1.0.0", "pr_dataset"),
}

LATENCY_FIELDS = [
    "dataset",
    "norm",
    "precision",
    "size_mb",
    "latency_mean_ms",
    "p50_ms",
    "p95_ms",
    "device_note",
    "model",
    "artifact_path",
    "quant_mode",
    "n_timed",
    "warmup",
    "input_shape",
    "h5_mb",
    "savedmodel_mb",
    "infer_call_mean_ms",
    "infer_call_p50_ms",
    "infer_call_p95_ms",
]

ACCURACY_FIELDS = [
    "dataset",
    "norm",
    "precision",
    "measurement",
    "n",
    "max_abs_logit_diff",
    "mean_abs_logit_diff",
    "argmax_agreement",
    "keras_accuracy",
    "tflite_accuracy",
    "accuracy_delta",
    "notes",
]


def force_cpu():
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception as exc:
        print("GPU hide:", exc)


def cpu_model_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def file_size(path):
    return os.path.getsize(path) if os.path.isfile(path) else 0


def savedmodel_bytes(model_dir):
    """Inference SavedModel payload, excluding the duplicate .h5 and logs."""
    names = [
        os.path.join(model_dir, "saved_model.pb"),
        os.path.join(model_dir, "keras_metadata.pb"),
    ]
    total = sum(file_size(path) for path in names)
    variables = os.path.join(model_dir, "variables")
    if os.path.isdir(variables):
        for root, _dirs, files in os.walk(variables):
            for name in files:
                total += file_size(os.path.join(root, name))
    return total


def mb(num_bytes):
    return num_bytes / float(1024 ** 2)


def fmt(value, digits=4):
    if value is None or value == "":
        return ""
    return ("%%.%df" % digits) % float(value)


def latency_stats(samples_ms):
    samples = np.asarray(samples_ms, dtype=np.float64)
    return {
        "latency_mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def time_calls(fn, warmup, n_timed):
    for _ in range(warmup):
        fn()
    samples = np.empty(n_timed, dtype=np.float64)
    for index in range(n_timed):
        start = time.perf_counter()
        fn()
        samples[index] = (time.perf_counter() - start) * 1000.0
    return samples


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataset_root(dataset_key):
    return os.path.abspath(DATASET_RELATIVE[dataset_key])


def data_mount_note():
    link = DATA_ROOT
    if os.path.islink(link) and not os.path.isdir(link):
        return "%s -> %s is not mounted" % (link, os.readlink(link))
    if not os.path.isdir(link):
        return "%s is not a directory" % link
    return ""


def concrete_input_shape(input_shape):
    """Replace a symbolic batch dim with 1. EMG images are (1, H, W, C)."""
    dims = []
    for dim in input_shape:
        if dim is None:
            dims.append(1)
        else:
            dims.append(int(dim))
    if dims[0] != 1:
        dims = [1] + dims
    return tuple(dims)


def synthetic_inputs(input_shape, n_samples, seed):
    """Uniform [0, 1] inputs. Deployment minmax windows live in that range."""
    rng = np.random.default_rng(seed)
    single = concrete_input_shape(input_shape)
    batch_shape = (int(n_samples),) + single[1:]
    return rng.random(batch_shape, dtype=np.float32)


def tflite_logits(embedded, sample):
    embedded.set_input_tensor(sample)
    embedded.interpreter.invoke()
    output = embedded.interpreter.get_output_details()[0]
    return np.asarray(embedded.interpreter.get_tensor(output["index"]), dtype=np.float32).reshape(-1)


def numerical_agreement(keras_model, embedded, input_shape, n_samples, seed):
    inputs = synthetic_inputs(input_shape, n_samples, seed)
    keras_logits = np.asarray(keras_model.predict(inputs, verbose=0), dtype=np.float32)
    tflite_batch = np.stack([tflite_logits(embedded, sample) for sample in inputs], axis=0)
    if keras_logits.shape != tflite_batch.shape:
        raise RuntimeError(
            "logit shape mismatch keras %s vs tflite %s" % (keras_logits.shape, tflite_batch.shape)
        )
    diff = np.abs(keras_logits - tflite_batch)
    keras_ids = np.argmax(keras_logits, axis=-1)
    tflite_ids = np.argmax(tflite_batch, axis=-1)
    return {
        "n": int(n_samples),
        "max_abs_logit_diff": float(np.max(diff)),
        "mean_abs_logit_diff": float(np.mean(diff)),
        "argmax_agreement": float(np.mean(keras_ids == tflite_ids)),
    }


def clean_window_agreement(keras_model, embedded, spec, max_windows):
    """One subject, clean windows, deployment preprocess. Bounded length.

    Returns a result dict, or a skip dict with ``skipped`` set.
    This is not a Table V rerun.
    """
    root = dataset_root(spec["dataset_key"])
    mount_note = data_mount_note()
    if not os.path.isdir(root):
        reason = "clean-window subset skipped; dataset directory missing: %s" % root
        if mount_note:
            reason = "%s (%s)" % (reason, mount_note)
        return {"skipped": reason}

    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    try:
        import Pipeline_deployment_v2 as pipe
    except ImportError as exc:
        return {"skipped": "clean-window subset skipped; deployment import failed: %s" % exc}

    params = dict(pipe.DEFAULT_PARAMS)
    pipe.apply_dataset_profile(params, spec["dataset_key"])
    params["normalization_mode"] = spec["norm"]
    params["n_trials"] = 1
    params["dataset_path"] = os.path.relpath(root, PIPELINE_DIR)

    try:
        files = pipe.build_files_list(params, PIPELINE_DIR, subject_indices=[0])
    except Exception as exc:
        return {"skipped": "clean-window subset skipped; subject discovery failed: %s" % exc}

    subject_path = files[0]
    emg, labels = pipe.load_subject_data(subject_path, params)
    gestures = list(params["gesture_list"])
    emg, _raw_labels, y_data = pipe.select_gesture_data(emg, labels, gestures, gestures)

    step = int(params["step"])
    need = (int(max_windows) + 1) * step
    emg = np.asarray(emg[:, :need])
    y_data = np.asarray(y_data[:need])
    num_windows = (emg.shape[1] - step) // step
    if num_windows < 2:
        return {
            "skipped": "clean-window subset skipped; subject stream shorter than 2 windows (%s)"
            % subject_path
        }

    def tflite_infer(batch_input):
        data = batch_input.numpy() if hasattr(batch_input, "numpy") else np.asarray(batch_input)
        rows = []
        for sample in data:
            rows.append(tflite_logits(embedded, sample))
        return tf.constant(np.stack(rows, axis=0), dtype=tf.float32)

    keras_infer = pipe.build_infer_fn(keras_model)
    y_keras, y_true, _latency_k = pipe.predict_gesture_fast(
        model=keras_model,
        stream_data=emg,
        y_data=y_data,
        params=params,
        batch_size=32,
        chunk_steps=max(num_windows, 1),
        infer_fn=keras_infer,
    )
    y_tflite, y_true_t, _latency_t = pipe.predict_gesture_fast(
        model=keras_model,
        stream_data=emg,
        y_data=y_data,
        params=params,
        batch_size=32,
        chunk_steps=max(num_windows, 1),
        infer_fn=tflite_infer,
    )
    if len(y_keras) != len(y_tflite) or not np.array_equal(y_true, y_true_t):
        return {
            "skipped": "clean-window subset produced mismatched prediction lengths (%d vs %d)"
            % (len(y_keras), len(y_tflite))
        }

    keras_acc = float(np.mean(y_keras == y_true)) if len(y_true) else float("nan")
    tflite_acc = float(np.mean(y_tflite == y_true)) if len(y_true) else float("nan")
    return {
        "skipped": "",
        "n": int(len(y_true)),
        "argmax_agreement": float(np.mean(y_keras == y_tflite)) if len(y_keras) else float("nan"),
        "keras_accuracy": keras_acc,
        "tflite_accuracy": tflite_acc,
        "accuracy_delta": float(tflite_acc - keras_acc),
        "subject": subject_path,
        "notes": (
            "clean windows, fault_mode none, first filtered subject, first %d windows "
            "after deployment preprocess and majority vote (n_voting=%s). "
            "Not a full Table V reproduction. subject=%s"
            % (len(y_true), params.get("n_voting"), subject_path)
        ),
    }


def benchmark_model(spec, device_note, warmup, n_timed, agreement_n, seed, max_windows):
    model_dir = os.path.join(TRAININGS_DIR, spec["name"])
    if not os.path.isfile(os.path.join(model_dir, "saved_model.pb")):
        raise FileNotFoundError("SavedModel missing: %s" % model_dir)

    print("=== %s ===" % spec["name"], flush=True)
    converted = convert_saved_model_to_tflite(model_dir)
    print(
        "tflite dynamic %s bytes=%d mode=%s"
        % (converted["tflite_path"], converted["num_bytes"], converted["quant_mode"]),
        flush=True,
    )
    float_converted = convert_saved_model_to_float_tflite(model_dir)
    print(
        "tflite float %s bytes=%d mode=%s"
        % (
            float_converted["tflite_path"],
            float_converted["num_bytes"],
            float_converted["quant_mode"],
        ),
        flush=True,
    )

    keras_model = tf.keras.models.load_model(model_dir, compile=False)
    input_shape = tuple(keras_model.input_shape)
    if len(input_shape) != 4:
        raise RuntimeError("expected rank-4 input, got %s" % (input_shape,))

    sample = synthetic_inputs(input_shape, 1, seed)
    embedded = Embedded_Model(model_dir, select_tf_ops=converted["select_tf_ops"])
    embedded_float = Embedded_Model(
        model_dir,
        select_tf_ops=float_converted["select_tf_ops"],
        tflite_path=float_converted["tflite_path"],
    )
    # Exercise both the unbatched rank-4 image and the explicit batch-1 tensor.
    embedded.set_input_tensor(sample[0])
    embedded.set_input_tensor(sample)
    embedded_float.set_input_tensor(sample[0])
    embedded_float.set_input_tensor(sample)

    output_detail = embedded.interpreter.get_output_details()[0]
    float_output = embedded_float.interpreter.get_output_details()[0]

    def keras_predict():
        keras_model.predict(sample, verbose=0)

    def keras_infer():
        np.asarray(keras_model(sample, training=False).numpy())

    def tflite_invoke():
        embedded.interpreter.invoke()
        embedded.interpreter.get_tensor(output_detail["index"])

    def tflite_float_invoke():
        embedded_float.interpreter.invoke()
        embedded_float.interpreter.get_tensor(float_output["index"])

    print("timing keras predict...", flush=True)
    predict_samples = time_calls(keras_predict, warmup, n_timed)
    print("timing keras forward...", flush=True)
    infer_samples = time_calls(keras_infer, warmup, n_timed)
    embedded_float.set_input_tensor(sample)
    print("timing tflite float invoke...", flush=True)
    float_samples = time_calls(tflite_float_invoke, warmup, n_timed)
    embedded.set_input_tensor(sample)
    print("timing tflite dynamic invoke...", flush=True)
    tflite_samples = time_calls(tflite_invoke, warmup, n_timed)

    predict_stats = latency_stats(predict_samples)
    infer_stats = latency_stats(infer_samples)
    tflite_stats = latency_stats(tflite_samples)
    float_stats = latency_stats(float_samples)

    sm_mb = mb(savedmodel_bytes(model_dir))
    h5_mb = mb(file_size(os.path.join(model_dir, "ResNet2D.h5")))
    tflite_mb = mb(converted["num_bytes"])
    float_mb = mb(float_converted["num_bytes"])
    quant_mode = converted["quant_mode"]
    shape_text = "x".join(str(dim) for dim in concrete_input_shape(input_shape))

    fp32_row = {
        "dataset": spec["dataset"],
        "norm": spec["norm"],
        "precision": "fp32",
        "size_mb": fmt(sm_mb),
        "latency_mean_ms": fmt(predict_stats["latency_mean_ms"]),
        "p50_ms": fmt(predict_stats["p50_ms"]),
        "p95_ms": fmt(predict_stats["p95_ms"]),
        "device_note": device_note,
        "model": spec["name"],
        "artifact_path": model_dir,
        "quant_mode": "fp32_savedmodel",
        "n_timed": n_timed,
        "warmup": warmup,
        "input_shape": shape_text,
        "h5_mb": fmt(h5_mb),
        "savedmodel_mb": fmt(sm_mb),
        "infer_call_mean_ms": fmt(infer_stats["latency_mean_ms"]),
        "infer_call_p50_ms": fmt(infer_stats["p50_ms"]),
        "infer_call_p95_ms": fmt(infer_stats["p95_ms"]),
    }
    tflite_row = {
        "dataset": spec["dataset"],
        "norm": spec["norm"],
        "precision": "tflite_dynamic_range",
        "size_mb": fmt(tflite_mb),
        "latency_mean_ms": fmt(tflite_stats["latency_mean_ms"]),
        "p50_ms": fmt(tflite_stats["p50_ms"]),
        "p95_ms": fmt(tflite_stats["p95_ms"]),
        "device_note": device_note + "; TFLite column is Interpreter.invoke only",
        "model": spec["name"],
        "artifact_path": converted["tflite_path"],
        "quant_mode": quant_mode,
        "n_timed": n_timed,
        "warmup": warmup,
        "input_shape": shape_text,
        "h5_mb": fmt(h5_mb),
        "savedmodel_mb": fmt(sm_mb),
        "infer_call_mean_ms": "",
        "infer_call_p50_ms": "",
        "infer_call_p95_ms": "",
    }
    # Paper column. Same forward the deployment pipeline calls, not Model.predict.
    keras_forward_row = {
        "dataset": spec["dataset"],
        "norm": spec["norm"],
        "precision": "keras_fp32",
        "size_mb": fmt(h5_mb),
        "latency_mean_ms": fmt(infer_stats["latency_mean_ms"]),
        "p50_ms": fmt(infer_stats["p50_ms"]),
        "p95_ms": fmt(infer_stats["p95_ms"]),
        "device_note": device_note
        + "; size_mb is ResNet2D.h5 weights; savedmodel_mb includes AdamW slots",
        "model": spec["name"],
        "artifact_path": os.path.join(model_dir, "ResNet2D.h5"),
        "quant_mode": "keras_forward",
        "n_timed": n_timed,
        "warmup": warmup,
        "input_shape": shape_text,
        "h5_mb": fmt(h5_mb),
        "savedmodel_mb": fmt(sm_mb),
        "infer_call_mean_ms": fmt(infer_stats["latency_mean_ms"]),
        "infer_call_p50_ms": fmt(infer_stats["p50_ms"]),
        "infer_call_p95_ms": fmt(infer_stats["p95_ms"]),
    }
    float_row = {
        "dataset": spec["dataset"],
        "norm": spec["norm"],
        "precision": "tflite_float",
        "size_mb": fmt(float_mb),
        "latency_mean_ms": fmt(float_stats["latency_mean_ms"]),
        "p50_ms": fmt(float_stats["p50_ms"]),
        "p95_ms": fmt(float_stats["p95_ms"]),
        "device_note": device_note
        + "; TFLite float column is Interpreter.invoke only; converter optimizations unset",
        "model": spec["name"],
        "artifact_path": float_converted["tflite_path"],
        "quant_mode": float_converted["quant_mode"],
        "n_timed": n_timed,
        "warmup": warmup,
        "input_shape": shape_text,
        "h5_mb": fmt(h5_mb),
        "savedmodel_mb": fmt(sm_mb),
        "infer_call_mean_ms": "",
        "infer_call_p50_ms": "",
        "infer_call_p95_ms": "",
    }

    print("numerical agreement n=%d..." % agreement_n, flush=True)
    agree = numerical_agreement(keras_model, embedded, input_shape, agreement_n, seed)
    agree_row = {
        "dataset": spec["dataset"],
        "norm": spec["norm"],
        "precision": "tflite_dynamic_range",
        "measurement": "synthetic_logit_agreement",
        "n": agree["n"],
        "max_abs_logit_diff": fmt(agree["max_abs_logit_diff"], 6),
        "mean_abs_logit_diff": fmt(agree["mean_abs_logit_diff"], 6),
        "argmax_agreement": fmt(agree["argmax_agreement"], 6),
        "keras_accuracy": "",
        "tflite_accuracy": "",
        "accuracy_delta": "",
        "notes": (
            "Same %d uniform[0,1] inputs, seed=%d. Keras predict vs TFLite. "
            "Scores are logits (SCE from_logits, no softmax in the graph). "
            "Not classification accuracy and not Table V."
            % (agreement_n, seed)
        ),
    }
    print("float numerical agreement n=%d..." % agreement_n, flush=True)
    agree_float = numerical_agreement(
        keras_model, embedded_float, input_shape, agreement_n, seed
    )
    agree_float_row = {
        "dataset": spec["dataset"],
        "norm": spec["norm"],
        "precision": "tflite_float",
        "measurement": "synthetic_logit_agreement",
        "n": agree_float["n"],
        "max_abs_logit_diff": fmt(agree_float["max_abs_logit_diff"], 6),
        "mean_abs_logit_diff": fmt(agree_float["mean_abs_logit_diff"], 6),
        "argmax_agreement": fmt(agree_float["argmax_agreement"], 6),
        "keras_accuracy": "",
        "tflite_accuracy": "",
        "accuracy_delta": "",
        "notes": (
            "Same %d uniform[0,1] inputs, seed=%d. Keras predict vs float TFLite "
            "(converter optimizations unset). Scores are logits. "
            "Not classification accuracy on EMG and not Table V."
            % (agreement_n, seed)
        ),
    }

    print("clean-window check...", flush=True)
    try:
        windows = clean_window_agreement(keras_model, embedded, spec, max_windows)
    except Exception as exc:
        windows = {"skipped": "clean-window subset failed: %s" % exc}

    accuracy_rows = [agree_row, agree_float_row]
    if windows.get("skipped"):
        accuracy_rows.append(
            {
                "dataset": spec["dataset"],
                "norm": spec["norm"],
                "precision": "tflite_dynamic_range",
                "measurement": "clean_windows_subset",
                "n": "",
                "max_abs_logit_diff": "",
                "mean_abs_logit_diff": "",
                "argmax_agreement": "",
                "keras_accuracy": "",
                "tflite_accuracy": "",
                "accuracy_delta": "",
                "notes": windows["skipped"],
            }
        )
    else:
        accuracy_rows.append(
            {
                "dataset": spec["dataset"],
                "norm": spec["norm"],
                "precision": "tflite_dynamic_range",
                "measurement": "clean_windows_subset",
                "n": windows["n"],
                "max_abs_logit_diff": "",
                "mean_abs_logit_diff": "",
                "argmax_agreement": fmt(windows["argmax_agreement"], 6),
                "keras_accuracy": fmt(windows["keras_accuracy"], 6),
                "tflite_accuracy": fmt(windows["tflite_accuracy"], 6),
                "accuracy_delta": fmt(windows["accuracy_delta"], 6),
                "notes": windows["notes"],
            }
        )

    del keras_model
    del embedded
    del embedded_float
    tf.keras.backend.clear_session()
    gc.collect()
    return [fp32_row, keras_forward_row, float_row, tflite_row], accuracy_rows


def write_readme(path, latency_rows, accuracy_rows, meta):
    lines = [
        "# EMG 2D-CNN quantization (host run)",
        "",
        "Float and dynamic-range TensorFlow Lite of the four CBEB SavedModels.",
        "Generated by `fx_bios_ia/embedded/Quantization/run_emg_2d_cnn_quantization.py`.",
        "",
        "## Quantization mode",
        "",
        "Two conversions of each SavedModel:",
        "",
        "- `tflite_float`: `TFLiteConverter.from_saved_model` with optimizations left",
        "  unset (float32 weights and activations). File: `trainings/<name>.float.tflite`.",
        "- `tflite_dynamic_range`: same converter with",
        "  `converter.optimizations = [tf.lite.Optimize.DEFAULT]` (int8 weights,",
        "  float activations and I/O). File: `trainings/<name>.tflite`.",
        "",
        "This is **not** full-integer INT8. No representative dataset was calibrated,",
        "because the raw 65HD and Hyser windows are not mounted on this host",
        "(`%s`). INT8 is skipped until dynamic-range accuracy on those windows" % meta["data_note"],
        "passes the 1–2 pp gate.",
        "`SELECT_TF_OPS` is used only if the builtin conversion throws. This run: `%s`."
        % meta["select_tf_ops_used"],
        "",
        "Dynamic-range files stay at `model_path + '.tflite'` so `Embedded_Model`",
        "keeps its historical path. Float files use `model_path + '.float.tflite'`.",
        "",
        "## How to rerun",
        "",
        "```bash",
        "python3 fx_bios_ia/embedded/Quantization/run_emg_2d_cnn_quantization.py",
        "```",
        "",
        "From the repo root. Optional flags: `--n-timed`, `--warmup`, `--agreement-n`,",
        "`--max-windows`, `--out-dir`. Requires TensorFlow (tested with 2.13.1 on",
        "Python 3.8). `tflite_runtime` is optional; `Embedded_Model` falls back to",
        "`tf.lite.Interpreter`.",
        "",
        "The script forces CPU (`CUDA_VISIBLE_DEVICES` empty). It does not retrain",
        "and it does not overwrite SavedModels, history CSVs, or `ResNet2D.h5`.",
        "",
        "## What the CSVs measure",
        "",
        "- `quantization_latency.csv`",
        "  - `precision=fp32`: `size_mb` is the SavedModel payload",
        "    (`saved_model.pb` + `keras_metadata.pb` + `variables/`).",
        "    `latency_*` is `Model.predict` at batch 1 (Keras overhead).",
        "    `infer_call_*` is `model(x, training=False)`.",
        "  - `precision=keras_fp32`: paper column. `size_mb` is `ResNet2D.h5`",
        "    (weights only). `latency_*` is that same forward call.",
        "    `savedmodel_mb` still includes AdamW optimizer slots and is not the",
        "    deployment size.",
        "  - `precision=tflite_float`: `size_mb` is the `.float.tflite` file.",
        "    `latency_*` is `Interpreter.invoke` only (input already copied).",
        "  - `precision=tflite_dynamic_range`: `size_mb` is the `.tflite` file.",
        "    `latency_*` is `Interpreter.invoke` only (input already copied).",
        "  - Compare `keras_fp32`, `tflite_float`, and `tflite_dynamic_range`.",
        "    Do not compare `Model.predict` with `Interpreter.invoke`.",
        "- `quantization_accuracy.csv`",
        "  - `synthetic_logit_agreement`: max/mean absolute logit difference and",
        "    argmax agreement on shared uniform `[0, 1]` inputs.",
        "    This is not accuracy on EMG and not Table V.",
        "  - `clean_windows_subset`: filled only when a raw dataset directory is",
        "    mounted. One subject, clean windows, deployment preprocess. Still not",
        "    a full table rerun.",
        "",
        "Host: %s. TensorFlow %s. Warmup %d, timed invokes %d, agreement samples %d."
        % (meta["device"], meta["tf"], meta["warmup"], meta["n_timed"], meta["agreement_n"]),
        "Input shape observed: %s. Outputs are 12-class logits (no softmax)."
        % meta["input_shape"],
        "",
        "## This run",
        "",
        "| dataset | norm | precision | size_mb | mean ms | p50 ms | p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in latency_rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["dataset"],
                row["norm"],
                row["precision"],
                row["size_mb"],
                row["latency_mean_ms"],
                row["p50_ms"],
                row["p95_ms"],
            )
        )
    lines.extend(
        [
            "",
            "| dataset | norm | measurement | n | max abs logit | argmax agree | keras acc | tflite acc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in accuracy_rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["dataset"],
                row["norm"],
                row["measurement"],
                row["n"],
                row["max_abs_logit_diff"],
                row["argmax_agreement"],
                row["keras_accuracy"],
                row["tflite_accuracy"],
            )
        )
    lines.extend(
        [
            "",
            "## Paper table and EMG accuracy",
            "",
            "`quantization_paper_table.csv` is written by",
            "`run_emg_tflite_corruption.py`. Columns: dataset, norm, precision",
            "(keras_fp32 / tflite_float / tflite_dynamic), acc_clean, acc at k=4",
            "on noise, p50/p95 ms, size_mb, machine. The machine string is testbench",
            "CPU, not an edge device and not a prosthesis.",
            "Keras `acc_*` is the unweighted mean of the existing FP32 experiment CSVs.",
            "TFLite `acc_*` stays empty until that script can read the raw EMG.",
            "INT8 is not produced here.",
            "",
            "On this host, float TFLite `Interpreter.invoke` is faster than",
            "dynamic-range invoke. The large gap versus `keras_fp32` is the runtime",
            "(Keras forward vs TFLite invoke), not the weight quantization.",
            "Do not quote `precision=fp32` (`Model.predict`, ~80 ms) as the",
            "deployment latency.",
            "",
            "## Caveats",
            "",
            "- These latencies are testbench CPU only. This run does not measure",
            "  board, edge, or prosthesis latency, and it does not measure energy.",
            "- `Model.predict` (`precision=fp32`) includes Keras call overhead.",
            "  The paper latency column is `precision=keras_fp32`.",
            "- Dynamic-range TFLite is not bit-exact with FP32. Argmax can still match",
            "  when the logit margin is larger than the quantization error.",
            "- Do not cite these rows as a reproduction of the paper's full accuracy",
            "  table. Stress-test CSVs next to the SavedModels remain the FP32",
            "  experiment_1 (noise) and experiment_3 (dropout) results.",
            "- Labels are 0-indexed. `Embedded_Model.classify_batch_data` no longer",
            "  subtracts 1 before argmax.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-timed", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--agreement-n", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_timed < 100:
        raise SystemExit("--n-timed must be >= 100")
    if args.warmup < 1:
        raise SystemExit("--warmup must be >= 1")

    force_cpu()
    intra = tf.config.threading.get_intra_op_parallelism_threads()
    inter = tf.config.threading.get_inter_op_parallelism_threads()
    device_note = (
        "testbench CPU (%s); not an edge device and not a prosthesis; "
        "batch=1; model latency only (no EMG preprocess); "
        "precision=fp32 latency_* is Keras Model.predict; "
        "precision=keras_fp32 latency_* is model(x, training=False); "
        "TFLite latency_* is Interpreter.invoke only; "
        "TF threads intra=%s inter=%s (0=default)"
        % (cpu_model_name(), intra, inter)
    )
    print(device_note, flush=True)
    print("tensorflow", tf.__version__, flush=True)

    latency_rows = []
    accuracy_rows = []
    select_flags = []
    input_shape = ""
    for spec in PAPER_MODELS:
        model_latency, model_accuracy = benchmark_model(
            spec,
            device_note,
            args.warmup,
            args.n_timed,
            args.agreement_n,
            args.seed,
            args.max_windows,
        )
        latency_rows.extend(model_latency)
        accuracy_rows.extend(model_accuracy)
        select_flags.append(model_latency[2]["quant_mode"])
        select_flags.append(model_latency[3]["quant_mode"])
        input_shape = model_latency[0]["input_shape"]

    os.makedirs(args.out_dir, exist_ok=True)
    latency_path = os.path.join(args.out_dir, "quantization_latency.csv")
    accuracy_path = os.path.join(args.out_dir, "quantization_accuracy.csv")
    readme_path = os.path.join(args.out_dir, "README.md")
    write_csv(latency_path, LATENCY_FIELDS, latency_rows)
    write_csv(accuracy_path, ACCURACY_FIELDS, accuracy_rows)
    unique_modes = sorted(set(select_flags))
    write_readme(
        readme_path,
        latency_rows,
        accuracy_rows,
        {
            "data_note": data_mount_note() or "dataset directories were checked per model",
            "select_tf_ops_used": ", ".join(unique_modes),
            "device": device_note,
            "tf": tf.__version__,
            "warmup": args.warmup,
            "n_timed": args.n_timed,
            "agreement_n": args.agreement_n,
            "input_shape": input_shape,
        },
    )
    print("wrote", latency_path, flush=True)
    print("wrote", accuracy_path, flush=True)
    print("wrote", readme_path, flush=True)


if __name__ == "__main__":
    main()
