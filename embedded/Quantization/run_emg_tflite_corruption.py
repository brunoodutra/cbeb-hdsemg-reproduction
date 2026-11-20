#!/usr/bin/env python3
"""Dynamic-range TFLite on the same noise and dropout grid as the FP32 CSVs.

Testbench only. Does not claim board latency, energy, or full-integer INT8.

Default scope when the raw EMG is mounted:
  one dataset (65hd; pass --dataset hyser after the 65hd delta is stable),
  both norms, experiment 1 (noise, k=0..7) and experiment 3 (dropout, k=0..7).

The subject list is the basename set already present in the FP32
``*_experiment_{1,3}_*_all.csv`` files, not a new cohort.

If ``data`` is not mounted this script still writes:
  - FP32 reference curves taken from those CSVs (real numbers),
  - a corruption table whose TFLite accuracy cells are empty and marked BLOCKED,
  - the paper table (size and latency filled from quantization_latency.csv when
    that file exists; TFLite accuracy left blank).

It does not invent EMG accuracy.
"""
import argparse
import csv
import os
import sys

QUANT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(QUANT_DIR, "..", ".."))
PIPELINE_DIR = os.path.join(
    PACKAGE_ROOT, "EMG", "Experiments", "Deep_learning", "Train"
)
TRAININGS_DIR = os.path.join(PIPELINE_DIR, "trainings")
DEFAULT_OUT = os.path.join(PACKAGE_ROOT, "results", "quantization")
FUNCTIONS_DIR = os.path.join(QUANT_DIR, "functions")
DATA_ROOT = os.path.abspath(os.environ.get("CBEB_DATA_ROOT", os.path.join(PACKAGE_ROOT, "data")))

DATASET_RELATIVE = {
    "65hd": os.path.join(DATA_ROOT, "65_HDsEMG"),
    "hyser": os.path.join(DATA_ROOT, "HYSER", "files", "hd-semg", "1.0.0", "pr_dataset"),
}
MODEL_NAMES = {
    ("65hd", "bychannel"): "EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_bychannel_adamW",
    ("65hd", "bysample"): "EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_bysample_adamW",
    ("hyser", "bychannel"): "EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_bychannel_adamW",
    ("hyser", "bysample"): "EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_bysample_adamW",
}
PRECISION_FILE = {
    "tflite_dynamic": ".tflite",
    "tflite_float": ".float.tflite",
}
SELECTED_K = 4
GATE_PP = 2.0
EXPERIMENTS = (1, 3)


def dataset_root(dataset):
    return os.path.abspath(DATASET_RELATIVE[dataset])


def mount_note():
    link = DATA_ROOT
    if os.path.islink(link) and not os.path.isdir(link):
        return "%s -> %s is not mounted" % (link, os.readlink(link))
    if not os.path.isdir(link):
        return "%s is not a directory" % link
    return ""


def blocked_reason():
    missing = []
    for dataset in ("65hd", "hyser"):
        root = dataset_root(dataset)
        if not os.path.isdir(root):
            missing.append(root)
    note = mount_note()
    reason = (
        "BLOCKED: raw EMG is not available, so TFLite accuracy was not computed "
        "and no EMG accuracy was invented. Missing dataset directory: %s"
        % "; ".join(missing)
    )
    if note:
        reason = "%s (%s)" % (reason, note)
    return reason, missing


def fp32_csv(dataset, norm, exp):
    name = MODEL_NAMES[(dataset, norm)]
    tag = "hyser" if dataset == "hyser" else "65hd"
    return os.path.join(
        TRAININGS_DIR, "%s_experiment_%d_%s_all.csv" % (name, exp, tag)
    )


def file_base(path):
    text = str(path).replace("\\", "/").rstrip("/")
    return text.split("/")[-1]


def xcol(exp):
    return "num_noisy_channels" if int(exp) == 1 else "num_fault_channels"


def read_fp32(dataset, norm, exp):
    import pandas as pd

    path = fp32_csv(dataset, norm, exp)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["file_base"] = frame["file"].map(file_base)
    return frame


def mean_at_k(frame, exp, k, n_trials=None):
    column = xcol(exp)
    part = frame[frame[column] == int(k)]
    if n_trials is not None and "n_trials" in part.columns:
        part = part[part["n_trials"] == int(n_trials)]
    if part.empty:
        return None
    return float(part["accuracy"].mean())


def majority_n_trials(frame):
    """Paper grids are n_trials=1. A few FP32 rows are n_trials=3; do not follow those."""
    return int(frame["n_trials"].mode().iloc[0])


def count_n_trials(path, n_trials):
    if not os.path.isfile(path):
        return 0
    import pandas as pd

    frame = pd.read_csv(path, usecols=["n_trials"])
    return int((frame["n_trials"] == int(n_trials)).sum())


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value, digits=6):
    if value is None or value == "":
        return ""
    return ("%%.%df" % digits) % float(value)


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def machine_label():
    return (
        "testbench CPU (%s); not an edge device and not a prosthesis" % cpu_name()
    )


def fp32_reference_rows():
    """Unweighted row means, same aggregation as Fig. 3 / Table V clean rows."""
    rows = []
    for dataset in ("65hd", "hyser"):
        for norm in ("bychannel", "bysample"):
            for exp in EXPERIMENTS:
                frame = read_fp32(dataset, norm, exp)
                for k in range(0, 8):
                    rows.append({
                        "dataset": dataset,
                        "norm": norm,
                        "experiment": exp,
                        "condition": "noise" if exp == 1 else "dropout",
                        "k": k,
                        "fp32_acc": fmt(mean_at_k(frame, exp, k)),
                        "n_rows": int((frame[xcol(exp)] == k).sum()),
                        "aggregation": "unweighted mean of CSV accuracy rows",
                        "source_csv": os.path.basename(fp32_csv(dataset, norm, exp)),
                    })
    return rows


def corruption_rows_blocked(reason, reference):
    by_key = {}
    for row in reference:
        by_key[(row["dataset"], row["norm"], int(row["experiment"]), int(row["k"]))] = row
    rows = []
    for dataset in ("65hd", "hyser"):
        for exp in EXPERIMENTS:
            for k in range(0, 8):
                channel = by_key[(dataset, "bychannel", exp, k)]
                sample = by_key[(dataset, "bysample", exp, k)]
                ch = float(channel["fp32_acc"])
                sa = float(sample["fp32_acc"])
                fp32_rank = "bychannel>=bysample" if ch + 1e-12 >= sa else "bysample>bychannel"
                for norm, src in (("bychannel", channel), ("bysample", sample)):
                    rows.append({
                        "dataset": dataset,
                        "norm": norm,
                        "experiment": exp,
                        "condition": "noise" if exp == 1 else "dropout",
                        "k": k,
                        "precision": "tflite_dynamic",
                        "fp32_acc": src["fp32_acc"],
                        "tflite_acc": "",
                        "delta_pp": "",
                        "fp32_rank": fp32_rank,
                        "tflite_rank": "",
                        "status": "BLOCKED",
                        "reason": reason,
                    })
    return rows


def gate_rows_blocked(reason):
    return [{
        "dataset": "65hd+hyser",
        "precision": "tflite_dynamic",
        "clean_max_abs_delta_pp": "",
        "noise_ranking_holds": "",
        "gate_pp": GATE_PP,
        "selected_k": SELECTED_K,
        "status": "BLOCKED",
        "reason": reason
        + " Gate not evaluated. Planned order when data is mounted: 65hd first "
        "(both norms, noise k=0..7 and dropout k=0..7), then hyser if the "
        "65hd clean delta stays within %.1f pp and the noise ranking "
        "(bychannel >= bysample for k=1..7) does not flip. INT8 is not run."
        % GATE_PP,
    }]


def load_latency(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paper_rows(reference, latency_rows, reason):
    """dataset x norm x precision, with TFLite accuracy left blank when blocked."""
    ref = {}
    for row in reference:
        if int(row["experiment"]) != 1:
            continue
        ref[(row["dataset"], row["norm"], int(row["k"]))] = row["fp32_acc"]
    latency = {}
    for row in latency_rows:
        latency[(row["dataset"], row["norm"], row["precision"])] = row
    precision_map = (
        ("keras_fp32", "keras_fp32", True),
        ("tflite_float", "tflite_float", False),
        ("tflite_dynamic", "tflite_dynamic_range", False),
    )
    machine = machine_label()
    rows = []
    for dataset in ("65hd", "hyser"):
        for norm in ("bychannel", "bysample"):
            for paper_name, latency_name, is_fp32 in precision_map:
                lat = latency.get((dataset, norm, latency_name), {})
                if is_fp32:
                    acc_clean = ref.get((dataset, norm, 0), "")
                    acc_k = ref.get((dataset, norm, SELECTED_K), "")
                    status = "fp32_from_existing_csv"
                    note = (
                        "Acc is the unweighted mean of the existing FP32 experiment_1 "
                        "CSV (clean k=0; selected k=%d noise). Not a new measurement. "
                        "size_mb is ResNet2D.h5 weights; SavedModel is larger because "
                        "it still holds AdamW slots. Latency is model(x, training=False) "
                        "on the testbench CPU."
                        % SELECTED_K
                    )
                else:
                    acc_clean = ""
                    acc_k = ""
                    status = "BLOCKED"
                    note = reason
                rows.append({
                    "dataset": dataset,
                    "norm": norm,
                    "precision": paper_name,
                    "acc_clean": acc_clean,
                    "acc_k": acc_k,
                    "k": SELECTED_K,
                    "k_condition": "noise_experiment_1",
                    "p50_ms": lat.get("p50_ms", ""),
                    "p95_ms": lat.get("p95_ms", ""),
                    "size_mb": lat.get("size_mb", ""),
                    "machine": machine,
                    "acc_status": status,
                    "notes": note,
                })
    return rows


PAPER_FIELDS = [
    "dataset", "norm", "precision", "acc_clean", "acc_k", "k", "k_condition",
    "p50_ms", "p95_ms", "size_mb", "machine", "acc_status", "notes",
]
REF_FIELDS = [
    "dataset", "norm", "experiment", "condition", "k", "fp32_acc", "n_rows",
    "aggregation", "source_csv",
]
CORRUPT_FIELDS = [
    "dataset", "norm", "experiment", "condition", "k", "precision",
    "fp32_acc", "tflite_acc", "delta_pp", "fp32_rank", "tflite_rank",
    "status", "reason",
]
GATE_FIELDS = [
    "dataset", "precision", "clean_max_abs_delta_pp", "noise_ranking_holds",
    "gate_pp", "selected_k", "status", "reason",
]


def write_blocked_bundle(out_dir, latency_csv):
    reason, _missing = blocked_reason()
    reference = fp32_reference_rows()
    write_csv(os.path.join(out_dir, "fp32_corruption_reference.csv"), REF_FIELDS, reference)
    corruption = corruption_rows_blocked(reason, reference)
    write_csv(os.path.join(out_dir, "corruption_accuracy.csv"), CORRUPT_FIELDS, corruption)
    gate = gate_rows_blocked(reason)
    write_csv(os.path.join(out_dir, "corruption_gate.csv"), GATE_FIELDS, gate)
    paper = paper_rows(reference, load_latency(latency_csv), reason)
    write_csv(os.path.join(out_dir, "quantization_paper_table.csv"), PAPER_FIELDS, paper)
    status_path = os.path.join(out_dir, "corruption_status.txt")
    with open(status_path, "w", encoding="utf-8") as handle:
        handle.write(reason + "\n")
        handle.write(
            "INT8 skipped: dynamic-range accuracy did not run, and there is no "
            "representative EMG set on this host.\n"
        )
    print(reason)
    print("wrote", os.path.join(out_dir, "corruption_accuracy.csv"))
    print("wrote", os.path.join(out_dir, "quantization_paper_table.csv"))
    return 0


def tflite_path_for(model_dir, precision):
    return model_dir + PRECISION_FILE[precision]


def make_tflite_infer(embedded):
    import numpy as np
    import tensorflow as tf

    def infer(batch_input):
        data = batch_input.numpy() if hasattr(batch_input, "numpy") else np.asarray(batch_input)
        rows = []
        for sample in data:
            embedded.set_input_tensor(sample)
            embedded.interpreter.invoke()
            output = embedded.interpreter.get_output_details()[0]
            rows.append(
                np.asarray(embedded.interpreter.get_tensor(output["index"]), dtype=np.float32).reshape(-1)
            )
        if not rows:
            return tf.zeros((0, 1), dtype=tf.float32)
        return tf.constant(np.stack(rows, axis=0), dtype=tf.float32)

    return infer


def files_matching_fp32(pipe, params, dataset, norm):
    """Same basename set as the FP32 experiment CSV. No new subjects."""
    frame = read_fp32(dataset, norm, 1)
    expected = []
    seen = set()
    for base in frame["file_base"].tolist():
        if base not in seen:
            seen.add(base)
            expected.append(base)
    discovered = pipe.build_files_list(params, PIPELINE_DIR, all_subjects=True)
    available = {}
    for path in discovered:
        available[file_base(path)] = path
    missing = [base for base in expected if base not in available]
    if missing:
        raise SystemExit(
            "FP32 CSV subjects are not on disk for %s/%s. Missing %d basenames, "
            "first: %s. Refusing to run a different cohort."
            % (dataset, norm, len(missing), missing[0])
        )
    return [available[base] for base in expected], expected


def run_one(dataset, norm, precision, experiments, batch_size, chunk_steps, out_dir, overwrite):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    if FUNCTIONS_DIR not in sys.path:
        sys.path.insert(0, FUNCTIONS_DIR)
    import tensorflow as tf
    import Pipeline_deployment_v2 as pipe
    from Embedded_Model import Embedded_Model

    tf.config.set_visible_devices([], "GPU")
    params = dict(pipe.DEFAULT_PARAMS)
    model_names = pipe.apply_dataset_profile(params, dataset)
    params["dataset_path"] = os.path.relpath(dataset_root(dataset), PIPELINE_DIR)
    frame = read_fp32(dataset, norm, 1)
    params["n_trials"] = majority_n_trials(frame)
    params["normalization_mode"] = norm
    params["channel_order_mode"] = str(frame["channel_order_mode"].iloc[0])
    params["corrupt_group"] = "" if params["channel_order_mode"] == "grid" else str(frame["corrupt_group"].iloc[0])

    model_name = model_names[norm]
    model_dir = os.path.join(TRAININGS_DIR, os.path.basename(model_name))
    written = []
    pending = []
    for experiment_id in experiments:
        output_path = os.path.join(
            out_dir,
            "corruption_runs",
            "%s_experiment_%d_%s_%s.csv" % (os.path.basename(model_name), experiment_id, dataset, precision),
        )
        reference = read_fp32(dataset, norm, experiment_id)
        need = int((reference["n_trials"] == params["n_trials"]).sum())
        have = 0 if overwrite else count_n_trials(output_path, params["n_trials"])
        if need > 0 and have >= need:
            print(
                "skip complete experiment_%d %s/%s rows=%d"
                % (experiment_id, dataset, norm, have),
                flush=True,
            )
            written.append(output_path)
        else:
            pending.append(experiment_id)
    if not pending:
        return written

    files, expected = files_matching_fp32(pipe, params, dataset, norm)
    print(
        "dataset=%s norm=%s precision=%s subjects=%d n_trials=%s pending=%s"
        % (dataset, norm, precision, len(expected), params["n_trials"], list(pending)),
        flush=True,
    )
    flatbuffer = tflite_path_for(model_dir, precision)
    if not os.path.isfile(flatbuffer):
        raise SystemExit("missing TFLite file: %s" % flatbuffer)

    model = tf.keras.models.load_model(model_dir, compile=False)
    # 4 threads is about 2x one thread on this CNN. 8 threads does not improve further,
    # and a batch-32 invoke was slightly slower than batch-1, so windows stay batch-1.
    embedded = Embedded_Model(model_dir, tflite_path=flatbuffer, num_threads=4)
    infer_fn = make_tflite_infer(embedded)
    configs = pipe.build_experiment_configs(
        files,
        gesture_list=list(params["gesture_list"]),
        normalization_mode=norm,
        num_channels=params["num_channels"],
    )
    for experiment_id in pending:
        config = configs[experiment_id]
        config["params"].update({
            "dataset_kind": params["dataset_kind"],
            "dataset_name": params["dataset_name"],
            "dataset_path": params["dataset_path"],
            "file_glob": params["file_glob"],
            "subject_filter": params["subject_filter"],
            "rest_repetition": params["rest_repetition"],
            "window_trim": params["window_trim"],
            "n_trials": params["n_trials"],
            "noise_seed_base": params["noise_seed_base"],
            "channel_order_mode": params["channel_order_mode"],
            "corrupt_group": params["corrupt_group"],
            "normalization_mode": norm,
        })
        # Keep the published sweep. Experiment 1 is noise k=0..7 across amplitudes;
        # experiment 3 is dropout k=0..7. Do not widen it here.
        output_path = os.path.join(
            out_dir,
            "corruption_runs",
            "%s_experiment_%d_%s_%s.csv" % (os.path.basename(model_name), experiment_id, dataset, precision),
        )
        pipe.run_experiment_block(
            experiment_id=experiment_id,
            model=model,
            model_name=model_name,
            files_to_run=config["files"],
            selection_builder=config["selection_builder"],
            params_template=config["params"],
            sweep_points=config["sweep_points"],
            batch_size=batch_size,
            chunk_steps=chunk_steps,
            output_suffix="_quant",
            pipeline_dir=PIPELINE_DIR,
            overwrite_csv=overwrite,
            infer_fn=infer_fn,
            output_path=output_path,
            reuse_clean=True,
        )
        written.append(output_path)
    del model
    del embedded
    tf.keras.backend.clear_session()
    return written


def compare_written(dataset, precision, out_dir):
    """Fill corruption rows for one dataset from TFLite CSVs. Others stay untouched."""
    import pandas as pd

    rows = []
    rank_at = {}
    for norm in ("bychannel", "bysample"):
        for exp in EXPERIMENTS:
            path = os.path.join(
                out_dir,
                "corruption_runs",
                "%s_experiment_%d_%s_%s.csv"
                % (MODEL_NAMES[(dataset, norm)], exp, dataset, precision),
            )
            if not os.path.isfile(path):
                raise SystemExit("expected TFLite CSV missing: %s" % path)
            quant = pd.read_csv(path)
            fp32 = read_fp32(dataset, norm, exp)
            n_trials = majority_n_trials(fp32)
            for k in range(0, 8):
                fp32_acc = mean_at_k(fp32, exp, k, n_trials=n_trials)
                quant_acc = mean_at_k(quant, exp, k, n_trials=n_trials)
                delta = None if quant_acc is None or fp32_acc is None else 100.0 * (quant_acc - fp32_acc)
                rank_at[(norm, exp, k)] = quant_acc
                rows.append({
                    "dataset": dataset,
                    "norm": norm,
                    "experiment": exp,
                    "k": k,
                    "fp32_acc": fp32_acc,
                    "tflite_acc": quant_acc,
                    "delta_pp": delta,
                })
    return rows, rank_at


def evaluate_gate(dataset, rank_at, detail_rows):
    clean_deltas = []
    for norm in ("bychannel", "bysample"):
        for row in detail_rows:
            if row["norm"] == norm and int(row["experiment"]) == 1 and int(row["k"]) == 0:
                clean_deltas.append(abs(float(row["delta_pp"])))
    ranking_ok = True
    for k in range(1, 8):
        channel = rank_at[("bychannel", 1, k)]
        sample = rank_at[("bysample", 1, k)]
        if channel is None or sample is None or channel + 1e-12 < sample:
            ranking_ok = False
    passed = bool(clean_deltas) and max(clean_deltas) <= GATE_PP and ranking_ok
    return {
        "dataset": dataset,
        "precision": "tflite_dynamic",
        "clean_max_abs_delta_pp": fmt(max(clean_deltas) if clean_deltas else None, 4),
        "noise_ranking_holds": "yes" if ranking_ok else "no",
        "gate_pp": GATE_PP,
        "selected_k": SELECTED_K,
        "status": "PASS" if passed else "FAIL",
        "reason": (
            "Continuous Acc is experiment_1 k=0, unweighted row mean, both norms. "
            "Noise ranking is bychannel >= bysample at every k=1..7. "
            "Dropout is reported in corruption_accuracy.csv and is not this gate. "
            "INT8 was not run."
        ),
    }


def read_table(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dataset_ready(dataset, precision, out_dir):
    for norm in ("bychannel", "bysample"):
        n_trials = majority_n_trials(read_fp32(dataset, norm, 1))
        for exp in EXPERIMENTS:
            reference = read_fp32(dataset, norm, exp)
            need = int((reference["n_trials"] == n_trials).sum())
            path = os.path.join(
                out_dir,
                "corruption_runs",
                "%s_experiment_%d_%s_%s.csv" % (MODEL_NAMES[(dataset, norm)], exp, dataset, precision),
            )
            if count_n_trials(path, n_trials) < need:
                return False
    return True


def refresh_paper_from_results(out_dir, latency_csv, quant_detail):
    """quant_detail maps (dataset, norm, exp, k) -> acc. Empty means still blocked."""
    reference = fp32_reference_rows()
    reason, _missing = blocked_reason()
    rows = paper_rows(reference, load_latency(latency_csv), reason if reason else "")
    measured_datasets = {key[0] for key in quant_detail}
    previous = {
        (row["dataset"], row["norm"], row["precision"]): row
        for row in read_table(os.path.join(out_dir, "quantization_paper_table.csv"))
    }
    for row in rows:
        if row["dataset"] in measured_datasets:
            continue
        prev = previous.get((row["dataset"], row["norm"], row["precision"]))
        if prev and prev.get("acc_status") == "measured_tflite_dynamic":
            row["acc_clean"] = prev.get("acc_clean", "")
            row["acc_k"] = prev.get("acc_k", "")
            row["acc_status"] = prev["acc_status"]
            row["notes"] = prev.get("notes", "")
    if quant_detail:
        for row in rows:
            if row["precision"] != "tflite_dynamic":
                continue
            clean = quant_detail.get((row["dataset"], row["norm"], 1, 0))
            at_k = quant_detail.get((row["dataset"], row["norm"], 1, SELECTED_K))
            if clean is None:
                continue
            row["acc_clean"] = fmt(clean)
            row["acc_k"] = fmt(at_k)
            row["acc_status"] = "measured_tflite_dynamic"
            row["notes"] = (
                "TFLite dynamic-range accuracy, same deployment preprocess as "
                "Pipeline_deployment_v2, subject basenames matched to the FP32 CSV. "
                "acc_clean is noise k=0; acc_k is noise k=%d." % SELECTED_K
            )
    write_csv(os.path.join(out_dir, "quantization_paper_table.csv"), PAPER_FIELDS, rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="65hd", choices=["65hd", "hyser", "both"])
    parser.add_argument("--norm", default="both", choices=["both", "bychannel", "bysample"])
    parser.add_argument("--precision", default="tflite_dynamic", choices=sorted(PRECISION_FILE))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-steps", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--latency-csv",
        default=os.path.join(DEFAULT_OUT, "quantization_latency.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    missing = [name for name in ("65hd", "hyser") if not os.path.isdir(dataset_root(name))]
    requested = ["65hd", "hyser"] if args.dataset == "both" else [args.dataset]
    if all(name in missing for name in requested):
        return write_blocked_bundle(args.out_dir, args.latency_csv)

    if args.precision != "tflite_dynamic":
        print(
            "Running %s for latency-control accuracy. The acceptance gate in this "
            "script is defined for tflite_dynamic." % args.precision,
            flush=True,
        )
    datasets = [name for name in requested if name not in missing]
    if missing:
        print("Not mounted, left BLOCKED:", ", ".join(missing), flush=True)
    if args.dataset == "both":
        print(
            "Full grid requested. Prefer 65hd alone first; pass --dataset hyser "
            "only after that delta is within %.1f pp." % GATE_PP,
            flush=True,
        )
    norms = ("bychannel", "bysample") if args.norm == "both" else (args.norm,)
    written_detail = []
    for dataset in datasets:
        for norm in norms:
            run_one(
                dataset,
                norm,
                args.precision,
                EXPERIMENTS,
                args.batch_size,
                args.chunk_steps,
                args.out_dir,
                args.overwrite,
            )
        if args.precision != "tflite_dynamic":
            continue
        if not dataset_ready(dataset, args.precision, args.out_dir):
            print(
                "dataset %s still incomplete; comparison tables left unchanged" % dataset,
                flush=True,
            )
            continue
        detail, rank_at = compare_written(dataset, args.precision, args.out_dir)
        written_detail.append((dataset, detail, rank_at))
    if not written_detail:
        return 0

    # Rebuild the comparison table: measured datasets plus BLOCKED for the rest.
    reference = fp32_reference_rows()
    write_csv(os.path.join(args.out_dir, "fp32_corruption_reference.csv"), REF_FIELDS, reference)
    blocked = corruption_rows_blocked(
        "BLOCKED: this dataset was not in this run. Start with 65hd; run hyser "
        "only if the 65hd clean delta stays within %.1f pp and the noise ranking holds."
        % GATE_PP,
        reference,
    )
    ran = {dataset for dataset, _detail, _rank in written_detail}
    old_measured = [
        row for row in read_table(os.path.join(args.out_dir, "corruption_accuracy.csv"))
        if row.get("status") == "measured" and row.get("dataset") not in ran
    ]
    measured_keys = {
        (row["dataset"], row["norm"], int(row["experiment"]), int(row["k"]))
        for row in old_measured
    }
    quant_detail = {}
    gate = [
        row for row in read_table(os.path.join(args.out_dir, "corruption_gate.csv"))
        if row.get("dataset") not in ran
    ]
    for dataset, detail, rank_at in written_detail:
        gate.append(evaluate_gate(dataset, rank_at, detail))
        for row in detail:
            measured_keys.add((dataset, row["norm"], int(row["experiment"]), int(row["k"])))
            quant_detail[(dataset, row["norm"], int(row["experiment"]), int(row["k"]))] = row["tflite_acc"]
    merged = list(old_measured)
    for row in blocked:
        key = (row["dataset"], row["norm"], int(row["experiment"]), int(row["k"]))
        if key in measured_keys:
            continue
        merged.append(row)
    for dataset, detail, rank_at in written_detail:
        for row in detail:
            ch = rank_at[("bychannel", int(row["experiment"]), int(row["k"]))]
            sa = rank_at[("bysample", int(row["experiment"]), int(row["k"]))]
            if ch is None or sa is None:
                q_rank = ""
            elif ch + 1e-12 >= sa:
                q_rank = "bychannel>=bysample"
            else:
                q_rank = "bysample>bychannel"
            merged.append({
                "dataset": dataset,
                "norm": row["norm"],
                "experiment": row["experiment"],
                "condition": "noise" if int(row["experiment"]) == 1 else "dropout",
                "k": row["k"],
                "precision": "tflite_dynamic",
                "fp32_acc": fmt(row["fp32_acc"]),
                "tflite_acc": fmt(row["tflite_acc"]),
                "delta_pp": fmt(row["delta_pp"], 4),
                "fp32_rank": "",
                "tflite_rank": q_rank,
                "status": "measured",
                "reason": "",
            })
    # Fill fp32_rank on measured rows from the reference means.
    ref_acc = {}
    for row in reference:
        ref_acc[(row["dataset"], row["norm"], int(row["experiment"]), int(row["k"]))] = float(row["fp32_acc"])
    for row in merged:
        if row["status"] != "measured":
            continue
        ch = ref_acc[(row["dataset"], "bychannel", int(row["experiment"]), int(row["k"]))]
        sa = ref_acc[(row["dataset"], "bysample", int(row["experiment"]), int(row["k"]))]
        row["fp32_rank"] = "bychannel>=bysample" if ch + 1e-12 >= sa else "bysample>bychannel"
    write_csv(os.path.join(args.out_dir, "corruption_accuracy.csv"), CORRUPT_FIELDS, merged)
    if not gate:
        gate = gate_rows_blocked("No tflite_dynamic comparison rows were produced.")
    write_csv(os.path.join(args.out_dir, "corruption_gate.csv"), GATE_FIELDS, gate)
    refresh_paper_from_results(args.out_dir, args.latency_csv, quant_detail)
    for row in gate:
        print("GATE", row["dataset"], row["status"], "clean_max_abs_delta_pp", row["clean_max_abs_delta_pp"])
    print("INT8 not run. It stays out of scope until dynamic-range Acc passes this gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
