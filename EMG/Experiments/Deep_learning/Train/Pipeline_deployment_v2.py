#!/usr/bin/env python
# coding: utf-8
"""
Deployment / robustness pipeline v2.

Merges:
  - Pipeline_deployment_loop_experiments.py  (batched inference, CLI, multi-subject CSV)
  - Pipeline_sess1-65HD-armband-Noisy-oneGesture-unified.ipynb  (additive noise,
    differential 65HD fault model, dead-electrode mode, channel_order grid, n_trials)

Supports raw datasets via --dataset:
  - 65hd  -> $CBEB_DATA_ROOT/65_HDsEMG  (or <package>/data/65_HDsEMG)
  - hyser -> $CBEB_DATA_ROOT/HYSER/files/hd-semg/1.0.0/pr_dataset
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PIPELINE_DIR = SCRIPT_DIR
PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
DATA_ROOT = os.path.abspath(os.environ.get("CBEB_DATA_ROOT", os.path.join(PACKAGE_ROOT, "data")))
HYSER_CODE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../DataBase_Treatment/HYSER/pr_dataset"))
PROCESSING_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../Processing"))

# Gesture ids as stored in each raw dataset (class remapping uses np.sort internally).
GESTURE_LIST_65HD = [0, 10, 8, 6, 4, 2, 13, 14, 15, 16, 61, 58]
GESTURE_LIST_HYSER = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 30, 31]

DATASET_PROFILES = {
    "65hd": {
        "dataset_kind": "65hd",
        "dataset_path": os.path.join(DATA_ROOT, "65_HDsEMG"),
        "dataset_name": "65_HDsEMG",
        "gesture_list": GESTURE_LIST_65HD,
        "file_glob": "2*",
        # Legacy 65HD pipeline skipped the first sorted subject (index 0).
        "subject_filter": list(range(1, 17)),
        "model_names": {
            "bychannel": "trainings/EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_bychannel_adamW",
            "bysample": "trainings/EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_bysample_adamW",
        },
    },
    "hyser": {
        "dataset_kind": "hyser",
        "dataset_path": os.path.join(DATA_ROOT, "HYSER", "files", "hd-semg", "1.0.0", "pr_dataset"),
        "dataset_name": "Hyser",
        "gesture_list": GESTURE_LIST_HYSER,
        "file_glob": "subject*_session*",
        "subject_filter": None,
        "rest_repetition": 1,
        "window_trim": 128,
        "model_names": {
            "bychannel": "trainings/EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_bychannel_adamW",
            "bysample": "trainings/EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_bysample_adamW",
        },
    },
}

DEFAULT_DATASET = "65hd"
DEFAULT_PROFILE = DATASET_PROFILES[DEFAULT_DATASET]

DEFAULT_PARAMS = {
    "dataset_kind": DEFAULT_PROFILE["dataset_kind"],
    "dataset_path": DEFAULT_PROFILE["dataset_path"],
    "dataset_name": DEFAULT_PROFILE["dataset_name"],
    "window": 128,
    "step": 32,
    "num_channels": 128,
    "num_classes": len(DEFAULT_PROFILE["gesture_list"]),
    "num_samples": 256,
    "num_windows": 10,
    "img_height": 8,
    "img_width": 16,
    "gesture_list": list(DEFAULT_PROFILE["gesture_list"]),
    "file_glob": DEFAULT_PROFILE["file_glob"],
    "subject_filter": DEFAULT_PROFILE.get("subject_filter"),
    "rest_repetition": DEFAULT_PROFILE.get("rest_repetition", 1),
    "window_trim": DEFAULT_PROFILE.get("window_trim", 128),
    "n_voting": 3,
    "armband": False,
    "MAFilter": 32,
    "noise_seed_base": 0,
    "n_trials": 3,
    "channel_order_mode": "grid",
    "corrupt_group": "interleave",
}

FAULT_MODES = ("none", "noise", "dead")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_index_list(raw_values):
    if raw_values is None:
        return None

    parsed_values = []
    for raw_value in raw_values:
        for chunk in str(raw_value).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start_str, end_str = chunk.split("-", 1)
                start_idx = int(start_str)
                end_idx = int(end_str)
                step = 1 if end_idx >= start_idx else -1
                parsed_values.extend(list(range(start_idx, end_idx + step, step)))
            else:
                parsed_values.append(int(chunk))
    return parsed_values


def select_by_indices(values, selected_indices, label):
    if selected_indices is None:
        return list(values)

    selected_values = []
    for idx in selected_indices:
        if idx < 0 or idx >= len(values):
            raise ValueError(f"{label} index {idx} is out of range for {len(values)} available items.")
        selected_values.append(values[idx])
    return selected_values


def load_gesture_list_from_json(json_path, id_field="hdemg_id"):
    with open(json_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return sorted(int(entry[id_field]) for entry in payload["gesture_mapping"])


def apply_dataset_profile(params, dataset_key, dataset_path_override=None):
    if dataset_key not in DATASET_PROFILES:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASET_PROFILES)}")

    profile = DATASET_PROFILES[dataset_key]
    params["dataset_kind"] = profile["dataset_kind"]
    params["dataset_path"] = dataset_path_override or profile["dataset_path"]
    params["dataset_name"] = profile["dataset_name"]
    params["gesture_list"] = list(profile["gesture_list"])
    params["num_classes"] = len(profile["gesture_list"])
    params["file_glob"] = profile["file_glob"]
    params["subject_filter"] = profile.get("subject_filter")
    params["rest_repetition"] = profile.get("rest_repetition", 1)
    params["window_trim"] = profile.get("window_trim", 128)
    return profile["model_names"]


def configure_gpu(gpu_index=None):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected. Running on CPU.")
        return

    try:
        if gpu_index is not None:
            if gpu_index < 0 or gpu_index >= len(gpus):
                raise ValueError(
                    f"Requested GPU index {gpu_index} is out of range. "
                    f"{len(gpus)} physical GPU(s) available."
                )
            tf.config.set_visible_devices(gpus[gpu_index], "GPU")
            gpus = tf.config.list_physical_devices("GPU")
            print("selected gpu:", gpu_index, gpus[0].name)

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as exc:
        print(exc)


def resolve_from_pipeline_dir(pipeline_dir, relative_path):
    return os.path.abspath(os.path.join(pipeline_dir, relative_path))


def discover_dataset_files(params, pipeline_dir):
    dataset_path = resolve_from_pipeline_dir(pipeline_dir, params["dataset_path"])
    pattern = params.get("file_glob", "2*")
    files_list = glob.glob(os.path.join(dataset_path, pattern))
    files_list = sorted({path.rstrip(os.sep) for path in files_list if os.path.exists(path)})

    if params.get("dataset_kind") == "hyser":
        files_list = [path for path in files_list if os.path.isdir(path)]
        files_list.sort()

    if not files_list:
        raise RuntimeError(
            f"No files matched pattern '{pattern}' under resolved dataset path '{dataset_path}'."
        )
    return files_list


def build_files_list(params, pipeline_dir, subject_indices=None, all_subjects=False):
    files_list = discover_dataset_files(params, pipeline_dir)

    subject_filter = params.get("subject_filter")
    if subject_filter is not None and not all_subjects:
        missing = [idx for idx in subject_filter if idx < 0 or idx >= len(files_list)]
        if missing:
            raise RuntimeError(
                f"subject_filter indices {missing} are out of range for {len(files_list)} discovered files."
            )
        files_list = [files_list[idx] for idx in subject_filter]

    return select_by_indices(files_list, subject_indices, "Subject")


# ---------------------------------------------------------------------------
# Noise / fault (unified notebook)
# ---------------------------------------------------------------------------


def fftnoise(frequencies, rng):
    frequencies = np.array(frequencies, dtype="complex")
    num_positive = (len(frequencies) - 1) // 2
    phases = rng.uniform(0, 2 * np.pi, num_positive)
    phases = np.cos(phases) + 1j * np.sin(phases)
    frequencies[1 : num_positive + 1] *= phases
    frequencies[-1 : -1 - num_positive : -1] = np.conj(frequencies[1 : num_positive + 1])
    return np.fft.ifft(frequencies).real


def band_limited_noise(min_freq, max_freq, samples, samplerate, rng):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    frequencies = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    frequencies[idx] = 1
    return fftnoise(frequencies, rng)


def additive_noise(channel_data, max_var, rng, min_freq=10, max_freq=900, samplerate=2048):
    data_len = channel_data.shape[-1]
    span = np.max(channel_data) - np.min(channel_data)
    noise = band_limited_noise(min_freq, max_freq, data_len, samplerate, rng)
    noise = noise - noise.mean()
    peak = np.max(np.abs(noise)) + sys.float_info.epsilon
    return noise / peak * (span * max_var / 2)


def build_channel_order(num_channels, img_height, img_width, shuffle_seed=None):
    grid_size = img_height * img_width
    if grid_size < num_channels:
        raise ValueError(
            f"grid {img_height}x{img_width}={grid_size} smaller than num_channels={num_channels}"
        )
    order = np.arange(grid_size).reshape(img_height, img_width).flatten(order="F")
    order = order[order < num_channels]
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    return order.astype(np.int32)


def build_channel_index_map():
    index1 = np.append(np.arange(8), np.arange(8))
    index1 = np.append(index1, index1)
    index1 = np.append(index1, index1)

    index2 = np.repeat(np.arange(8), 4)
    index2 = np.append(index2, np.roll(index2, -4))
    return index1, index2


def build_legacy_channel_order(base_index_map, corrupt_group):
    extensor_map = [int(idx) for idx in base_index_map]
    flexor_map = [int(idx) + 64 for idx in base_index_map]

    if corrupt_group == "extensor":
        return np.array(extensor_map, dtype=np.int32)
    if corrupt_group == "flexor":
        return np.array(flexor_map, dtype=np.int32)
    if corrupt_group == "both":
        return np.array(extensor_map + flexor_map, dtype=np.int32)
    if corrupt_group == "interleave":
        mixed_map = []
        max_len = max(len(extensor_map), len(flexor_map))
        for idx in range(max_len):
            if idx < len(extensor_map):
                mixed_map.append(extensor_map[idx])
            if idx < len(flexor_map):
                mixed_map.append(flexor_map[idx])
        return np.array(mixed_map, dtype=np.int32)

    raise ValueError(f"Unsupported corrupt_group: {corrupt_group}")


def resolve_channel_order(num_channels, params):
    if params.get("channel_order_mode") == "grid":
        return build_channel_order(
            num_channels,
            params["img_height"],
            params["img_width"],
            params.get("channel_shuffle_seed"),
        )

    index1, index2 = build_channel_index_map()
    base_index_map = (index1 + index2 * 8).astype(np.int32)
    return build_legacy_channel_order(base_index_map, params["corrupt_group"])


def apply_electrode_noise(data, channel_order, n_channels, max_var, rng, dataset_name):
    data = data.copy()
    if dataset_name == "65_HDsEMG":
        for nn in range(n_channels):
            ch = int(channel_order[nn])
            noise = additive_noise(data[ch, :], max_var, rng)
            data[ch, :] = data[ch, :] + noise
            if ch + 1 < data.shape[0]:
                data[ch + 1, :] = data[ch + 1, :] - noise
    else:
        for nn in range(n_channels):
            ch = int(channel_order[nn])
            data[ch, :] = data[ch, :] + additive_noise(data[ch, :], max_var, rng)
    return data


def apply_dead_electrodes(data, channel_order, n_channels, dataset_name):
    data = data.copy()
    if dataset_name == "65_HDsEMG":
        mono = np.cumsum(data, axis=0)
        for nn in range(n_channels):
            mono[int(channel_order[nn]), :] = 0
        data = np.diff(np.vstack([np.zeros((1, mono.shape[1]), dtype=mono.dtype), mono]), axis=0)
    else:
        for nn in range(n_channels):
            data[int(channel_order[nn]), :] = 0
    return data


def apply_corruption(base_data, channel_order, n_channels, max_var, fault_mode, rng, dataset_name):
    if fault_mode == "none" or n_channels <= 0:
        return base_data.copy()
    if fault_mode == "noise":
        return apply_electrode_noise(
            base_data, channel_order, n_channels, max_var, rng, dataset_name
        )
    if fault_mode == "dead":
        return apply_dead_electrodes(base_data, channel_order, n_channels, dataset_name)
    raise ValueError(f"unknown fault_mode: {fault_mode}")


def format_channel_label(channel_idx):
    channel_idx = int(channel_idx)
    if channel_idx < 64:
        return f"E{channel_idx:02d}"
    return f"F{channel_idx - 64:02d}"


def describe_channels(channel_indices):
    channel_indices = [int(idx) for idx in channel_indices]
    if not channel_indices:
        return "none"
    formatted = [f"{format_channel_label(idx)}[{idx}]" for idx in channel_indices]
    return ", ".join(formatted)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def ensure_hyser_import_paths():
    for path in (HYSER_CODE_PATH, PROCESSING_PATH):
        if path not in sys.path:
            sys.path.insert(0, path)


def get_rest_sequence(sequence, rest_repetition):
    best_start, best_end = 0, 0
    curr_start = 0
    prev_val = sequence[0]
    repetition = 0
    find_sequence = False

    for idx, curr_val in enumerate(sequence[1:], start=1):
        if prev_val:
            find_sequence = True
            best_start = curr_start
            best_end = idx
        else:
            curr_start = idx
            if find_sequence:
                repetition += 1
                find_sequence = False
                if repetition == rest_repetition:
                    break
        prev_val = curr_val

    if repetition != rest_repetition and bool(sequence[-1]) and (repetition + 1) == rest_repetition:
        best_end = len(sequence)

    return best_start, best_end


def extract_rest_emg(emg, active_idx, rest_repetition):
    is_rest = np.ones(emg.shape[1], dtype=bool)
    if len(active_idx) > 0:
        is_rest[active_idx] = False
    idx_ini, idx_end = get_rest_sequence(is_rest, rest_repetition)
    return emg[:, idx_ini:idx_end]


def trim_hyser_emg(emg, window_size, extract_rest=False, rest_repetition=1):
    import preprocessing as prep

    active_idx = prep.removing_noise(emg, window_size)
    emg_active = prep.get_emg_from_index(emg, active_idx, remove_outliers=False)
    if not extract_rest:
        return emg_active, None
    emg_rest = extract_rest_emg(emg, active_idx, rest_repetition)
    return emg_active, emg_rest


def load_hyser_subject(subject_path, params):
    ensure_hyser_import_paths()
    from HYSERSubject import Subject

    subject_path = subject_path.rstrip(os.sep)
    labels_dynamic = os.path.join(subject_path, "label_dynamic.txt")
    labels_maintenance = os.path.join(subject_path, "label_maintenance.txt")
    if not os.path.isdir(subject_path):
        raise FileNotFoundError(f"HYSER session folder not found: {subject_path}")

    file_paths = sorted(
        {
            os.path.join(subject_path, item.split(".")[0])
            for item in os.listdir(subject_path)
            if "preprocess" in item
        }
    )
    if not file_paths:
        raise RuntimeError(f"No preprocess WFDB records found under {subject_path}")

    subject_name = os.path.basename(subject_path).split("_")[0].replace("subject", "")
    subject = Subject(subject_name=subject_name)
    subject.loadLabels(labels_dynamic, labels_maintenance)
    subject.loadMultipleSamples(file_paths)

    gesture_list = [int(g) for g in params["gesture_list"]]
    extract_rest = 0 in gesture_list
    real_gestures = {int(g) for g in gesture_list if int(g) != 0}
    window_size = int(params.get("window_trim", 128))
    rest_repetition = int(params.get("rest_repetition", 1))

    segments = []
    for sample in subject.samples:
        movement_id = int(sample.getMovement())
        if movement_id not in real_gestures:
            continue

        emg = np.concatenate(
            (sample.getEPSignalData().T, sample.getFPSignalData().T),
            axis=0,
        ).astype(np.float32)

        if extract_rest:
            emg_active, emg_rest = trim_hyser_emg(
                emg, window_size, extract_rest=True, rest_repetition=rest_repetition
            )
        else:
            emg_active, emg_rest = trim_hyser_emg(emg, window_size, extract_rest=False)
            emg_rest = None

        if emg_active.shape[1] > 0:
            segments.append((emg_active, movement_id))
        if extract_rest and emg_rest is not None and emg_rest.shape[1] > 0:
            segments.append((emg_rest, 0))

    if not segments:
        raise RuntimeError(f"No HYSER segments matched gesture_list under {subject_path}")

    emg_data = np.concatenate([segment[0] for segment in segments], axis=1)
    labels = np.concatenate(
        [np.full(segment[0].shape[1], segment[1], dtype=np.int32) for segment in segments]
    ).reshape(1, -1)
    return emg_data, labels


def load_65hd_subject(file_path, armband):
    with h5py.File(file_path, "r") as h5_file:
        extensor = h5_file["emg_extensors"][...].reshape(64, -1).astype(np.float32)
        flexor = h5_file["emg_flexors"][...].reshape(64, -1).astype(np.float32)
        labels = h5_file["adjusted_class"][...]

    full_emg_data = np.concatenate((extensor, flexor), axis=0)

    if armband:
        idx = [
            np.arange(0, 121, 8),
            np.arange(1, 58, 8),
            np.arange(2, 59, 8),
            np.arange(5, 62, 8),
            np.arange(6, 63, 8),
            np.arange(7, 129, 8),
        ]
        idx = np.hstack(idx)
        emg_data = full_emg_data.copy()
        emg_data[idx, :] = 0
    else:
        emg_data = full_emg_data

    return emg_data, labels


def load_subject_data(file_path, params):
    if params.get("dataset_kind") == "hyser":
        return load_hyser_subject(file_path, params)
    return load_65hd_subject(file_path, params["armband"])


def select_gesture_data(emg_data, labels, gesture_values, all_gestures):
    idx_gestures = []
    for gesture in gesture_values:
        idx_gestures.append(np.argwhere(labels[0, :] == gesture))

    stacked_idx = np.vstack(idx_gestures)[:, 0]
    selected_data = emg_data[:, stacked_idx]
    selected_labels = labels[:, stacked_idx][0]

    all_gestures_sorted = np.sort(all_gestures)
    gesture_remap = {gesture: new_gesture for new_gesture, gesture in enumerate(all_gestures_sorted)}
    y_data = np.array([gesture_remap[gesture] for gesture in selected_labels], dtype=np.int32)
    return selected_data, selected_labels, y_data


# ---------------------------------------------------------------------------
# Fast streaming inference
# ---------------------------------------------------------------------------


def build_prefix_counts(y_data):
    num_classes = int(np.max(y_data)) + 1
    prefix_counts = np.zeros((num_classes, len(y_data) + 1), dtype=np.int32)
    for class_idx in range(num_classes):
        prefix_counts[class_idx, 1:] = np.cumsum(y_data == class_idx, dtype=np.int32)
    return prefix_counts


def compute_true_labels(prefix_counts, starts, window_size, total_labels):
    ends = np.minimum(starts + window_size, total_labels)
    counts = prefix_counts[:, ends] - prefix_counts[:, starts]
    return np.argmax(counts, axis=0).astype(np.int32)


def majority_vote(raw_predictions, n_voting):
    voted_predictions = []
    y_cnn = np.zeros((n_voting,), dtype=np.int32)

    for pred in raw_predictions:
        y_cnn = np.roll(y_cnn, -1)
        y_cnn[-1] = pred

        if len(voted_predictions) < n_voting:
            voted_predictions.append(int(y_cnn[-1]))
            continue

        values, count = np.unique(y_cnn, return_counts=True)
        if len(count) == n_voting:
            voted_predictions.append(voted_predictions[-1])
        else:
            voted_predictions.append(int(values[np.argmax(count)]))

    return np.asarray(voted_predictions, dtype=np.int32)


def normalize_windows(batch_windows, normalization_mode):
    axis_map = {"bychannel": 2, "bysample": 1}
    normalization_axis = axis_map[normalization_mode]
    samples_min = batch_windows.min(axis=normalization_axis, keepdims=True)
    normalized = batch_windows - samples_min
    samples_max = normalized.max(axis=normalization_axis, keepdims=True) + sys.float_info.epsilon
    return (normalized / samples_max).astype(np.float32, copy=False)


def compute_processed_stream_chunk(raw_chunk, prev_rect_block, step_size, ma_filter):
    num_channels = raw_chunk.shape[0]
    num_steps = raw_chunk.shape[1] // step_size

    rectified = np.abs(raw_chunk.reshape(num_channels, num_steps, step_size))
    prev_blocks = np.concatenate((prev_rect_block[:, None, :], rectified[:, :-1, :]), axis=1)
    paired = np.concatenate((prev_blocks, rectified), axis=2)

    tke = paired[:, :, 1:-1] * paired[:, :, 1:-1] - paired[:, :, :-2] * paired[:, :, 2:]
    cumsum = np.cumsum(tke, axis=2, dtype=float)
    cumsum[:, :, ma_filter:] = cumsum[:, :, ma_filter:] - cumsum[:, :, :-ma_filter]
    filtered = cumsum[:, :, ma_filter - 1 :] / ma_filter

    next_prev_rect = rectified[:, -1, :].copy()
    stream_chunk = filtered.reshape(num_channels, -1)
    return stream_chunk, next_prev_rect


def build_window_batch(stream_chunk, history_tail, num_samples, processed_width):
    combined_stream = np.concatenate((history_tail, stream_chunk), axis=1)
    windows = sliding_window_view(combined_stream, num_samples, axis=1)
    windows = windows[:, ::processed_width, :]
    num_steps = stream_chunk.shape[1] // processed_width
    windows = windows[:, :num_steps, :]
    next_tail = combined_stream[:, -history_tail.shape[1] :].copy()
    return windows.transpose(1, 0, 2), next_tail


def build_infer_fn(model):
    @tf.function(reduce_retracing=True)
    def infer(batch_input):
        return model(batch_input, training=False)

    return infer


def run_inference_batches(infer_fn, window_batch, batch_size, params):
    raw_predictions = []
    img_width = params["img_width"]
    img_height = params["img_height"]

    for batch_start in range(0, len(window_batch), batch_size):
        batch_end = min(batch_start + batch_size, len(window_batch))
        normalized = normalize_windows(window_batch[batch_start:batch_end], params["normalization_mode"])
        normalized = normalized.reshape(-1, img_width, img_height, params["num_samples"])
        logits = infer_fn(tf.convert_to_tensor(normalized, dtype=tf.float32))
        batch_pred = np.argmax(logits.numpy(), axis=1).astype(np.int32)
        raw_predictions.append(batch_pred)

    if not raw_predictions:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(raw_predictions, axis=0)


def measure_first_two_latencies(stream_data, infer_fn, params):
    num_samples = params["num_samples"]
    processed_width = params["step"] - 1
    step = params["step"]
    img_width = params["img_width"]
    img_height = params["img_height"]
    ma_filter = params["MAFilter"]

    num_windows = (stream_data.shape[1] - step) // step
    if num_windows <= 0:
        return {"first_bacth": 0.0, "second_bacth": 0.0}

    history_tail = np.zeros((stream_data.shape[0], num_samples - processed_width), dtype=float)
    prev_rect_block = np.zeros((stream_data.shape[0], step), dtype=np.float32)

    t_init = time.time()
    raw_chunk = stream_data[:, : min(2, num_windows) * step]
    stream_chunk, _ = compute_processed_stream_chunk(raw_chunk, prev_rect_block, step, ma_filter)
    first_two_windows, _ = build_window_batch(stream_chunk, history_tail, num_samples, processed_width)

    first_window = normalize_windows(first_two_windows[:1], params["normalization_mode"]).reshape(
        1, img_width, img_height, num_samples
    )
    infer_fn(tf.convert_to_tensor(first_window, dtype=tf.float32))
    first_latency = time.time() - t_init

    second_latency = 0.0
    if num_windows > 1:
        t_second = time.time()
        second_window = normalize_windows(first_two_windows[1:2], params["normalization_mode"]).reshape(
            1, img_width, img_height, num_samples
        )
        infer_fn(tf.convert_to_tensor(second_window, dtype=tf.float32))
        second_latency = time.time() - t_second

    return {"first_bacth": first_latency, "second_bacth": second_latency}


def predict_gesture_fast(model, stream_data, y_data, params, batch_size, chunk_steps, infer_fn=None):
    if infer_fn is None:
        infer_fn = build_infer_fn(model)

    num_windows = (stream_data.shape[1] - params["step"]) // params["step"]
    starts = np.arange(0, num_windows * params["step"], params["step"], dtype=np.int32)
    prefix_counts = build_prefix_counts(y_data)
    latency_times = measure_first_two_latencies(stream_data, infer_fn, params)

    processed_width = params["step"] - 1
    history_tail = np.zeros((stream_data.shape[0], params["num_samples"] - processed_width), dtype=float)
    prev_rect_block = np.zeros((stream_data.shape[0], params["step"]), dtype=np.float32)
    raw_predictions = []
    true_labels = []

    t_total = time.time()

    for chunk_start in range(0, num_windows, chunk_steps):
        chunk_end = min(chunk_start + chunk_steps, num_windows)
        start_col = chunk_start * params["step"]
        end_col = chunk_end * params["step"]

        raw_chunk = stream_data[:, start_col:end_col]
        stream_chunk, prev_rect_block = compute_processed_stream_chunk(
            raw_chunk,
            prev_rect_block,
            params["step"],
            params["MAFilter"],
        )

        window_batch, history_tail = build_window_batch(
            stream_chunk,
            history_tail,
            params["num_samples"],
            processed_width,
        )

        raw_predictions.append(run_inference_batches(infer_fn, window_batch, batch_size, params))
        true_labels.append(
            compute_true_labels(
                prefix_counts,
                starts[chunk_start:chunk_end],
                params["window"],
                len(y_data),
            )
        )

    if raw_predictions:
        raw_predictions = np.concatenate(raw_predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
    else:
        raw_predictions = np.empty((0,), dtype=np.int32)
        true_labels = np.empty((0,), dtype=np.int32)

    latency_times["mean"] = (time.time() - t_total) / max(len(raw_predictions), 1)
    voted_predictions = majority_vote(raw_predictions, params["n_voting"])
    return voted_predictions, true_labels, latency_times


def predict_with_trials(
    model,
    base_data,
    y_data,
    params,
    channel_order,
    n_channels,
    max_var,
    fault_mode,
    batch_size,
    chunk_steps,
    infer_fn=None,
):
    n_trials = int(params.get("n_trials", 1))
    seed_base = int(params.get("noise_seed_base", 0))
    trial_results = []

    if infer_fn is None:
        infer_fn = build_infer_fn(model)

    for trial in range(n_trials):
        rng = np.random.default_rng(seed_base + trial)
        corrupted = apply_corruption(
            base_data,
            channel_order,
            n_channels,
            max_var,
            fault_mode,
            rng,
            params["dataset_name"],
        )
        y_pred, y_true, latency_times = predict_gesture_fast(
            model=model,
            stream_data=corrupted,
            y_data=y_data,
            params=params,
            batch_size=batch_size,
            chunk_steps=chunk_steps,
            infer_fn=infer_fn,
        )
        trial_results.append((y_pred, y_true, latency_times))

    trial_accuracies = np.array(
        [float(skm.accuracy_score(y_true, y_pred)) for y_pred, y_true, _ in trial_results],
        dtype=np.float64,
    )
    y_pred, y_true, latency_times = trial_results[-1]
    return y_pred, y_true, latency_times, trial_accuracies, trial_results


# ---------------------------------------------------------------------------
# Selection builders
# ---------------------------------------------------------------------------


def describe_gesture_scope(gesture_values):
    if len(gesture_values) == 1:
        return int(gesture_values[0])
    return "all_selected"


def build_standard_selection(base_data_full, all_labels, params):
    all_gestures = list(params["gesture_list"])
    gesture_values = all_gestures if params.get("gesture_subset") is None else params["gesture_subset"]
    emg_data, _, y_data = select_gesture_data(base_data_full, all_labels, gesture_values, all_gestures)
    channel_order = resolve_channel_order(emg_data.shape[0], params)

    return [
        {
            "emg_data": emg_data,
            "y_data": y_data,
            "gesture_scope": describe_gesture_scope(gesture_values),
            "gesture_values": list(gesture_values),
            "all_gestures": list(all_gestures),
            "channel_order": channel_order,
            "cache_key": (
                tuple(gesture_values),
                params["armband"],
                params.get("channel_order_mode"),
                params.get("corrupt_group"),
            ),
        }
    ]


def build_per_gesture_selection(base_data_full, all_labels, params):
    selections = []
    all_gestures = list(params["gesture_list"])

    for gesture in all_gestures:
        emg_data, _, y_data = select_gesture_data(base_data_full, all_labels, [gesture], all_gestures)
        channel_order = resolve_channel_order(emg_data.shape[0], params)
        selections.append(
            {
                "emg_data": emg_data,
                "y_data": y_data,
                "gesture_scope": int(gesture),
                "gesture_values": [int(gesture)],
                "all_gestures": list(all_gestures),
                "channel_order": channel_order,
                "cache_key": ((int(gesture),), params["armband"], params.get("channel_order_mode")),
            }
        )

    return selections


# ---------------------------------------------------------------------------
# CSV / experiment runner
# ---------------------------------------------------------------------------


def append_rows_to_csv(output_path, rows):
    if not rows:
        return
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    write_header = not os.path.exists(output_path)
    pd.DataFrame(rows).to_csv(output_path, mode="a", header=write_header, index=False)


def make_row_key(
    experiment_id,
    file_path,
    gesture_subset,
    gesture,
    gesture_class,
    gesture_subset_values,
    fault_mode,
    max_noise_amplitude,
    num_channels,
    channel_order_mode,
    corrupt_group,
    corrupted_channel_indices,
    n_voting,
    n_trials,
    armband,
    normalization_mode,
    dataset_kind,
):
    parts = [
        str(dataset_kind),
        str(int(experiment_id)),
        str(file_path),
        str(gesture_subset),
        str(int(gesture)),
        str(int(gesture_class)),
        str(gesture_subset_values),
        str(fault_mode),
        f"{float(max_noise_amplitude):.10g}",
        str(int(num_channels)),
        str(channel_order_mode),
        str(corrupt_group),
        str(corrupted_channel_indices),
        str(int(n_voting)),
        str(int(n_trials)),
        "1" if armband else "0",
        str(normalization_mode),
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def load_existing_row_keys(output_path):
    if not os.path.exists(output_path):
        return set()
    existing = pd.read_csv(output_path, usecols=["row_key"])
    return set(existing["row_key"].dropna().astype(str).tolist())


def run_experiment_block(
    experiment_id,
    model,
    model_name,
    files_to_run,
    selection_builder,
    params_template,
    sweep_points,
    batch_size,
    chunk_steps,
    output_suffix,
    pipeline_dir,
    overwrite_csv=False,
    infer_fn=None,
    output_path=None,
    reuse_clean=False,
):
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(resolve_from_pipeline_dir(pipeline_dir, model_name)),
            f"{os.path.basename(model_name)}_experiment_{experiment_id}{output_suffix}.csv",
        )
    existing_row_keys = set()
    if overwrite_csv:
        if os.path.exists(output_path):
            os.remove(output_path)
    else:
        try:
            existing_row_keys = load_existing_row_keys(output_path)
            if existing_row_keys:
                print("resume enabled - existing rows:", len(existing_row_keys))
        except ValueError as exc:
            raise RuntimeError(
                f"Existing CSV '{output_path}' cannot be used for resume (missing row_key column). "
                "Use --overwrite-csv to regenerate from scratch."
            ) from exc

    if infer_fn is None:
        infer_fn = build_infer_fn(model)

    for file_path in tqdm(files_to_run, desc=f"experiment_{experiment_id}", leave=True):
        print(file_path)
        base_data_full, all_labels = load_subject_data(file_path, params_template)
        selection_items = selection_builder(base_data_full, all_labels, params_template)

        for selection_item in selection_items:
            emg_data = selection_item["emg_data"]
            y_data = selection_item["y_data"]
            params = dict(params_template)
            params["num_windows"] = (emg_data.shape[1] - params["step"]) // params["step"]
            channel_order = selection_item["channel_order"]

            print("num windows", params["num_windows"])
            print(float(np.min(emg_data)), float(np.max(emg_data)))

            all_gestures_sorted = list(np.sort(selection_item["all_gestures"]))
            subset_gestures = list(selection_item["gesture_values"])
            subset_gestures_set = set(int(value) for value in subset_gestures)
            remap = {gesture: new_gesture for new_gesture, gesture in enumerate(all_gestures_sorted)}
            subset_classes = [remap[int(value)] for value in subset_gestures if int(value) in remap]
            subset_gestures_str = ",".join(str(int(value)) for value in subset_gestures)
            clean_cache = None
            skipped_points = 0

            t_experiment = time.time()

            for max_var, requested_nch, fault_mode in sweep_points:
                effective_order = channel_order[:requested_nch]
                corrupted_channels = describe_channels(effective_order)
                corrupted_indices = ",".join(str(int(idx)) for idx in effective_order.tolist())
                num_noisy = requested_nch if fault_mode == "noise" else 0
                num_dead = requested_nch if fault_mode == "dead" else 0
                point_keys = []
                for class_idx in subset_classes:
                    gesture_original = int(all_gestures_sorted[int(class_idx)])
                    if gesture_original not in subset_gestures_set:
                        continue
                    point_keys.append(make_row_key(
                        experiment_id=experiment_id,
                        file_path=file_path,
                        gesture_subset=selection_item["gesture_scope"],
                        gesture=gesture_original,
                        gesture_class=int(class_idx),
                        gesture_subset_values=subset_gestures_str,
                        fault_mode=fault_mode,
                        max_noise_amplitude=float(max_var),
                        num_channels=int(requested_nch),
                        channel_order_mode=params.get("channel_order_mode", "grid"),
                        corrupt_group=params.get("corrupt_group", ""),
                        corrupted_channel_indices=corrupted_indices,
                        n_voting=int(params["n_voting"]),
                        n_trials=int(params["n_trials"]),
                        armband=bool(params["armband"]),
                        normalization_mode=params["normalization_mode"],
                        dataset_kind=params.get("dataset_kind", "65hd"),
                    ))
                if point_keys and all(key in existing_row_keys for key in point_keys):
                    skipped_points += 1
                    continue

                print(
                    f"fault={fault_mode} max_var={max_var} n_channels={requested_nch} "
                    f"channels={corrupted_channels}"
                )

                if reuse_clean and int(requested_nch) == 0 and clean_cache is not None:
                    _, y_true, latency_times, trial_accuracies, trial_results = clean_cache
                    print("reuse unchanged signal (0 channels)")
                else:
                    _, y_true, latency_times, trial_accuracies, trial_results = predict_with_trials(
                        model=model,
                        base_data=emg_data,
                        y_data=y_data,
                        params=params,
                        channel_order=channel_order,
                        n_channels=requested_nch,
                        max_var=max_var,
                        fault_mode=fault_mode,
                        batch_size=batch_size,
                        chunk_steps=chunk_steps,
                        infer_fn=infer_fn,
                    )
                    if reuse_clean and int(requested_nch) == 0:
                        clean_cache = (
                            None, y_true, latency_times, trial_accuracies, trial_results
                        )

                total_time = time.time() - t_experiment
                overall_acc = float(np.mean(trial_accuracies))
                overall_std = float(np.std(trial_accuracies))
                print(
                    f"User: {file_path} accuracy={overall_acc:.4f}+-{overall_std:.4f} "
                    f"(n_trials={params['n_trials']})"
                )

                rows = []
                for class_idx in subset_classes:
                    gesture_original = int(all_gestures_sorted[int(class_idx)])
                    if gesture_original not in subset_gestures_set:
                        continue

                    class_accs = []
                    for y_pred_trial, y_true_trial, _ in trial_results:
                        mask = y_true_trial == int(class_idx)
                        if not np.any(mask):
                            continue
                        class_accs.append(
                            float(skm.accuracy_score(y_true_trial[mask], y_pred_trial[mask]))
                        )

                    if not class_accs:
                        continue

                    acc_mean = float(np.mean(class_accs))
                    acc_std = float(np.std(class_accs))

                    row_key = make_row_key(
                        experiment_id=experiment_id,
                        file_path=file_path,
                        gesture_subset=selection_item["gesture_scope"],
                        gesture=gesture_original,
                        gesture_class=int(class_idx),
                        gesture_subset_values=subset_gestures_str,
                        fault_mode=fault_mode,
                        max_noise_amplitude=float(max_var),
                        num_channels=int(requested_nch),
                        channel_order_mode=params.get("channel_order_mode", "grid"),
                        corrupt_group=params.get("corrupt_group", ""),
                        corrupted_channel_indices=corrupted_indices,
                        n_voting=int(params["n_voting"]),
                        n_trials=int(params["n_trials"]),
                        armband=bool(params["armband"]),
                        normalization_mode=params["normalization_mode"],
                        dataset_kind=params.get("dataset_kind", "65hd"),
                    )
                    if row_key in existing_row_keys:
                        continue
                    existing_row_keys.add(row_key)
                    rows.append(
                        {
                            "row_key": row_key,
                            "dataset_kind": params.get("dataset_kind", "65hd"),
                            "file": file_path,
                            "gesture_subset": selection_item["gesture_scope"],
                            "gesture": gesture_original,
                            "gesture_class": int(class_idx),
                            "gesture_subset_values": subset_gestures_str,
                            "gesture_subset_size": int(len(subset_gestures)),
                            "fault_mode": fault_mode,
                            "max_noise_amplitude": float(max_var),
                            "num_noisy_channels": int(num_noisy),
                            "num_fault_channels": int(num_dead),
                            "num_channels": int(requested_nch),
                            "channel_order_mode": params.get("channel_order_mode", "grid"),
                            "corrupt_group": params.get("corrupt_group", ""),
                            "corrupted_channels": corrupted_channels,
                            "corrupted_channel_indices": corrupted_indices,
                            "n_voting": int(params["n_voting"]),
                            "n_trials": int(params["n_trials"]),
                            "armband": bool(params["armband"]),
                            "normalization_mode": params["normalization_mode"],
                            "accuracy": acc_mean,
                            "accuracy_std": acc_std,
                            "latency_first": float(latency_times["first_bacth"]),
                            "latency_second": float(latency_times["second_bacth"]),
                            "latency_avg": float(latency_times["mean"]),
                            "total_time": float(total_time),
                        }
                    )

                append_rows_to_csv(output_path, rows)
                gc.collect()

            if skipped_points:
                print(
                    "skipped %d sweep points already stored for %s"
                    % (skipped_points, file_path),
                    flush=True,
                )

        gc.collect()

    print("saved", output_path)


def build_sweep_noise(max_var_values, channel_values):
    return [(float(max_var), int(nch), "noise") for max_var in max_var_values for nch in channel_values]


def build_sweep_dead(channel_values):
    return [(0.0, int(nch), "dead") for nch in channel_values]


def build_sweep_clean():
    return [(0.0, 0, "none")]


def build_experiment_configs(files_list, gesture_list=None, normalization_mode="bychannel", num_channels=128):
    base_params = dict(DEFAULT_PARAMS)
    if gesture_list is not None:
        base_params["gesture_list"] = list(gesture_list)
        base_params["num_classes"] = len(gesture_list)

    base_params["normalization_mode"] = normalization_mode

    exp1_params = dict(base_params, n_voting=3, armband=False, gesture_subset=list(base_params["gesture_list"]))
    exp2_params = dict(exp1_params)
    exp3_params = dict(exp1_params)
    exp4_params = dict(base_params, n_voting=5, armband=False, gesture_subset=list(base_params["gesture_list"]))
    exp5_params = dict(base_params, n_voting=5, armband=False)

    channel_sweep_small = list(range(8))
    channel_sweep_wide = list(range(8, num_channels, 8))
    max_var_values = [round(value / 10, 1) for value in range(8, 13)]

    return {
        1: {
            "files": files_list[:],
            "selection_builder": build_standard_selection,
            "params": exp1_params,
            "sweep_points": build_sweep_noise(max_var_values, channel_sweep_small),
        },
        2: {
            "files": files_list[:],
            "selection_builder": build_standard_selection,
            "params": exp2_params,
            "sweep_points": build_sweep_noise(max_var_values, channel_sweep_wide),
        },
        3: {
            "files": files_list[:],
            "selection_builder": build_standard_selection,
            "params": exp3_params,
            "sweep_points": build_sweep_dead(channel_sweep_small),
        },
        4: {
            "files": files_list[:],
            "selection_builder": build_standard_selection,
            "params": exp4_params,
            "sweep_points": build_sweep_noise(max_var_values, channel_sweep_small),
        },
        5: {
            "files": files_list[1:2],
            "selection_builder": build_per_gesture_selection,
            "params": exp5_params,
            "sweep_points": build_sweep_clean(),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deployment robustness pipeline v2: batched inference + additive noise / "
            "dead-electrode faults (unified notebook model)."
        )
    )
    parser.add_argument(
        "--pipeline-dir",
        default=os.environ.get("EMG_PIPELINE_DIR", DEFAULT_PIPELINE_DIR),
        help="Directory containing trainings/ and this script.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Experiment ids to run.",
    )
    parser.add_argument(
        "--gesture-list-json",
        default=None,
        help="Optional JSON (gestures_comparison_65HD_Hyser.json) to load hdemg gesture ids.",
    )
    parser.add_argument(
        "--gesture-indices",
        nargs="+",
        default=None,
        help="Indices into gesture_list. Supports ranges like 0-3.",
    )
    parser.add_argument(
        "--subject-indices",
        nargs="+",
        default=None,
        help="Indices into the discovered subject/session list. Supports ranges.",
    )
    parser.add_argument(
        "--single-gesture-index",
        type=int,
        default=None,
        help="Pick one gesture by index into gesture_list.",
    )
    parser.add_argument(
        "--single-subject-index",
        type=int,
        default=None,
        help="Pick one subject by index into the filtered file list.",
    )
    parser.add_argument(
        "--normalization-mode",
        default="bychannel",
        choices=["bychannel", "bysample"],
        help="Normalization axis and model checkpoint to load.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(DATASET_PROFILES.keys()),
        help="Dataset profile: 65hd (raw 65_HDsEMG) or hyser (raw pr_dataset WFDB).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override raw dataset root path from the selected profile.",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Use every discovered subject/session instead of the profile subject_filter.",
    )
    parser.add_argument(
        "--channel-order",
        default="grid",
        choices=["grid", "extensor", "flexor", "both", "interleave"],
        help="'grid' = column-major electrode order (unified). Others = legacy index1/index2 maps.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_PARAMS["n_trials"],
        help="Independent noise trials per sweep point (mean/std reported).",
    )
    parser.add_argument(
        "--noise-seed-base",
        type=int,
        default=DEFAULT_PARAMS["noise_seed_base"],
        help="Base seed; trial t uses noise_seed_base + t.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU index for TensorFlow.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=1024,
        help="Streaming preprocess chunk size (in 32-sample steps).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_v2",
        help="Suffix for generated CSV files.",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Delete existing CSV and regenerate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_gpu(args.gpu)

    params = dict(DEFAULT_PARAMS)
    model_names = apply_dataset_profile(params, args.dataset, dataset_path_override=args.dataset_path)
    params["n_trials"] = args.n_trials
    params["noise_seed_base"] = args.noise_seed_base

    if args.channel_order == "grid":
        params["channel_order_mode"] = "grid"
        params["corrupt_group"] = ""
    else:
        params["channel_order_mode"] = "legacy"
        params["corrupt_group"] = args.channel_order

    if args.gesture_list_json:
        json_path = args.gesture_list_json
        if not os.path.isabs(json_path):
            json_path = resolve_from_pipeline_dir(args.pipeline_dir, json_path)
        id_field = "hyser_id" if args.dataset == "hyser" else "hdemg_id"
        params["gesture_list"] = load_gesture_list_from_json(json_path, id_field=id_field)
        params["num_classes"] = len(params["gesture_list"])

    if args.single_gesture_index is not None and args.gesture_indices is not None:
        raise ValueError("Use either --gesture-indices or --single-gesture-index, not both.")
    if args.single_subject_index is not None and args.subject_indices is not None:
        raise ValueError("Use either --subject-indices or --single-subject-index, not both.")

    gesture_indices = parse_index_list(args.gesture_indices)
    subject_indices = parse_index_list(args.subject_indices)
    if args.single_gesture_index is not None:
        gesture_indices = [args.single_gesture_index]
    if args.single_subject_index is not None:
        subject_indices = [args.single_subject_index]

    selected_gestures = select_by_indices(params["gesture_list"], gesture_indices, "Gesture")
    files_list = build_files_list(
        params,
        args.pipeline_dir,
        subject_indices=subject_indices,
        all_subjects=args.all_subjects,
    )
    model_name = model_names[args.normalization_mode]

    print("dataset:", args.dataset)
    print("dataset path:", resolve_from_pipeline_dir(args.pipeline_dir, params["dataset_path"]))
    print("gesture list:", selected_gestures)
    print("normalization:", args.normalization_mode)
    print("fault injection profile:", params["dataset_name"])
    print("subjects/sessions:", len(files_list))
    print("channel_order:", args.channel_order)
    print("n_trials:", args.n_trials)
    for idx, file_path in enumerate(files_list):
        print(idx, file_path)

    model_path = resolve_from_pipeline_dir(args.pipeline_dir, model_name)
    model = tf.keras.models.load_model(model_path, compile=False)

    experiment_configs = build_experiment_configs(
        files_list,
        gesture_list=selected_gestures,
        normalization_mode=args.normalization_mode,
        num_channels=params["num_channels"],
    )

    for experiment_id in args.experiments:
        config = experiment_configs[experiment_id]
        config["params"].update(
            {
                "dataset_kind": params["dataset_kind"],
                "dataset_name": params["dataset_name"],
                "dataset_path": params["dataset_path"],
                "file_glob": params["file_glob"],
                "subject_filter": params["subject_filter"],
                "rest_repetition": params["rest_repetition"],
                "window_trim": params["window_trim"],
                "n_trials": args.n_trials,
                "noise_seed_base": args.noise_seed_base,
                "channel_order_mode": params["channel_order_mode"],
                "corrupt_group": params["corrupt_group"],
            }
        )

        run_experiment_block(
            experiment_id=experiment_id,
            model=model,
            model_name=model_name,
            files_to_run=config["files"],
            selection_builder=config["selection_builder"],
            params_template=config["params"],
            sweep_points=config["sweep_points"],
            batch_size=args.batch_size,
            chunk_steps=args.chunk_steps,
            output_suffix=args.output_suffix,
            pipeline_dir=args.pipeline_dir,
            overwrite_csv=args.overwrite_csv,
        )


if __name__ == "__main__":
    main()
