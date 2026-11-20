#!/usr/bin/env python3
"""Figure and table functions for the CBEB HD-sEMG paper.

One public function per paper item: ``fig3``, ``fig4``, ``fig5``, ``fig6``,
``fig7``, ``table_v``, ``table_vi``, ``table_quant_loop``. Paths are resolved
from this file, so the working directory can be the package root or
``notebooks/``.

Fig. 3 bands use ``band='all_rows'`` and pandas ``.agg(['mean','std','count'])``
(pandas 1.x safe). ``extra_panels`` redraws the auxiliary IEEE pairs and
heatmaps; it is not a paper figure.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

try:
    matplotlib.use("Agg", force=False)
except Exception:
    pass
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import norm as scipy_norm

ROOT = Path(__file__).resolve().parents[1]
TRAININGS = ROOT / "results" / "deployment"
QUANT_RUNS = ROOT / "results" / "quantization" / "tflite_runs"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 120, "savefig.dpi": 300,
})
COLOR = {"bychannel": "#1f77b4", "bysample": "#ff7f0e"}
MARKER = {"bychannel": "o", "bysample": "s"}
GESTURE_PAIRS = [
    (0, 0), (1, 10), (2, 8), (3, 6), (4, 4), (5, 2),
    (6, 13), (7, 14), (10, 15), (11, 16), (30, 61), (31, 58),
]
PAIR_LABELS = ["(%s, %s)" % (h, s) for h, s in GESTURE_PAIRS]
PAPER_V = {
    ("65hd", "bysample"): 92.6, ("65hd", "bychannel"): 87.7,
    ("hyser", "bysample"): 93.6, ("hyser", "bychannel"): 85.0,
}

_CACHE = {"data": None}


def csv_path(dataset, norm, exp):
    if dataset == "hyser":
        return TRAININGS / (
            "EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_%s_adamW_"
            "experiment_%s_hyser_all.csv" % (norm, exp)
        )
    return TRAININGS / (
        "EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_%s_adamW_"
        "experiment_%s_65hd_all.csv" % (norm, exp)
    )


def load(dataset, norm, exp):
    df = pd.read_csv(csv_path(dataset, norm, exp)).copy()
    df["dataset"] = dataset
    df["norm"] = norm
    df["experiment"] = exp
    df["unit"] = df["file"]
    df["subject_core"] = df["file"].astype(str).str.extract(r"(subject\d+)", expand=False)
    missing = df["subject_core"].isna()
    if missing.any():
        df.loc[missing, "subject_core"] = (
            df.loc[missing, "file"].astype(str).str.replace(r"\\", "/", regex=True).str.split("/").str[-1]
        )
    return df


def load_all(refresh=False, verbose=False):
    """Load the 16 deployment CSVs (2 datasets x 2 norms x experiments 1–4)."""
    if _CACHE["data"] is not None and not refresh:
        return _CACHE["data"]
    data = {}
    for dataset in ("65hd", "hyser"):
        for norm in ("bysample", "bychannel"):
            for exp in (1, 2, 3, 4):
                frame = load(dataset, norm, exp)
                data[(dataset, norm, exp)] = frame
                if verbose:
                    print("  %s: n=%d units=%d" % (
                        (dataset, norm, exp), len(frame), frame.unit.nunique()
                    ))
    _CACHE["data"] = data
    return data


def xcol_for(df):
    if "fault_mode" in df.columns and str(df["fault_mode"].iloc[0]) == "dead":
        return "num_fault_channels"
    return "num_noisy_channels"


def per_unit_curve(df, xcol=None):
    """Mean accuracy over gestures/amplitudes → one value per (unit, n)."""
    xcol = xcol or xcol_for(df)
    return (
        df.groupby(["unit", xcol], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "acc", xcol: "n"})
    )


def curve_stats(df, xcol=None, band="all_rows"):
    """Mean ± SD vs number of corrupted channels.

    band='all_rows' (paper Fig.3 / Hyser_noise.ipynb):
        groupby(n) on every CSV row (subject × gesture × amplitude).
        This yields the wide shaded bands in the published figure.

    band='subjects' (previous mistaken reproduce):
        first average to one value per subject, then SD across subjects
        → much tighter bands than the PDF.
    """
    xcol = xcol or xcol_for(df)
    # Use list-form .agg([...]) — named kwargs on SeriesGroupBy fail on older pandas
    # (TypeError: Must provide 'func' or tuples of '(column, aggfunc)').
    if band == "subjects":
        units = per_unit_curve(df, xcol)
        return (
            units.groupby("n")["acc"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"count": "n_units"})
        )
    return (
        df.groupby(xcol)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={xcol: "n", "count": "n_rows"})
    )


def _save(fig, stem):
    png = OUT / ("%s.png" % stem)
    pdf = OUT / ("%s.pdf" % stem)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("Saved", png)
    return fig


def _norm_label(norm):
    return "ByChannel" if norm == "bychannel" else "BySample"


def table_v(data=None):
    """Table V continuous accuracy: unweighted mean of clean experiment-1 rows.

    Returns a DataFrame with the CSV percent, the published percent, and whether
    they agree within 0.15 percentage points.
    """
    data = data or load_all()

    def clean_acc(dataset, norm):
        frame = data[(dataset, norm, 1)]
        clean = frame[frame["num_noisy_channels"] == 0]
        return float(clean["accuracy"].mean())

    rows = []
    print("=== Table V continuous Acc ===")
    for key, paper in PAPER_V.items():
        percent = 100.0 * clean_acc(*key)
        match = abs(percent - paper) < 0.15
        print("%s: CSV=%.1f paper=%s match=%s" % (key, percent, paper, match))
        rows.append({
            "dataset": key[0],
            "norm": key[1],
            "csv_percent": round(percent, 4),
            "paper_percent": paper,
            "abs_diff_pp": round(abs(percent - paper), 4),
            "match": bool(match),
        })
    return pd.DataFrame(rows)


def _loop_points(path, kcol):
    """One latency per sweep point. The CSV copies latency_avg onto every gesture row."""
    frame = pd.read_csv(path)
    n_trials = int(frame["n_trials"].mode().iloc[0])
    frame = frame[frame["n_trials"] == n_trials]
    keys = ["file", kcol, "max_noise_amplitude", "fault_mode"]
    latency = frame.drop_duplicates(keys)
    accuracy = frame.groupby(keys, as_index=False)["accuracy"].mean()
    return latency, accuracy


def _clean_accuracy(stem, tag, exp, kcol):
    """Mean clean accuracy of one TFLite run. FP32 uses the same windows' reference CSV."""
    path = QUANT_RUNS / ("%s_experiment_%d_%s_tflite_dynamic.csv" % (stem, exp, tag))
    _, quant = _loop_points(path, kcol)
    fp32_path = TRAININGS / ("%s_experiment_%d_%s_all.csv" % (stem, exp, tag))
    _, reference = _loop_points(fp32_path, kcol)
    return (
        float(reference.loc[reference[kcol] == 0, "accuracy"].mean()),
        float(quant.loc[quant[kcol] == 0, "accuracy"].mean()),
    )


def table_quant_loop():
    """One-sample model latency on the same CPU, plus clean accuracy.

    Latency comes from ``quantization_latency.csv``: Keras ``model(x)`` with
    batch 1 versus one TFLite ``invoke``. Norms and datasets are pooled.
    ``latency_avg`` in the deployment CSVs is not used: the FP32 loop batches
    32 windows, so dividing by the window count makes FP32 look faster.

    Accuracy is the clean (k=0) mean. 65HD sample-wise dropout is omitted
    because that run is incomplete.

    Returns a DataFrame and writes ``results/quantization/loop_latency_table.csv``.
    """
    latency_path = ROOT / "results" / "quantization" / "quantization_latency.csv"
    timed = pd.read_csv(latency_path)
    # One row per dataset x norm. keras_fp32 is model(x, training=False), batch 1.
    # tflite_dynamic_range latency_mean_ms is Interpreter.invoke, batch 1.
    pieces = {
        "FP32 (Keras)": timed.loc[timed["precision"] == "keras_fp32", "infer_call_mean_ms"],
        "TFLite dynamic-range": timed.loc[
            timed["precision"] == "tflite_dynamic_range", "latency_mean_ms"
        ],
    }
    specs = []
    for dataset, prefix, tag in (
        ("65hd", "EMG_65HDsEMG", "65hd"),
        ("hyser", "EMG_HysersEMG", "hyser"),
    ):
        for norm in ("bychannel", "bysample"):
            stem = "%s_2D_CNN_128f_128ch_SCE_normminmax_%s_adamW" % (prefix, norm)
            for exp, kcol in ((1, "num_noisy_channels"), (3, "num_fault_channels")):
                if dataset == "65hd" and norm == "bysample" and exp == 3:
                    continue
                specs.append((stem, tag, exp, kcol))

    clean = {"FP32 (Keras)": [], "TFLite dynamic-range": []}
    for stem, tag, exp, kcol in specs:
        fp32_acc, quant_acc = _clean_accuracy(stem, tag, exp, kcol)
        clean["FP32 (Keras)"].append(fp32_acc)
        clean["TFLite dynamic-range"].append(quant_acc)

    rows = []
    for label, samples in pieces.items():
        values = samples.astype(float)
        rows.append({
            "model": label,
            "latency_ms_mean": round(float(values.mean()), 3),
            "latency_ms_std": round(float(values.std(ddof=1)), 3),
            "clean_accuracy_pct": round(float(np.mean(clean[label])) * 100, 3),
            "n_models": int(len(values)),
        })
    table = pd.DataFrame(rows)
    out = ROOT / "results" / "quantization" / "loop_latency_table.csv"
    table.to_csv(out, index=False)
    print("Latency: one sample, batch=1, same CPU (AMD EPYC 7742). Keras model(x) vs TFLite invoke.")
    print("Std is across the 4 models (2 datasets x 2 norms), not split in the table.")
    print("Clean accuracy at k=0. Excluded: incomplete 65HD sample-wise dropout.")
    print("wrote", out)
    return table


def table_vi(data=None):
    """Table VI-like Wilcoxon signed-rank on subject-mean accuracy.

    Hyser pairs subjects by ``subject_core`` (session files collapse to the
    subject id, same as the original reproduction). 65HD pairs file units.
    """
    data = data or load_all()
    rows = []

    def one(left, right, label):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        result = wilcoxon(left, right)
        delta = left - right
        ranks = pd.Series(np.abs(delta)).rank()
        wilcoxon_w = float(min(ranks[delta > 0].sum(), ranks[delta < 0].sum()))
        if result.pvalue > 0:
            zed = abs(float(scipy_norm.ppf(result.pvalue / 2.0)))
        else:
            zed = np.inf
        effect_r = zed / np.sqrt(len(left))
        print("%s: n=%d W=%.1f p=%.2e r≈%.2f" % (label, len(left), wilcoxon_w, result.pvalue, effect_r))
        rows.append({
            "label": label,
            "n": int(len(left)),
            "W": wilcoxon_w,
            "p": float(result.pvalue),
            "r": float(effect_r),
        })

    print("=== Table VI-like Wilcoxon ===")
    for dataset in ("65hd", "hyser"):
        for exp, tag in ((1, "noise"), (3, "dropout")):
            sample = data[(dataset, "bysample", exp)]
            channel = data[(dataset, "bychannel", exp)]
            key = "subject_core" if dataset == "hyser" else "unit"
            left = sample.groupby(key)["accuracy"].mean()
            right = channel.groupby(key)["accuracy"].mean()
            common = sorted(set(left.index) & set(right.index))
            one(left.loc[common], right.loc[common], "%s %s all-rows" % (dataset, tag))
    return pd.DataFrame(rows)


def _plot_norm_pair(ax, data, dataset, exp, xlabel):
    for norm in ("bychannel", "bysample"):
        stats = curve_stats(data[(dataset, norm, exp)], band="all_rows")
        lo = (stats["mean"] - stats["std"]).clip(lower=0.0)
        hi = (stats["mean"] + stats["std"]).clip(upper=1.0)
        ax.plot(
            stats["n"], stats["mean"], color=COLOR[norm], marker=MARKER[norm],
            linewidth=2, label=_norm_label(norm),
        )
        ax.fill_between(stats["n"], lo, hi, color=COLOR[norm], alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 8))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="best")


def fig3(data=None):
    """Fig. 3 — 2×2 robustness. Shaded band is mean ± SD over all CSV rows."""
    data = data or load_all()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharey=True, constrained_layout=True)
    _plot_norm_pair(axes[0, 0], data, "hyser", 1, "Noisy channels")
    axes[0, 0].set_title("Hyser")
    _plot_norm_pair(axes[0, 1], data, "65hd", 1, "Noisy channels")
    axes[0, 1].set_title("65HD-sEMG")
    _plot_norm_pair(axes[1, 0], data, "hyser", 3, "Dropout channels")
    axes[1, 0].set_title("Hyser")
    _plot_norm_pair(axes[1, 1], data, "65hd", 3, "Dropout channels")
    axes[1, 1].set_title("65HD-sEMG")
    fig.suptitle("Fig. 3 — Robustness under channel corruption (mean ± SD)", y=1.02)
    return _save(fig, "fig3_robustness_2x2")


def _gesture_means(data, dataset, norm, exp, exclude_clean=True):
    frame = data[(dataset, norm, exp)]
    xcol = xcol_for(frame)
    if exclude_clean:
        frame = frame[frame[xcol] > 0]
    return frame.groupby("gesture")["accuracy"].mean()


def _radar_panel(ax, values_dict, title):
    count = len(GESTURE_PAIRS)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), PAIR_LABELS, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    colors = {
        "65HD-sEMG sample-wise": "#2ca02c",
        "Hyser sample-wise": "#d62728",
        "65HD-sEMG channel-wise": "#1f77b4",
        "Hyser channel-wise": "#8c564b",
    }
    for label, vals in values_dict.items():
        closed = list(vals) + [vals[0]]
        ax.plot(angles, closed, linewidth=2, label=label, color=colors[label])
        ax.fill(angles, closed, alpha=0.12, color=colors[label])
    ax.set_title(title, pad=18)


def fig4(data=None):
    """Fig. 4 — accuracy by matched gesture pair, noise vs dropout."""
    data = data or load_all()
    series_noise = {
        "65HD-sEMG sample-wise": [_gesture_means(data, "65hd", "bysample", 1).get(s, np.nan) for _, s in GESTURE_PAIRS],
        "Hyser sample-wise": [_gesture_means(data, "hyser", "bysample", 1).get(h, np.nan) for h, _ in GESTURE_PAIRS],
        "65HD-sEMG channel-wise": [_gesture_means(data, "65hd", "bychannel", 1).get(s, np.nan) for _, s in GESTURE_PAIRS],
        "Hyser channel-wise": [_gesture_means(data, "hyser", "bychannel", 1).get(h, np.nan) for h, _ in GESTURE_PAIRS],
    }
    series_drop = {
        "65HD-sEMG sample-wise": [_gesture_means(data, "65hd", "bysample", 3).get(s, np.nan) for _, s in GESTURE_PAIRS],
        "Hyser sample-wise": [_gesture_means(data, "hyser", "bysample", 3).get(h, np.nan) for h, _ in GESTURE_PAIRS],
        "65HD-sEMG channel-wise": [_gesture_means(data, "65hd", "bychannel", 3).get(s, np.nan) for _, s in GESTURE_PAIRS],
        "Hyser channel-wise": [_gesture_means(data, "hyser", "bychannel", 3).get(h, np.nan) for h, _ in GESTURE_PAIRS],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), subplot_kw={"polar": True}, constrained_layout=True)
    _radar_panel(axes[0], series_noise, "Accuracy by Gesture (Noisy channels)")
    _radar_panel(axes[1], series_drop, "Accuracy by Gesture (Dropout channels)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    return _save(fig, "fig4_accuracy_by_gesture_radar")


def _gesture_heatmap_matrix(data, dataset, norm, exp):
    frame = data[(dataset, norm, exp)]
    xcol = xcol_for(frame)
    return frame.groupby(["gesture", xcol])["accuracy"].mean().unstack(xcol).sort_index()


def fig5(data=None):
    """Fig. 5 — gesture × corruption-count heatmaps."""
    data = data or load_all()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), constrained_layout=True)
    configs = [
        ("hyser", "bychannel", 1, "Hyser — Noisy"),
        ("65hd", "bychannel", 1, "65HD — Noisy"),
        ("hyser", "bychannel", 3, "Hyser — Dropout"),
        ("65hd", "bychannel", 3, "65HD — Dropout"),
        ("hyser", "bysample", 1, "Hyser — Noisy"),
        ("65hd", "bysample", 1, "65HD — Noisy"),
        ("hyser", "bysample", 3, "Hyser — Dropout"),
        ("65hd", "bysample", 3, "65HD — Dropout"),
    ]
    image = None
    for ax, (dataset, norm, exp, title) in zip(axes.ravel(), configs):
        matrix = _gesture_heatmap_matrix(data, dataset, norm, exp)
        image = ax.imshow(matrix.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns.astype(int))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.astype(int), fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Noisy channels" if exp == 1 else "Dropout channels")
        if ax in list(axes[:, 0]):
            ax.set_ylabel("Gesture")
    axes[0, 0].annotate(
        "Channel-wise", xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-0.55, 0.5), textcoords="axes fraction",
        rotation=90, va="center", fontsize=10, fontweight="bold",
    )
    axes[1, 0].annotate(
        "Sample-wise", xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-0.55, 0.5), textcoords="axes fraction",
        rotation=90, va="center", fontsize=10, fontweight="bold",
    )
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    colorbar.set_label("Accuracy")
    fig.suptitle("Fig. 5 — Accuracy under incremental corruption across gestures", y=1.03)
    return _save(fig, "fig5_gesture_heatmaps")


def fig6(data=None):
    """Fig. 6 — noise amplitude × number of noisy channels."""
    data = data or load_all()
    amps = [0.8, 0.9, 1.0, 1.1, 1.2]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), subplot_kw={"polar": True}, constrained_layout=True)
    panels = [
        ("hyser", "bychannel", "HYSER — Channel-wise"),
        ("65hd", "bychannel", "65HD-sEMG — Channel-wise"),
        ("hyser", "bysample", "HYSER — Sample-wise"),
        ("65hd", "bysample", "65HD-sEMG — Sample-wise"),
    ]
    cmap = plt.get_cmap("tab10")
    angles = np.linspace(0, 2 * np.pi, len(amps), endpoint=False).tolist()
    angles += angles[:1]
    for ax, (dataset, norm, title) in zip(axes.ravel(), panels):
        frame = data[(dataset, norm, 1)]
        matrix = (
            frame.groupby(["num_noisy_channels", "max_noise_amplitude"])["accuracy"]
            .mean()
            .unstack("max_noise_amplitude")
            .sort_index()
        )
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), ["%g" % amp for amp in amps])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        for index, count in enumerate(sorted(matrix.index)):
            vals = [matrix.loc[count, amp] for amp in amps]
            vals = vals + [vals[0]]
            ax.plot(angles, vals, color=cmap(index), linewidth=1.8, label="noisy ch %d" % int(count))
            if int(count) == int(matrix.index.max()):
                ax.fill(angles, vals, color="0.5", alpha=0.15)
        ax.set_title(title, pad=16, fontsize=10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7, frameon=False)
    fig.suptitle("Fig. 6 — Accuracy by noise amplitude and #noisy channels", y=1.02)
    return _save(fig, "fig6_noise_amplitude_radar")


def fig7(data=None):
    """Fig. 7 — extensive corruption (experiment 2, 8–120 noisy channels)."""
    data = data or load_all()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True, constrained_layout=True)
    for ax, dataset, title in zip(axes, ("hyser", "65hd"), ("Hyser", "65HD-sEMG")):
        for norm in ("bychannel", "bysample"):
            stats = curve_stats(data[(dataset, norm, 2)], xcol="num_noisy_channels", band="all_rows")
            lo = (stats["mean"] - stats["std"]).clip(lower=0.0)
            hi = (stats["mean"] + stats["std"]).clip(upper=1.0)
            ax.plot(
                stats["n"], stats["mean"], color=COLOR[norm], marker=MARKER[norm],
                linewidth=2, label=_norm_label(norm),
            )
            ax.fill_between(stats["n"], lo, hi, color=COLOR[norm], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Noisy channels")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
    stats = curve_stats(data[("65hd", "bychannel", 2)], xcol="num_noisy_channels", band="all_rows")
    at_16 = float(stats.loc[stats["n"] == 16, "mean"].iloc[0]) * 100
    print("65HD bychannel @16 noisy: %.1f%% (paper >50%%)" % at_16)
    fig.suptitle("Fig. 7 — Extensive channel corruption (mean ± SD)", y=1.05)
    return _save(fig, "fig7_extensive_corruption")


def extra_panels(data=None):
    """Auxiliary IEEE pairs and Hyser heatmaps. Not Figs. 3–7."""
    data = data or load_all()

    def ieee_pair_gain(dataset, exp, xlabel, stem):
        channel = curve_stats(data[(dataset, "bychannel", exp)], band="all_rows")
        sample = curve_stats(data[(dataset, "bysample", exp)], band="all_rows")
        gain = channel[["n", "mean"]].merge(sample[["n", "mean"]], on="n", suffixes=("_ch", "_sa"))
        gain["delta"] = gain["mean_ch"] - gain["mean_sa"]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        ax = axes[0]
        for stats, norm in ((channel, "bychannel"), (sample, "bysample")):
            lo = (stats["mean"] - stats["std"]).clip(lower=0.0)
            hi = (stats["mean"] + stats["std"]).clip(upper=1.0)
            ax.plot(
                stats["n"], stats["mean"], color=COLOR[norm], marker=MARKER[norm],
                linewidth=2, label=_norm_label(norm),
            )
            ax.fill_between(stats["n"], lo, hi, color=COLOR[norm], alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        ax.set_title("(a) Classification accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=False)
        ax = axes[1]
        ax.plot(gain["n"], gain["delta"], color="black", marker="o", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta$ Accuracy")
        ax.set_title("(b) Relative gain")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        _save(fig, stem)
        plt.close(fig)

    ieee_pair_gain("hyser", 1, "Noisy channels", "hyser_noise_ieee_pair")
    ieee_pair_gain("hyser", 3, "Faulty channels", "hyser_dropout_ieee_pair")
    ieee_pair_gain("65hd", 1, "Noisy channels", "65hd_noise_ieee_pair")
    ieee_pair_gain("65hd", 3, "Faulty channels", "65hd_dropout_ieee_pair")

    frame_ch = data[("hyser", "bychannel", 3)].copy()
    frame_sa = data[("hyser", "bysample", 3)].copy()
    for frame in (frame_ch, frame_sa):
        frame["subject"] = frame["subject_core"]
    acc_ch = frame_ch.groupby(["subject", "num_fault_channels"])["accuracy"].mean().reset_index()
    acc_sa = frame_sa.groupby(["subject", "num_fault_channels"])["accuracy"].mean().reset_index()
    gain = acc_ch.merge(acc_sa, on=["subject", "num_fault_channels"], suffixes=("_channel", "_sample"))
    gain["delta"] = gain["accuracy_channel"] - gain["accuracy_sample"]
    heatmap = gain.pivot(index="subject", columns="num_fault_channels", values="delta")
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(heatmap.values, cmap="RdBu_r", aspect="auto")
    colorbar = fig.colorbar(image, ax=ax, label=r"$\Delta$ Accuracy")
    colorbar.ax.invert_yaxis()
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=7)
    ax.set_xlabel("Faulty channels")
    ax.set_ylabel("Subject")
    fig.tight_layout()
    _save(fig, "subject_gain_heatmap_hyser_dropout")
    plt.close(fig)

    subject_gain = gain.groupby("subject")["delta"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    subject_gain.plot.bar(ax=ax, color="#1f77b4")
    ax.set_ylabel("Mean Relative Gain")
    ax.set_xlabel("Subject")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "subject_mean_gain_hyser_dropout")
    plt.close(fig)

    frame_ch = data[("hyser", "bychannel", 1)]
    frame_sa = data[("hyser", "bysample", 1)]
    acc_ch = frame_ch.groupby(["gesture", "num_noisy_channels"])["accuracy"].mean().reset_index()
    acc_sa = frame_sa.groupby(["gesture", "num_noisy_channels"])["accuracy"].mean().reset_index()
    gain = acc_ch.merge(acc_sa, on=["gesture", "num_noisy_channels"], suffixes=("_channel", "_sample"))
    gain["delta"] = gain["accuracy_channel"] - gain["accuracy_sample"]
    heatmap = gain.pivot(index="gesture", columns="num_noisy_channels", values="delta")
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    image = ax.imshow(heatmap.values, cmap="RdBu_r", aspect="auto", vmin=-0.8, vmax=0.3)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_xlabel("Faulty channels")
    ax.set_ylabel("Gesture")
    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label(r"$\Delta$ Accuracy")
    colorbar.ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "relative_gain_heatmap_hyser_noise")
    plt.close(fig)

    frame_ch = data[("hyser", "bychannel", 3)]
    frame_sa = data[("hyser", "bysample", 3)]
    gestures = sorted(frame_ch["gesture"].unique())
    ncols = 4
    nrows = int(np.ceil(len(gestures) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    last = 0
    for last, gesture in enumerate(gestures):
        ax = axes[last]
        for frame, norm in ((frame_ch, "bychannel"), (frame_sa, "bysample")):
            series = frame[frame["gesture"] == gesture].groupby("num_fault_channels")["accuracy"].mean().reset_index()
            ax.plot(
                series["num_fault_channels"], series["accuracy"],
                marker=MARKER[norm], color=COLOR[norm], linewidth=1.8, label=_norm_label(norm),
            )
        ax.set_title("Gesture %s" % gesture)
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1.05)
    for index in range(last + 1, len(axes)):
        fig.delaxes(axes[index])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.supxlabel("Number of faulty channels")
    fig.supylabel("Accuracy")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "accuracy_per_gesture_hyser_dropout")
    plt.close(fig)
    print("Saved auxiliary panels")
