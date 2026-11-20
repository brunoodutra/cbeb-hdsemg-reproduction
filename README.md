# CBEB HD-sEMG reproduction package

Self-contained code to reproduce the HD-sEMG pattern-recognition paper:
channel-wise vs sample-wise min-max, 2D CNN, noise/dropout robustness on
**Hyser** and **65HD-sEMG**, plus testbench dynamic-range TFLite latency.

**Not included:** raw EMG recordings, trained SavedModels / optimizer slots,
lab-only paths. You download the public datasets, preprocess, train, evaluate,
quantize, and redraw figures with the scripts below.

## Quick map

| Path | Role |
|---|---|
| [docs/DATASETS.md](docs/DATASETS.md) | **Where to download** Hyser & 65HD + how to preprocess |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Full pipeline: train → deploy → quantize → figures |
| `EMG/DataBase_Treatment/65HD_EMG/` | 65HD raw readers + HDF5 creator notebook |
| `EMG/DataBase_Treatment/HYSER/pr_dataset/` | Hyser WFDB loaders + preprocess notebook |
| `EMG/Dataloader/` | Training Keras sequence (`EMGGestureDataGen`) |
| `EMG/Processing/` | Min-max / TKEO pipeline used in training |
| `EMG/Keras_deep_models/` | 2D CNN (`EMGConv2DResNetModel`) |
| `EMG/Experiments/Deep_learning/Train/` | Four training notebooks + `Pipeline_deployment_v2.py` |
| `embedded/Quantization/` | Convert to TFLite, latency, corruption re-run |
| `src/cbeb_figures.py` | Figs. 3–7, Tables V–VI, latency table |
| `notebooks/CBEB_figures.ipynb` | One cell per figure/table |
| `results/deployment/` | Paper evaluation CSVs (sanitized ids) |
| `results/quantization/` | Latency table + finished TFLite runs |
| `results/figures/` | Regenerated PNGs/PDFs |
| `models/` | Optional reference `.tflite` (no training checkpoints) |

## Minimal path (figures only)

```bash
pip install -r requirements.txt
python3 -c "import sys; sys.path.insert(0,'src'); import cbeb_figures as c; d=c.load_all(); c.fig3(d); c.table_v(d); c.table_quant_loop()"
```

## Full path (train from public data)

1. Download datasets — [docs/DATASETS.md](docs/DATASETS.md)  
2. Preprocess with the notebooks under `EMG/DataBase_Treatment/`  
3. Train the four notebooks under `EMG/Experiments/Deep_learning/Train/`  
4. Run `Pipeline_deployment_v2.py` for noise/dropout CSVs  
5. Run `embedded/Quantization/run_emg_2d_cnn_quantization.py`  
6. Redraw figures — [docs/REPRODUCE.md](docs/REPRODUCE.md)

```bash
export CBEB_DATA_ROOT="$(pwd)/data"
```

## Publish this package

Prefer a **separate public GitHub repo** (not the laboratory remote). Push the
`ieee-cbeb` orphan branch as `main`.
