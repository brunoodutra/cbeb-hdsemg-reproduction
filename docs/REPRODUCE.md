# End-to-end reproduction

This package supports the full paper path without shipping raw EMG or trained
checkpoints:

1. Download datasets → 2. Preprocess → 3. Train → 4. Evaluate (noise/dropout) →
5. Quantize → 6. Redraw figures/tables

Published CSVs under `results/` already match the paper figures, so steps 3–5
are optional if you only want to redraw plots.

## 0. Environment

```bash
cd <this-package>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CBEB_DATA_ROOT="$(pwd)/data"   # default if unset
```

## 1. Download data

Follow [DATASETS.md](DATASETS.md). Put raw files under `data/65_HDsEMG/` and
`data/HYSER/files/hd-semg/1.0.0/pr_dataset/`.

## 2. Preprocess

**65HD** — open and run:

`EMG/DataBase_Treatment/65HD_EMG/65hdsemg_custom_h5_dataset_creator.ipynb`

**Hyser** — open and run:

`EMG/DataBase_Treatment/HYSER/pr_dataset/preprocessing_HYSER_h5File.ipynb`

Training notebooks read only these two trees (already wired; do not rename):

```text
data/65HD_gestures_common/{train,val,test}/
data/hyser_gestures_common_mono/{train,val,test}/
```

## 3. Train the four 2D CNNs

Network: `EMG/Keras_deep_models/EMGConv2DResNetModel.py`  
Dataloader: `EMG/Dataloader/EMGGestureDataGen.py`  
Processing: `EMG/Processing/ProcessingPipeline.py`

Notebooks:

| Dataset | Norm | Notebook |
|---|---|---|
| 65HD | bychannel | `EMG/Experiments/Deep_learning/Train/65hdsemg_training_2D_CNN_multiclass_4preproc_armband_minmax_bychannel.ipynb` |
| 65HD | bysample | `.../65hdsemg_training_2D_CNN_multiclass_4preproc_armband_minmax_bysample.ipynb` |
| Hyser | bychannel | `.../Hyser_training_2D_CNN_multiclass_4preproc_armband_minmax_bychannel.ipynb` |
| Hyser | bysample | `.../Hyser_training_2D_CNN_multiclass_4preproc_armband_minmax_bysample.ipynb` |

SavedModels are written under
`EMG/Experiments/Deep_learning/Train/trainings/` (gitignored). Paper names:

```text
EMG_65HDsEMG_2D_CNN_128f_128ch_SCE_normminmax_{bychannel,bysample}_adamW
EMG_HysersEMG_2D_CNN_128f_128ch_SCE_normminmax_{bychannel,bysample}_adamW
```

## 4. Robustness evaluation (FP32)

```bash
cd EMG/Experiments/Deep_learning/Train
python3 Pipeline_deployment_v2.py --dataset 65hd
python3 Pipeline_deployment_v2.py --dataset hyser
```

Writes `trainings/*_experiment_{1,2,3,4}_*_all.csv`. Experiments 1 and 3 feed
Figs. 3–6 and Tables V–VI; experiment 2 feeds Fig. 7.

## 5. Quantize and re-check

```bash
# From package root, after SavedModels exist:
python3 embedded/Quantization/run_emg_2d_cnn_quantization.py
python3 embedded/Quantization/run_emg_tflite_corruption.py --dataset both
```

Dynamic-range TFLite: int8 weights, float activations. Outputs go to
`results/quantization/`. Acceptance used in the paper prep:

- clean accuracy |Δ| ≤ 2 percentage points vs FP32  
- under noise, bychannel ≥ bysample for k = 1..7  

Reference one-sample latency (AMD EPYC 7742, batch 1) is already in
`results/quantization/loop_latency_table.csv`.

## 6. Figures and tables

```bash
python3 -c "import sys; sys.path.insert(0,'src'); import cbeb_figures as c; d=c.load_all(verbose=True); c.fig3(d); c.fig4(d); c.fig5(d); c.fig6(d); c.fig7(d); print(c.table_v(d)); print(c.table_vi(d)); print(c.table_quant_loop())"
```

Or open `notebooks/CBEB_figures.ipynb`. Images land in `results/figures/`.

`cbeb_figures.py` reads the sanitized CSVs in `results/deployment/` by default
(so figure redraw works without re-running deployment). After you regenerate
CSVs locally, copy them into `results/deployment/` or point `TRAININGS` in
`src/cbeb_figures.py` at `EMG/Experiments/Deep_learning/Train/trainings`.
