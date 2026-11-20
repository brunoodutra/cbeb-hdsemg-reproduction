# Datasets: where to download and how to preprocess

Raw EMG is **not** shipped in this package. Download the public datasets below,
place them under `data/` (or set `CBEB_DATA_ROOT`), then run the preprocessing
notebooks/scripts in `EMG/DataBase_Treatment/`.

```bash
export CBEB_DATA_ROOT="$(pwd)/data"   # optional; this is the default
mkdir -p data
```

Everything the notebooks and `Pipeline_deployment_v2.py` read lives under
`<package>/data`. If the files are elsewhere, set `CBEB_DATA_ROOT` to that
folder. The names below are what the training notebooks already use
(`database_path`).

```text
data/
  65HD_gestures_common/train|val|test/     # processed 65HD HDF5 (training)
  hyser_gestures_common_mono/train|val|test/  # processed Hyser HDF5 (training)
  65_HDsEMG/                               # raw 65HD .mat (robustness loop)
  HYSER/files/hd-semg/1.0.0/pr_dataset/    # raw Hyser WFDB (robustness loop)
```

To **train only**, copy the processed trees into `data/65HD_gestures_common`
and `data/hyser_gestures_common_mono`. Each of `train/`, `val/`, and `test/`
must contain the subject HDF5 files and `dataset_param.json`.

To **rebuild** those trees, put the raw downloads in `data/65_HDsEMG` and
`data/HYSER/.../pr_dataset`, then run the preprocess notebooks. They write
the two processed folders above.

---

## 65HD-sEMG

Malešević et al., *A database of high-density surface electromyogram signals
comprising 65 isometric hand gestures*, Scientific Data, 2021.

| | |
|---|---|
| Paper | https://doi.org/10.1038/s41597-021-00843-9 |
| Files (Figshare collection) | https://doi.org/10.6084/m9.figshare.c.5090861 |
| Collection page | https://springernature.figshare.com/collections/5090861 |

Download the per-subject MATLAB files and put them in `data/65_HDsEMG/`.

### Preprocess (raw → HDF5 used for training)

Scripts (no raw files included):

- `EMG/DataBase_Treatment/65HD_EMG/HDSEMG65FileLoader.py` — read one subject `.mat`
- `EMG/DataBase_Treatment/65HD_EMG/HDSEMG65DatasetLoader.py` — extract gestures → HDF5
- `EMG/DataBase_Treatment/65HD_EMG/EMGGestureData.py`
- `EMG/DataBase_Treatment/65HD_EMG/65hdsemg_custom_h5_dataset_creator.ipynb`
  writes `data/65HD_gestures_common/{train,val,test}`
- Gesture map: `EMG/DataBase_Treatment/gestures_comparison_65HD_Hyser.json`

Paper gestures (class ids in the raw files):

`[0, 10, 8, 6, 4, 2, 13, 14, 15, 16, 61, 58]`

Example:

```python
from HDSEMG65FileLoader import HDSEMG65FileLoader
from HDSEMG65DatasetLoader import HDSEMG65DatasetLoader

loader = HDSEMG65FileLoader("data/65_HDsEMG/<subject>.mat")
# Or build the training HDF5 set with HDSEMG65DatasetLoader / the creator notebook.
```

Deployment (`Pipeline_deployment_v2.py`) reads the raw subject files under
`data/65_HDsEMG/` directly (files matching `2*`, first sorted subject skipped →
17 units).

---

## Hyser (PhysioNet `hd-semg` 1.0.0)

Jiang et al., high-density sEMG pattern-recognition subset.

| | |
|---|---|
| PhysioNet project | https://physionet.org/content/hd-semg/1.0.0/ |
| Direct tree | `hd-semg/1.0.0/pr_dataset/subject##_session#/` |

Suggested download (requires a PhysioNet account for bulk tools):

```bash
# Example with wget (follow current PhysioNet instructions if this changes)
mkdir -p data/HYSER/files
# Place or sync the pr_dataset tree at:
# data/HYSER/files/hd-semg/1.0.0/pr_dataset/
```

### Preprocess (raw WFDB → HDF5 used for training)

Scripts:

- `EMG/DataBase_Treatment/HYSER/pr_dataset/HYSERSubjSample.py`
- `EMG/DataBase_Treatment/HYSER/pr_dataset/HYSERSubject.py`
- `EMG/DataBase_Treatment/HYSER/pr_dataset/HYSERDataLoader.py`
- `EMG/DataBase_Treatment/HYSER/pr_dataset/preprocessing.py` (noise gating helpers)
- `EMG/DataBase_Treatment/HYSER/pr_dataset/hyser_movement.json`
- `EMG/DataBase_Treatment/HYSER/pr_dataset/preprocessing_HYSER_h5File.ipynb`
  writes `data/hyser_gestures_common_mono/{train,val,test}` (monopolar)

Paper gestures:

`[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 30, 31]`

20 subjects × 2 sessions = 40 folders. Open
`preprocessing_HYSER_h5File.ipynb` and run it. It reads the raw tree under
`data/HYSER/.../pr_dataset` and writes the processed folder the training
notebooks open.

Requires `wfdb` and `h5py` (see `requirements.txt`).

---

## Normalization used by the 2D CNN

Both training and deployment min-max scale windows to `[0, 1]`:

- **bychannel** — each electrode along time  
- **bysample** — each time step across electrodes  

Shared helpers: `src/processing.py` and `EMG/Processing/ProcessingPipeline.py`
(`NormMinMax`). Network input shape after reshape: `(16, 8, 256)`.
