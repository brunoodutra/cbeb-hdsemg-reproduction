# Put datasets here

Default root for every notebook and for `Pipeline_deployment_v2.py`.
Override with `CBEB_DATA_ROOT` if this folder is not `<package>/data`.

```text
data/65HD_gestures_common/{train,val,test}/          # processed, training
data/hyser_gestures_common_mono/{train,val,test}/    # processed, training
data/65_HDsEMG/                                      # raw .mat, robustness
data/HYSER/files/hd-semg/1.0.0/pr_dataset/           # raw WFDB, robustness
```

Download links and the preprocess notebooks are in `docs/DATASETS.md`.
Do not commit the recordings.
