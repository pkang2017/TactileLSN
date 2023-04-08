# Hybrid_LIF_GNN

## Requirements
torch-geometric==1.7.2
torch-scatter==2.0.8
torch-sparse==0.6.11

Please make sure you use the versions above to reproduce the results.

You may install any necessary packages in the `requirements.txt` with `pip install` or `conda install`.

## Datasets

1. Donwload the `preprocessed` data [here](https://clear-nus.github.io/visuotactile/download.html).
2. Save the preprocessed data for Objects, Containers, and Slip Detection in `datasets/preprocessed`.
3. Please refer [TactileSGNet](https://github.com/clear-nus/TactileSGNet) for the v0 datasets.

## Experiments

### Objects-v1 and Containers-v1

```bash
python train_SGNet_location_batch.py --data_dir <preporcessed data dir> --sample_file 1
```

### Slip detection

```bash
python train_SGNet_location_batch_sd.py --data_dir <preporcessed data dir> --sample_file 1
```

## Credits
The codes of this work are based on [TactileSGNet](https://github.com/clear-nus/TactileSGNet) and [STBP](https://github.com/yjwu17/BP-for-SpikingNN). Please consider citing these works if you use the codes.
