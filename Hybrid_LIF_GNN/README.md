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

```bash
python train_SGNet --epoch 500 --lr 0.001 --sample_file 1 --batch_size 8 --fingers both --data_dir <preporcessed data dir> --hidden_size 32 --loss NumSpikes --mode location --network_config <network_config>/container_weight_location.yml  --task cw --checkpoint_dir <checkpoint dir>
```
