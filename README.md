# Spectrum Aware Illumination Estimation Using Multispectral Image

This is repository for our paper "Spectrum Aware Illumination Estimation Using Multispectral Image".
The work is to estimate 36-channel illumination spectra from 15-channel hyperspectral images.

## Dataset

**MILD (Multispectral Image dataset with Lighting Diversity)**
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19026276.svg)](https://doi.org/10.5281/zenodo.19026276)

## Project Structure

```
conv3dformer_release/
├── main_conv3dformer.py              # Main training / testing entry point
├── trainer_conv3dformer.py           # Training and evaluation logic
├── requirements.txt
├── CMF_31.mat                        # Color matching function (31ch)
├── model/
│   └── conv3dformer_Full_v2.py       # Model architecture
├── model_utils/
│   ├── option.py                     # Argument parser & hyperparameters
│   ├── Loss.py                       # Angular loss functions
│   ├── utils.py                      # Utility functions
│   ├── visualization.py              # Visualization helpers
│   ├── UNIST_CCM.mat                 # Color correction matrix
│   ├── cmf_36.mat                    # Color matching function (36ch)
│   └── optimized_filters_responses_IRC.npy
├── dataloader/
│   ├── dataloader_v2.py              # DataLoader wrapper (use this)
│   ├── dataset_v2.py                 # Dataset class with lazy h5py loading
│   ├── load_dataset_v2.py            # Dataset path & key definitions
│   └── utils_dataset.py              # RandomCrop and helpers
├── dataset/                          # Place your data here (see Dataset Setup)
│   ├── BeyondRGB/
│   └── MILD/
└── save_dir/
    └── best_model/                   # Trained model checkpoints
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place HDF5 dataset files as follows:

```
dataset/
├── BeyondRGB/
│   ├── beyondRGB_train_image.hdf5
│   ├── beyondRGB_train_illum.hdf5
│   ├── beyondRGB_val_image.hdf5
│   ├── beyondRGB_val_illum.hdf5
│   ├── beyondRGB_test_image.hdf5
│   └── beyondRGB_test_illum.hdf5
└── MILD/
    ├── indoor_{train,val,test}_{image,illum}.hdf5
    ├── indoor_v2_{train,val,test}_{image,illum}.hdf5
    ├── ECDRI_1st_{train,val,test}_{image,illum}.hdf5
    ├── ECDRI_3rd_{train,val,test}_{image,illum}.hdf5
    └── ECDRI_7th_{train,val,test}_{image,illum}.hdf5
```

Each HDF5 file contains:
- **image file**: 15-channel hyperspectral images
- **illum file**: 36-channel ground truth illumination spectra

## Training

### Train on BeyondRGB dataset

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_conv3dformer.py \
  --batch_size 50 \
  --num_workers 30 \
  --num_epochs 1500 \
  --image BeyondRGB \
  --log_file_name conv3dformer_BeyondRGB.log \
  --logger_name conv3dformer_BeyondRGB \
  --model_name conv3dformer_Full_v2 \
  --lr_rate 5e-3 \
  --wc 1 --wc_xyz 0
```

### Train on MILD dataset

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_conv3dformer.py \
  --batch_size 50 \
  --num_workers 30 \
  --num_epochs 1700 \
  --image indoor indoor_v2 ECDRI_1st ECDRI_3rd ECDRI_7th \
  --log_file_name conv3dformer_MILD.log \
  --logger_name conv3dformer_MILD \
  --model_name conv3dformer_Full_v2 \
  --lr_rate 1e-3 \
  --wc 1 --wc_xyz 0
```

### Resume training from checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_conv3dformer.py \
  --batch_size 50 \
  --num_workers 30 \
  --num_epochs 2000 \
  --image indoor indoor_v2 ECDRI_1st ECDRI_3rd ECDRI_7th \
  --model_name conv3dformer_Full_v2 \
  --lr_rate 1e-3 \
  --wc 1 --wc_xyz 0 \
  --load_model \
  --model_path ./save_dir/best_model/best_Model_lr_0.001000_Batch_50_MILD.pt
```
Log and checkpoints are saved with `_conti` suffix automatically.

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_conv3dformer.py \
  --test \
  --load_model \
  --model_path ./save_dir/best_model/best_Model_lr_0.001000_Batch_50_MILD.pt \
  --model_name conv3dformer_Full_v2 \
  --image indoor indoor_v2 ECDRI_1st ECDRI_3rd ECDRI_7th \
  --batch_size 1
```

## Model Architecture

**conv3dformer_Full_v2**
- Input: 15-channel hyperspectral image (256×256)
- Output: 36-channel illumination spectrum
- Key components:
  - 3D Convolution layers for spatial-spectral feature extraction
  - Global Spectral Attention (GSA) modules
  - 3D Channel Attention Blocks (CABlock_3D)


## Loss Functions

- **Angular Error (AE)**: angle between predicted and ground truth illumination vectors
- **XYZ Loss**: angular error in XYZ color space
- **Combined**: `wc * AE_hyper + wc_xyz * AE_xyz`

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Scheduler | CosineAnnealingWarmupRestarts |
| first_cycle_steps | 500 |
| max_lr / min_lr | 1e-2 / 5e-6 |
| warmup_steps | 20 |
| gamma | 0.5 |

## Citation

If you use this code, please cite our work.
