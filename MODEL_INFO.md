# Model Information

## Available Checkpoints

### 1. MILD Model
**File**: `save_dir/best_model/best_Model_lr_0.001000_Batch_50_MILD.pt`
**Trained on**: indoor + indoor_v2 + ECDRI_1st + ECDRI_3rd + ECDRI_7th

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_conv3dformer.py \
  --batch_size 50 --num_workers 30 --num_epochs 1700 \
  --image indoor indoor_v2 ECDRI_1st ECDRI_3rd ECDRI_7th \
  --model_name conv3dformer_Full_v2 \
  --lr_rate 1e-3 --wc 1 --wc_xyz 0
```

### 2. BeyondRGB Model
**File**: `save_dir/best_model/best_Model_lr_0.005000_Batch_50_BeyondRGB.pt`
**Trained on**: BeyondRGB dataset

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_conv3dformer.py \
  --batch_size 50 --num_workers 30 --num_epochs 1500 \
  --image BeyondRGB \
  --model_name conv3dformer_Full_v2 \
  --lr_rate 5e-3 --wc 1 --wc_xyz 0
```

## Optimizer Configuration

- **Type**: AdamW
- **Epsilon**: 1e-6

## Scheduler Configuration

- **Type**: CosineAnnealingWarmupRestarts
- **first_cycle_steps**: 500
- **cycle_mult**: 1
- **max_lr**: 1e-2
- **min_lr**: 5e-6
- **warmup_steps**: 20
- **gamma**: 0.5

## Model Architecture — conv3dformer_Full_v2

### Conv3D Layers
1. **Layer 1**: 1 → 15 channels, kernel=3, stride=2, padding=1
2. **Layer 2**: 15 → 36 channels, kernel=3, stride=2, padding=1
3. **Layer 3**: 36 → 36 channels, kernel=3, stride=(2,2,2), padding=1

### Attention Modules
- **GSAttention Stage 1**: 15 channels, 3 heads
- **GSAttention Stage 2**: 36 channels, 6 heads

### Channel Attention Blocks
- **CA_15**: 15×8 = 120 channels, reduction=15
- **CA_36**: 36×4 = 144 channels, reduction=15

### Fully Connected Layers
- **FC1**: 16200 → 1000
- **FC2**: 1000 → 36

## Essential Data Files

| File | Description |
|---|---|
| `model_utils/UNIST_CCM.mat` | Color correction matrix |
| `model_utils/cmf_36.mat` | Color matching function (36ch) |
| `model_utils/optimized_filters_responses_IRC.npy` | IRC filter responses (16×36) |
| `CMF_31.mat` | Color matching function (31ch) |

## Performance Metrics

Evaluated using:
- **Angular Error (AE)** in hyperspectral domain
- **Angular Error in XYZ** color space
- Statistics: Mean, Median, Trimean, Best/Worst 25%, Std

Results saved as Excel (`.xlsx`) per run.
