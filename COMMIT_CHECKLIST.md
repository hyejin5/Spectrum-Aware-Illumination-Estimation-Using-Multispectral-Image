# Git Commit Checklist

## Files Included in Git

### Documentation
- [x] README.md
- [x] MODEL_INFO.md
- [x] COMMIT_CHECKLIST.md
- [x] .gitignore

### Main Scripts
- [x] main_conv3dformer.py
- [x] trainer_conv3dformer.py
- [x] requirements.txt

### Model Architecture
- [x] model/conv3dformer_Full_v2.py

### Model Utilities
- [x] model_utils/option.py
- [x] model_utils/Loss.py
- [x] model_utils/utils.py
- [x] model_utils/visualization.py

### Dataloader (v2 — lazy h5py loading)
- [x] dataloader/dataloader_v2.py
- [x] dataloader/dataset_v2.py
- [x] dataloader/load_dataset_v2.py
- [x] dataloader/utils_dataset.py

### Essential Data Files
- [x] model_utils/UNIST_CCM.mat
- [x] model_utils/cmf_36.mat
- [x] model_utils/optimized_filters_responses_IRC.npy
- [x] CMF_31.mat

### Model Checkpoints (Git LFS recommended)
- [ ] save_dir/best_model/best_Model_lr_0.001000_Batch_50_MILD.pt
- [ ] save_dir/best_model/best_Model_lr_0.005000_Batch_50_BeyondRGB.pt

## Files NOT Included (per .gitignore)

- **Legacy dataloader**: `dataloader/load_dataset.py`, `dataloader/dataset.py`, `dataloader/dataloader.py`
- **Datasets**: `dataset/` (*.hdf5) — distribute separately
- **Logs**: `*.log`, `runs/`
- **Checkpoints**: `save_dir/check_dir/`
- **Results**: `*.xlsx`, `save_dir/test_output/`
- **Training outputs**: `training_illum/`, `training_rgb/`

## Git LFS Setup (for checkpoint files)

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add save_dir/best_model/
```

## Commit Commands

```bash
# Stage all tracked files
git add .

# Verify what will be committed
git diff --cached --name-status

# Commit
git commit -m "Add Conv3DFormer release with lazy dataloader

- conv3dformer_Full_v2 model architecture
- Training scripts for BeyondRGB and MILD datasets
- Lazy h5py dataloader (dataset_v2) for multi-worker efficiency
- Best model checkpoints: MILD and BeyondRGB
- Full documentation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push origin main
```

## Verify Before Commit

```bash
# Check status
git status

# Check file sizes
git ls-files | xargs ls -lh 2>/dev/null | sort -k5 -rh | head -20

# Confirm large files are tracked by LFS
git lfs ls-files
```
