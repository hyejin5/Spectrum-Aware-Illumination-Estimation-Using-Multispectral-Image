"""
Conv3DFormer — Inference Usage Example

Checkpoints:
  - save_dir/best_model/best_Model_lr_0.001000_Batch_50_MILD.pt
  - save_dir/best_model/best_Model_lr_0.005000_Batch_50_BeyondRGB.pt
"""

import torch
import numpy as np
from importlib import import_module
from scipy import io

# ============================================================
# 1. Load Model
# ============================================================

MODEL_PATH = 'save_dir/best_model/best_Model_lr_0.001000_Batch_50_MILD.pt'

model_module = import_module('model.conv3dformer_Full_v2')
model = getattr(model_module, 'conv3dformer_Full_v2')()

checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model loaded from : {MODEL_PATH}")
print(f"Trained for       : {checkpoint['epoch']} epochs")

# ============================================================
# 2. Prepare Input
# ============================================================

# Input: 15-channel hyperspectral image (B, 15, 256, 256), normalized 0-1
batch_size = 1
input_image = torch.rand(batch_size, 15, 256, 256).to(device)

# ============================================================
# 3. Inference
# ============================================================

with torch.no_grad():
    output = model(input_image)   # (B, 36)
    output = output / output.max(dim=1, keepdim=True)[0].clamp(min=1e-8)

print(f"\nInput  shape : {input_image.shape}")
print(f"Output shape : {output.shape}  (36-channel illumination spectrum)")
print(f"  min={output.min().item():.4f}  max={output.max().item():.4f}")

# ============================================================
# 4. Convert Spectrum to XYZ
# ============================================================

from model_utils import utils

cmf_36 = io.loadmat('./model_utils/cmf_36.mat')
output_xyz = utils.hyper2xyz_illum_batch(output.detach().cpu(), cmf_36)
output_xyz = output_xyz / output_xyz.max().clamp(min=1e-8)

print(f"\nXYZ (first sample):")
print(f"  X={output_xyz[0,0].item():.4f}  Y={output_xyz[0,1].item():.4f}  Z={output_xyz[0,2].item():.4f}")

# ============================================================
# 5. Test with Real Dataloader
# ============================================================

def test_with_dataloader(image_list=('indoor', 'ECDRI_1st'), num_workers=4):
    """Run one test batch using the v2 dataloader."""
    from dataloader import dataloader_v2 as dataloader
    from model_utils.option import args

    args.image_list  = list(image_list)
    args.batch_size  = 1
    args.num_workers = num_workers

    _dataloader = dataloader.get_dataloader(args)

    for input_img, gt_illum, gt_xyz, name in _dataloader['test']:
        input_img = input_img.to(device)

        with torch.no_grad():
            pred = model(input_img)
            pred = pred / pred.max(dim=1, keepdim=True)[0].clamp(min=1e-8)

        print(f"\nTest image : {name[0]}")
        print(f"  Pred spectrum shape : {pred.shape}")
        print(f"  GT   spectrum shape : {gt_illum.shape}")
        break

    print("Test completed.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Conv3DFormer — Inference Example")
    print("=" * 60)

    # Basic inference already ran above.

    # Uncomment to test with real data:
    # test_with_dataloader(image_list=('indoor', 'indoor_v2', 'ECDRI_1st', 'ECDRI_3rd', 'ECDRI_7th'))

    print("\nDone.")
