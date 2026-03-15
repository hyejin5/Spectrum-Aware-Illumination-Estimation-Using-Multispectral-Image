import torch
from torch.utils.data import Dataset
import numpy as np
from dataloader.utils_dataset import RandomCrop
from importlib import import_module
from scipy import io
from skimage.transform import resize
import h5py
import colour

cmf_36 = io.loadmat('./model_utils/cmf_36.mat')
cmf_36 = cmf_36['cmf_36']

m = import_module('dataloader.' + 'load_dataset_v2')


def _resolve_illum_key(img_key, illum_key_set):
    """Return illum key corresponding to img_key (_L suffix, fallback _gt)."""
    candidate = img_key + '_L'
    if candidate in illum_key_set:
        return candidate
    return img_key + '_gt'


class _LazyH5Reader:
    """Per-worker h5py file handle cache. Opened on first access, never closed."""
    def get(self, path, key):
        if not hasattr(self, '_handles'):
            self._handles = {}
        if path not in self._handles:
            self._handles[path] = h5py.File(path, 'r')
        return np.array(self._handles[path][key])


class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.RandomCrop = RandomCrop
        self.image_paths = []   # list of (path, key)
        self.illum_paths = []   # list of (path, key)

        for name in self.args.image_list:
            ds = getattr(m, name)()   # ONE instance — no double-open
            illum_key_set = set(ds.keys_train_L)
            for img_key in ds.keys_train_image:
                illum_key = _resolve_illum_key(img_key, illum_key_set)
                self.image_paths.append((ds.train_image_path, img_key))
                self.illum_paths.append((ds.train_illum_path, illum_key))

    def _reader(self):
        if not hasattr(self, '_lazy'):
            self._lazy = _LazyH5Reader()
        return self._lazy

    def __getitem__(self, index):
        img_path, img_key     = self.image_paths[index]
        illum_path, illum_key = self.illum_paths[index]
        reader = self._reader()

        image = reader.get(img_path, img_key)
        illum = reader.get(illum_path, illum_key)

        illum = np.array(illum, dtype=np.float32).reshape(-1)
        illum = illum / illum.max()

        xyz_fromillum = np.matmul(illum.squeeze(), cmf_36)
        xyz_fromillum = torch.from_numpy(xyz_fromillum.transpose())
        xyz_fromillum = xyz_fromillum / xyz_fromillum.max()
        xyz_fromillum = np.clip(xyz_fromillum, 0, xyz_fromillum.max())
        gt_rgb = torch.tensor(xyz_fromillum)

        image = np.array(image, dtype=np.float32)
        if image.shape[2] == 16:
            image = np.delete(image, 15, axis=2)

        crop_image  = self.RandomCrop(image)
        input_image = np.clip(crop_image, 0, crop_image.max())
        input_image = input_image / input_image.max()
        input_image = input_image.transpose((2, 0, 1))

        Tensor_X = torch.tensor(input_image, dtype=torch.float32)
        Tensor_Y = torch.tensor(illum,        dtype=torch.float32)
        Tensor_Z = torch.tensor(gt_rgb).float()

        if torch.isnan(Tensor_X).any():
            print('!!!!!!!!!!!!!! NAN value in image file %s !!!!!!!!!!' % img_key); exit()
        if torch.isnan(Tensor_Y).any():
            print('!!!!!!!!!!!!!! NAN value in illum file %s !!!!!!!!!!' % img_key); exit()
        if torch.isnan(Tensor_Z).any():
            print('!!!!!!!!!!!!!! NAN value in xyz file %s !!!!!!!!!!'   % img_key); exit()
        if torch.isinf(Tensor_X).any():
            print('!!!!!!!!!!!!!! INF value in image file %s !!!!!!!!!!' % img_key); exit()
        if torch.isinf(Tensor_Y).any():
            print('!!!!!!!!!!!!!! INF value in illum file %s !!!!!!!!!!' % img_key); exit()
        if torch.isinf(Tensor_Z).any():
            print('!!!!!!!!!!!!!! INF value in xyz file %s !!!!!!!!!!'   % img_key); exit()

        return Tensor_X, Tensor_Y, Tensor_Z, img_key

    def __len__(self):
        return len(self.image_paths)


class EvalSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.image_paths = []
        self.illum_paths = []

        for name in self.args.image_list:
            ds = getattr(m, name)()   # ONE instance
            illum_key_set = set(ds.keys_val_L)
            for img_key in ds.keys_val_image:
                illum_key = _resolve_illum_key(img_key, illum_key_set)
                self.image_paths.append((ds.val_image_path, img_key))
                self.illum_paths.append((ds.val_illum_path, illum_key))

    def _reader(self):
        if not hasattr(self, '_lazy'):
            self._lazy = _LazyH5Reader()
        return self._lazy

    def __getitem__(self, index):
        img_path, img_key     = self.image_paths[index]
        illum_path, illum_key = self.illum_paths[index]
        reader = self._reader()
        W, H = 256, 256

        image = reader.get(img_path, img_key)
        image = np.array(image, dtype=np.float32)
        if image.shape[2] == 16:
            image = np.delete(image, 15, axis=2)
        image = resize(image, (W, H))   # resize once only

        illum = reader.get(illum_path, illum_key)
        illum = np.array(illum, dtype=np.float32).reshape(-1)
        illum = illum / illum.max()

        xyz_fromillum = np.matmul(illum.squeeze(), cmf_36).transpose()
        xyz_fromillum = xyz_fromillum / xyz_fromillum.max()
        xyz_fromillum = np.clip(xyz_fromillum, 0, xyz_fromillum.max())
        gt_rgb = torch.tensor(xyz_fromillum)

        input_image = np.clip(image, 0, image.max())
        input_image = input_image / input_image.max()
        input_image = input_image.transpose((2, 0, 1))

        Tensor_X = torch.from_numpy(input_image).float()
        Tensor_Y = torch.from_numpy(illum).float()
        Tensor_Z = torch.tensor(gt_rgb).float()

        return Tensor_X, Tensor_Y, Tensor_Z, img_key

    def __len__(self):
        return len(self.image_paths)


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.image_paths = []
        self.illum_paths = []

        for name in self.args.image_list:
            ds = getattr(m, name)()   # ONE instance
            illum_key_set = set(ds.keys_test_L)
            for img_key in ds.keys_test_image:
                illum_key = _resolve_illum_key(img_key, illum_key_set)
                self.image_paths.append((ds.test_image_path, img_key))
                self.illum_paths.append((ds.test_illum_path, illum_key))

    def _reader(self):
        if not hasattr(self, '_lazy'):
            self._lazy = _LazyH5Reader()
        return self._lazy

    def __getitem__(self, index):
        img_path, img_key     = self.image_paths[index]
        illum_path, illum_key = self.illum_paths[index]
        reader = self._reader()
        W, H = 256, 256

        image = reader.get(img_path, img_key)
        image = np.array(image, dtype=np.float32)
        if image.shape[2] == 16:
            image = np.delete(image, 15, axis=2)
        image = resize(image, (W, H))   # resize once only

        illum = reader.get(illum_path, illum_key)
        illum = np.array(illum, dtype=np.float32).reshape(-1)
        illum = illum / illum.max()

        xyz_fromillum = np.matmul(illum.squeeze(), cmf_36).transpose()
        xyz_fromillum = xyz_fromillum / xyz_fromillum.max()
        xyz_fromillum = np.clip(xyz_fromillum, 0, xyz_fromillum.max())
        gt_rgb = torch.tensor(xyz_fromillum)

        input_image = np.clip(image, 0, image.max())
        input_image = input_image / input_image.max()
        input_image = input_image.transpose((2, 0, 1))

        Tensor_X = torch.from_numpy(input_image).float()
        Tensor_Y = torch.from_numpy(illum).float()
        Tensor_Z = torch.tensor(gt_rgb).float()

        return Tensor_X, Tensor_Y, Tensor_Z, img_key

    def __len__(self):
        return len(self.image_paths)
