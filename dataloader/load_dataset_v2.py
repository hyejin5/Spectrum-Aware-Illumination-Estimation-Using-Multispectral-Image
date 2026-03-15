import h5py

path_dataset_beyondRGB = './dataset/BeyondRGB/'
path_dataset_MILD      = './dataset/MILD/'
path_dataset_single    = './dataset/MILD/Single/'

dataset_name = 'beyondRGB'
train_fold0_image = '%s_train_image.hdf5' % dataset_name
train_fold0_L     = '%s_train_illum.hdf5' % dataset_name
val_fold0_image   = '%s_val_image.hdf5'   % dataset_name
val_fold0_L       = '%s_val_illum.hdf5'   % dataset_name
test_fold0_image  = '%s_test_image.hdf5'  % dataset_name
test_fold0_L      = '%s_test_illum.hdf5'  % dataset_name


def _get_keys(path):
    """Open h5py file briefly to read keys, then close immediately."""
    with h5py.File(path, 'r') as f:
        return list(f.keys())


class BeyondRGB():
    def __init__(self):
        self.train_image_path = path_dataset_beyondRGB + train_fold0_image
        self.train_illum_path = path_dataset_beyondRGB + train_fold0_L
        self.keys_train_image = _get_keys(self.train_image_path)
        self.keys_train_L     = _get_keys(self.train_illum_path)

        self.val_image_path = path_dataset_beyondRGB + val_fold0_image
        self.val_illum_path = path_dataset_beyondRGB + val_fold0_L
        self.keys_val_image = _get_keys(self.val_image_path)
        self.keys_val_L     = _get_keys(self.val_illum_path)

        self.test_image_path = path_dataset_beyondRGB + test_fold0_image
        self.test_illum_path = path_dataset_beyondRGB + test_fold0_L
        self.keys_test_image = _get_keys(self.test_image_path)
        self.keys_test_L     = _get_keys(self.test_illum_path)


class indoor():
    def __init__(self):
        self.train_image_path = path_dataset_MILD + 'indoor_train_image.hdf5'
        self.train_illum_path = path_dataset_MILD + 'indoor_train_illum.hdf5'
        self.keys_train_image = _get_keys(self.train_image_path)
        self.keys_train_L     = _get_keys(self.train_illum_path)

        self.val_image_path = path_dataset_MILD + 'indoor_val_image.hdf5'
        self.val_illum_path = path_dataset_MILD + 'indoor_val_illum.hdf5'
        self.keys_val_image = _get_keys(self.val_image_path)
        self.keys_val_L     = _get_keys(self.val_illum_path)

        self.test_image_path = path_dataset_MILD + 'indoor_test_image.hdf5'
        self.test_illum_path = path_dataset_MILD + 'indoor_test_illum.hdf5'
        self.keys_test_image = _get_keys(self.test_image_path)
        self.keys_test_L     = _get_keys(self.test_illum_path)


class indoor_v2():
    def __init__(self):
        self.train_image_path = path_dataset_MILD + 'indoor_v2_train_image.hdf5'
        self.train_illum_path = path_dataset_MILD + 'indoor_v2_train_illum.hdf5'
        self.keys_train_image = _get_keys(self.train_image_path)
        self.keys_train_L     = _get_keys(self.train_illum_path)

        self.val_image_path = path_dataset_MILD + 'indoor_v2_val_image.hdf5'
        self.val_illum_path = path_dataset_MILD + 'indoor_v2_val_illum.hdf5'
        self.keys_val_image = _get_keys(self.val_image_path)
        self.keys_val_L     = _get_keys(self.val_illum_path)

        self.test_image_path = path_dataset_MILD + 'indoor_v2_test_image.hdf5'
        self.test_illum_path = path_dataset_MILD + 'indoor_v2_test_illum.hdf5'
        self.keys_test_image = _get_keys(self.test_image_path)
        self.keys_test_L     = _get_keys(self.test_illum_path)


class ECDRI_1st():
    def __init__(self):
        self.train_image_path = path_dataset_MILD + 'ECDRI_1st_train_image.hdf5'
        self.train_illum_path = path_dataset_MILD + 'ECDRI_1st_train_illum.hdf5'
        self.keys_train_image = _get_keys(self.train_image_path)
        self.keys_train_L     = _get_keys(self.train_illum_path)

        self.val_image_path = path_dataset_MILD + 'ECDRI_1st_val_image.hdf5'
        self.val_illum_path = path_dataset_MILD + 'ECDRI_1st_val_illum.hdf5'
        self.keys_val_image = _get_keys(self.val_image_path)
        self.keys_val_L     = _get_keys(self.val_illum_path)

        self.test_image_path = path_dataset_MILD + 'ECDRI_1st_test_image.hdf5'
        self.test_illum_path = path_dataset_MILD + 'ECDRI_1st_test_illum.hdf5'
        self.keys_test_image = _get_keys(self.test_image_path)
        self.keys_test_L     = _get_keys(self.test_illum_path)


class ECDRI_3rd():
    def __init__(self):
        self.train_image_path = path_dataset_MILD + 'ECDRI_3rd_train_image.hdf5'
        self.train_illum_path = path_dataset_MILD + 'ECDRI_3rd_train_illum.hdf5'
        self.keys_train_image = _get_keys(self.train_image_path)
        self.keys_train_L     = _get_keys(self.train_illum_path)

        self.val_image_path = path_dataset_MILD + 'ECDRI_3rd_val_image.hdf5'
        self.val_illum_path = path_dataset_MILD + 'ECDRI_3rd_val_illum.hdf5'
        self.keys_val_image = _get_keys(self.val_image_path)
        self.keys_val_L     = _get_keys(self.val_illum_path)

        self.test_image_path = path_dataset_MILD + 'ECDRI_3rd_test_image.hdf5'
        self.test_illum_path = path_dataset_MILD + 'ECDRI_3rd_test_illum.hdf5'
        self.keys_test_image = _get_keys(self.test_image_path)
        self.keys_test_L     = _get_keys(self.test_illum_path)


class ECDRI_7th():
    def __init__(self):
        self.train_image_path = path_dataset_MILD + 'ECDRI_7th_train_image.hdf5'
        self.train_illum_path = path_dataset_MILD + 'ECDRI_7th_train_illum.hdf5'
        self.keys_train_image = _get_keys(self.train_image_path)
        self.keys_train_L     = _get_keys(self.train_illum_path)

        self.val_image_path = path_dataset_MILD + 'ECDRI_7th_val_image.hdf5'
        self.val_illum_path = path_dataset_MILD + 'ECDRI_7th_val_illum.hdf5'
        self.keys_val_image = _get_keys(self.val_image_path)
        self.keys_val_L     = _get_keys(self.val_illum_path)

        self.test_image_path = path_dataset_MILD + 'ECDRI_7th_test_image.hdf5'
        self.test_illum_path = path_dataset_MILD + 'ECDRI_7th_test_illum.hdf5'
        self.keys_test_image = _get_keys(self.test_image_path)
        self.keys_test_L     = _get_keys(self.test_illum_path)


class ECDRI_single():
    def __init__(self):
        self.train_image_path = path_dataset_single + 'ECDRI_single_train_image.hdf5'
        self.train_illum_path = path_dataset_single + 'ECDRI_single_train_illum.hdf5'
        self.keys_train_image = _get_keys(self.train_image_path)
        self.keys_train_L     = _get_keys(self.train_illum_path)

        self.val_image_path = path_dataset_single + 'ECDRI_single_val_image.hdf5'
        self.val_illum_path = path_dataset_single + 'ECDRI_single_val_illum.hdf5'
        self.keys_val_image = _get_keys(self.val_image_path)
        self.keys_val_L     = _get_keys(self.val_illum_path)

        self.test_image_path = path_dataset_single + 'ECDRI_single_test_image.hdf5'
        self.test_illum_path = path_dataset_single + 'ECDRI_single_test_illum.hdf5'
        self.keys_test_image = _get_keys(self.test_image_path)
        self.keys_test_L     = _get_keys(self.test_illum_path)
