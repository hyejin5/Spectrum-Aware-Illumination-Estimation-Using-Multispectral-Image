import torch
import numpy as np
import pdb
import random
import math
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")

class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['image1'] = np.rot90(sample['image1'], k1).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 0):
            sample['image1'] = np.fliplr(sample['image1']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['image1'] = np.flipud(sample['image1']).copy()
            
        return sample

def rancrop(image,crop_width,crop_height):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return x, y

def RandomCrop(sample) :
    scale = [0.1, 1.0]
    scale = math.exp(random.random() * math.log(scale[1] / scale[0])) * scale[0]
    Y, X, _ = np.shape(sample)
    sample = resize(sample,(512,512))
    cropsize = min(max( int(round(min(X, Y) * scale)), 128), min(X, Y))

    strat_x,start_y=rancrop(sample,cropsize,cropsize)
    crop_sample = sample[start_y: start_y + cropsize, strat_x: strat_x + cropsize, :]
    crop_image = resize(crop_sample,(256,256))

    return crop_image

def RandomCrop_224(sample) :
    scale = [0.1, 1.0]
    scale = math.exp(random.random() * math.log(scale[1] / scale[0])) * scale[0]
    Y, X, _ = np.shape(sample)
    sample = resize(sample,(512,512))
    cropsize = min(max( int(round(min(X, Y) * scale)),128 ), min(X, Y))

    strat_x,start_y=rancrop(sample,cropsize,cropsize)
    crop_sample = sample[start_y: start_y + cropsize, strat_x: strat_x + cropsize, :]
    crop_image = resize(crop_sample,(224,224))

    return crop_image


def RandomCrop_224_wmask(sample,mask) :
    scale = [0.1, 1.0]
    scale = math.exp(random.random() * math.log(scale[1] / scale[0])) * scale[0]
    Y, X, _ = np.shape(sample)
    sample = resize(sample,(512,512))
    mask =resize(mask,(512,512))
    cropsize = min(max( int(round(min(X, Y) * scale)),128 ), min(X, Y))

    strat_x,start_y=rancrop(sample,cropsize,cropsize)
    crop_image = sample[start_y: start_y + cropsize, strat_x: strat_x + cropsize, :]
    crop_image = resize(crop_image,(224,224))
    
    crop_mask = mask[start_y: start_y + cropsize, strat_x: strat_x + cropsize, :]
    crop_mask = resize(crop_mask,(224,224))

    return crop_image, crop_mask


def rancrop_mid(image):
    max_x = int(image.shape[1]*0.1)
    max_y = int(image.shape[0]*0.1)
    mid_x = int(image.shape[1]*0.5)
    mid_y = int(image.shape[0]*0.5)
    t = random.randint(1,2)
    x = mid_x+((-1)**t)*(random.randint(0, max_x))
    y = mid_y+((-1)**t)*(random.randint(0, max_y))
    return x, y

def rancrop_mid_50(image):
    max_x = int(image.shape[1]*0.25)
    max_y = int(image.shape[0]*0.25)
    mid_x = int(image.shape[1]*0.5)
    mid_y = int(image.shape[0]*0.5)
    t = random.randint(1,2)
    x = mid_x+((-1)**t)*(random.randint(0, max_x))
    y = mid_y+((-1)**t)*(random.randint(0, max_y))
    return x, y


def RandomCrop_30_mid10(sample) :
    Y, X, _ = np.shape(sample)
    cropsize_x = int(0.3*X)
    cropsize_y = int(0.3*Y)

    strat_x,start_y=rancrop_mid(sample)

    crop_sample = sample[start_y-int(cropsize_y*0.5): start_y +int(cropsize_y*0.5), strat_x-int(cropsize_x*0.5): strat_x + int(cropsize_x*0.5), :]

    return crop_sample


def RandomCrop_30_mid50(sample) :
    Y, X, _ = np.shape(sample)
    cropsize_x = int(0.3*X)
    cropsize_y = int(0.3*Y)

    strat_x,start_y=rancrop_mid_50(sample)

    crop_sample = sample[start_y-int(cropsize_y*0.5): start_y +int(cropsize_y*0.5), strat_x-int(cropsize_x*0.5): strat_x + int(cropsize_x*0.5), :]

    return crop_sample

def RandomCrop_50_mid10(sample) :
    Y, X, _ = np.shape(sample)
    cropsize_x = int(0.5*X)
    cropsize_y = int(0.5*Y)

    strat_x,start_y=rancrop_mid(sample)

    crop_sample = sample[start_y-int(cropsize_y*0.5): start_y +int(cropsize_y*0.5), strat_x-int(cropsize_x*0.5): strat_x + int(cropsize_x*0.5), :]

    return crop_sample

def RandomCrop_50_mid50(sample) :
    Y, X, _ = np.shape(sample)
    cropsize_x = int(0.5*X)
    cropsize_y = int(0.5*Y)

    strat_x,start_y=rancrop_mid_50(sample)

    crop_sample = sample[start_y-int(cropsize_y*0.5): start_y +int(cropsize_y*0.5), strat_x-int(cropsize_x*0.5): strat_x + int(cropsize_x*0.5), :]

    return crop_sample


def midCrop30(sample) :
    Y, X, _ = np.shape(sample)
    cropsize_x = int(0.3*X)
    cropsize_y = int(0.3*Y)
    mid_x = int(X*0.5)
    mid_y = int(Y*0.5)

    crop_sample = sample[mid_y-int(cropsize_y*0.5): mid_y +int(cropsize_y*0.5), mid_x-int(cropsize_x*0.5): mid_x + int(cropsize_x*0.5), :]
    return crop_sample

def midCrop50(sample) :
    Y, X, _ = np.shape(sample)
    cropsize_x = int(0.5*X)
    cropsize_y = int(0.5*Y)
    mid_x = int(X*0.5)
    mid_y = int(Y*0.5)

    crop_sample = sample[mid_y-int(cropsize_y*0.5): mid_y +int(cropsize_y*0.5), mid_x-int(cropsize_x*0.5): mid_x + int(cropsize_x*0.5), :]
    return crop_sample

def expand_dims(x):
    return np.expand_dims(x,axis=2)

class ToTensor(object):
    def __call__(self, sample):
        
        image1 = sample['image1']
        # image1 = exp_tile(image1)
        image1 = image1.transpose((2,0,1))
        image1 = torch.from_numpy(image1).float()
        return {'image1':image1}


