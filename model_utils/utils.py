import math
import numpy as np
import logging
import os
import shutil
import torch
import torch.nn as nn
from scipy import io
from model_utils.option import args
import pdb
import statistics

# ccm = io.loadmat('./model_utils/MSI2XYZ_CCM_15.mat') #color matching function file
ccm = io.loadmat('./model_utils/UNIST_CCM.mat') #color matching function file
# pdb.set_trace()

cmf_36 = io.loadmat('./model_utils/cmf_36.mat') #color matching function file
# cmf_36= cmf_36['cmf_36']
cmf_31 = io.loadmat('./CMF_31.mat')
cmf_31= cmf_31['CMF_31']

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler



def statistics_AE(ang_lst):
    num_25 = max(1, round((len(ang_lst)*0.25)))
    num_75 = min(len(ang_lst)-1, round((len(ang_lst)*0.75)))

    ang_lst.sort(reverse=True) #AE_hyper
    worst_cos = ang_lst[0]
    best_cos = ang_lst[-1]
    worst25_cos = sum(ang_lst[:num_25])/len(ang_lst[:num_25])
    best75_cos = sum(ang_lst[num_25:])/len(ang_lst[num_25:])
    best25_cos = sum(ang_lst[num_75:])/len(ang_lst[num_75:])
    med_hyper = statistics.median(ang_lst)
    trimean=0.25 * (ang_lst[num_25] + 2 * med_hyper + ang_lst[num_75])
    average_cos = sum(ang_lst)/len(ang_lst) 
    std = np.std(ang_lst) 
    # Angular_xyz_lst.sort(reverse=True)
    # average_AE_xyz = sum(Angular_xyz_lst) / len(Angular_xyz_lst)
    # worst_AE_xyz = Angular_xyz_lst[0]
    # best_AE_xyz = Angular_xyz_lst[-1]
    # worst25_AE_xyz = sum(Angular_xyz_lst[:num_25])/len(Angular_xyz_lst[:num_25])
    # best75_AE_xyz = sum(Angular_xyz_lst[num_25:])/len(Angular_xyz_lst[num_25:])
    # best25_AE_xyz = sum(Angular_xyz_lst[num_75:])/len(Angular_xyz_lst[num_75:])

    # med_xyz = statistics.median(Angular_xyz_lst)

    return average_cos,med_hyper,worst25_cos,best25_cos,best75_cos,worst_cos,best_cos, std, trimean


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        if (not args.reset):
            #raise SystemExit('Error: save_dir "' + args.save_dir + '" already exists! Please set --reset True to delete the folder.')
            print('Load: save_dir "' + args.save_dir + '" already exists!."')
        else:
            shutil.rmtree(args.save_dir)
    else:
        os.makedirs(args.save_dir)
    # os.makedirs(os.path.join(args.save_dir, 'img'))

    if ((not args.eval) and (not args.test)):
        if (not os.path.join(args.save_dir, 'model')):
            os.makedirs(os.path.join(args.save_dir, 'model'))
    
    if ((args.eval and args.eval_save_results) or args.test):
        if (not os.path.join(args.save_dir, 'save_results')):
            os.makedirs(os.path.join(args.save_dir, 'save_results'))

    args_file = open(os.path.join(args.save_dir, 'args_%s.txt'%args.rand), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name), 
        logger_name=args.logger_name).get_log()

    return _logger


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False

# Mat_XYZ_to_sRGB = np.array([
#     [3.2406, -1.5372, -0.4986],
#     [-0.9689, 1.8758, 0.0415],
#     [0.0557, -0.2040, 1.0570]
# ])

# Mat_XYZ_to_cameraRGB = np.array([
#     [1.3301, -0.6162, -0.1582],
#     [-0.4316, 1.4482, 0.0713],
#     [-0.0537, 0.2139, 0.5419]
# ])

# def hyper_to_RGB(Im, ccm):  #ccm : (3*16)
#     ccm = ccm['HSI2XYZ'] # double
#     Im = np.dot(Im,np.transpose(ccm))
#     rgb = Mat_XYZ_to_cameraRGB

#     rgb[0] = rgb[0]/max(rgb[0])
#     rgb[1] = rgb[1]/max(rgb[1])
#     rgb[2] = rgb[2]/max(rgb[2])
    
#     height, width, channel = Im.shape

#     Temp = np.ones(shape=(height,width,channel))
#     TempR = Temp*rgb[0]
#     TempG = Temp*rgb[1]   
#     TempB = Temp*rgb[2]

#     RGB = np.zeros(shape=(height,width,3))
#     RGB[:,:,0] = (Im*TempR).sum(axis=2) #channel wise sum
#     RGB[:,:,0] = RGB[:,:,0].clip(0)

#     RGB[:,:,1] = (Im*TempG).sum(axis=2)
#     RGB[:,:,1] = RGB[:,:,1].clip(0)

#     RGB[:,:,2] = (Im*TempB).sum(axis=2)
#     RGB[:,:,2] = RGB[:,:,2].clip(0)

#     RGB = RGB/RGB.max()

#     return RGB


Mat_XYZ_to_sRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
])

Mat_XYZ_to_cameraRGB = np.array([
    [1.3877, -0.7256, -0.0898],
    [-0.2979, 1.2793, 0.1367],
    [0.0098, 0.1963, 0.5410]
])

def hyper_to_RGB(Im, ccm):  # Im:(height*width*15) # ccm : (15*3)
    # pdb.set_trace()
    ccm = ccm['UNIST_CCM']
    rgb_mat = Mat_XYZ_to_cameraRGB

    height, width, channel = Im.shape
    Im_flat = np.reshape(Im,(height*width,channel))
    Im_xyz_flat = np.dot(Im_flat,ccm)
    Im_rgb_flat = np.dot(Im_xyz_flat,rgb_mat)
    Im_rgb = np.reshape(Im_rgb_flat,(height,width,3))
    Im_rgb=Im_rgb.clip(0)
    RGB = Im_rgb/Im_rgb.max()

    return RGB


def hyper2xyz_illum(illum, cmf_36):  # illum : (1*16), ccm : (3*16)
    # ccm = ccm['UNIST_CCM']
    # ccm = np.transpose(ccm)
    cmf_36= cmf_36['cmf_36']

    np.reshape(illum, (1,36))
    cmf_36 = torch.from_numpy(cmf_36)
    cmf_36 = cmf_36.type(torch.FloatTensor)

    # rgb = torch.from_numpy(rgb)
    # rgb = rgb.type(torch.FloatTensor)

    xyz=np.matmul(illum, cmf_36)  # -> tensor
    # RGB=np.matmul(illum, rgb)  # -> tensor

    return xyz


def hyper2xyz_illum_15(illum, ccm):  # illum : (1*16), ccm : (3*16)
    ccm = ccm['UNIST_CCM']
    # ccm = np.transpose(ccm)
    # cmf_36= cmf_36['cmf_36']

    np.reshape(illum, (1,15))
    ccm = torch.from_numpy(ccm)
    ccm = ccm.type(torch.FloatTensor)

    # rgb = torch.from_numpy(rgb)
    # rgb = rgb.type(torch.FloatTensor)

    xyz=np.matmul(illum, ccm)  # -> tensor
    # RGB=np.matmul(illum, rgb)  # -> tensor

    return xyz


def hyper2xyz_ref(ref, ccm):  # illum : (1*16)
    ccm = ccm['UNIST_CCM']
    # ccm = np.transpose(ccm)
    ref=ref.squeeze()  #ref: (W,H,15)

    ccm = torch.from_numpy(ccm)
    ccm = ccm.type(torch.FloatTensor)

    # rgb = torch.from_numpy(rgb)
    # rgb = rgb.type(torch.FloatTensor)

    ref_xyz=np.matmul(ref, ccm)  # -> tensor
    # RGB=np.matmul(illum, rgb)  # -> tensor
    # pdb.set_trace()

    return ref_xyz



def hyper2xyz_illum_batch(illum, cmf_36):  # illum : (1*16), ccm : (3*16)
    # ccm = ccm['UNIST_CCM']
    # ccm = np.transpose(ccm)
    cmf_36= cmf_36['cmf_36']

    # rgb = np.dot(Mat_XYZ_to_cameraRGB, ccm)
    np.reshape(illum, (illum.size()[0], 1,36))
    cmf_36 = torch.from_numpy(cmf_36)
    cmf_36 = cmf_36.type(torch.FloatTensor)

    # rgb = torch.from_numpy(rgb)
    # rgb = rgb.type(torch.FloatTensor)

    xyz=np.matmul(illum, cmf_36)  # -> tensor
    # sRGB=np.matmul(illum, rgb)  # -> tensor
    # pdb.set_trace()

    return xyz

def hyper2xyz_illum_train_15(illum, ccm):  # illum : (1*16), ccm : (3*16)
    ccm = ccm['UNIST_CCM']
    # ccm = np.transpose(ccm)

    # rgb = np.dot(Mat_XYZ_to_cameraRGB, ccm)
    np.reshape(illum, (illum.size()[0], 1,15))
    ccm = torch.from_numpy(ccm)
    ccm = ccm.type(torch.FloatTensor)

    # rgb = torch.from_numpy(rgb)
    # rgb = rgb.type(torch.FloatTensor)

    xyz=np.matmul(illum, ccm)  # -> tensor
    # sRGB=np.matmul(illum, rgb)  # -> tensor
    # pdb.set_trace()

    return xyz



def calc_psnr(gt_im, output_im):
    '''gt'''
    gt_image = gt_im.detach()
    gt_image = gt_image.squeeze().cpu().numpy()
    gt_image = gt_image.transpose((1,2,0))
    srgb_gt_8bit = gt_image *255
    '''output'''
    srgb_output_8bit = output_im * 255

    mse = np.mean((srgb_gt_8bit-srgb_output_8bit)**2)
    psnr = 20*math.log10(255/math.sqrt(mse))
    
    return psnr #, gt_image, output_image

def ref_illum2image(ref, illum):  #ref:batch,W,H,31  illum: 1,31
    image_lst=list()
    ref=hyper2xyz_ref(ref,ccm)
    print(ref.shape)

    illum = illum*1023
    for i in range(np.shape(illum)[1]) : #illum:1*31
        ref_single=ref[:,:,i]*illum[0,i]     
        image_lst.append(ref_single)
    image = torch.stack(image_lst,dim=2)
    image=np.array(image, dtype=np.float32)
    image = image/image.max()
    return image


