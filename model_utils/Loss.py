import torch.nn as nn
import torch
from model_utils.option import args
import pdb
import numpy as np
import math
class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        # GPU_NUM = args.gpu_num
        # self.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        # self.loss = nn.L1Loss().to(self.device)
        self.loss = nn.L1Loss()

    def forward(self, x, y) :
        loss = self.loss(x,y)

        return loss

class L2_loss(nn.Module):
    def __init__(self):
        super(L2_loss, self).__init__()
        # GPU_NUM = args.gpu_num
        # self.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

        # self.loss = nn.MSELoss().to(self.device)
        self.loss = nn.MSELoss()

    def forward(self, x, y) :
        loss = self.loss(x,y)

        return loss

class Angular_loss(nn.Module):
    def __init__(self):
        super(Angular_loss,self).__init__()
        # GPU_NUM = args.gpu_num
        # self.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

        self.loss = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, x, y) :
        '''''''''''''''''

        oss=self.loss(x, y)
        # output_lst = list()
        epsilon = 1e-8
        loss=torch.clamp(loss, -1 ,1)

        loss = torch.acos(loss-epsilon)

        angle = loss*(180/math.pi)
        AE_lst =list(angle.detach().cpu().numpy())


        avg_angle = torch.mean(angle).type(torch.float32)
        '''''''''''''''''

        safe_v = 0.999999
        illum_normalized1 = torch.nn.functional.normalize(x,dim=1)
        illum_normalized2 = torch.nn.functional.normalize(y,dim=1)
        dot = torch.sum(illum_normalized1*illum_normalized2,dim=1)
        dot = torch.clamp(dot, -safe_v, safe_v)
        angle = torch.acos(dot)*(180/math.pi)
        AE_lst =list(angle.detach().cpu().numpy())

        avg_angle = torch.mean(angle).type(torch.float32)
        
        # for i in range(x.shape[0]):
        #     dot = torch.dot(x[i],y[i])
        #     dot = dot.unsqueeze(dim=0)
        #     norm_x = torch.sqrt(torch.sum(torch.pow(x[i],2))+epsilon)
        #     norm_y =  torch.sqrt(torch.sum(torch.pow(y[i],2))+epsilon)
        #     norm_x = norm_x.unsqueeze(dim=0)
        #     norm_y = norm_y.unsqueeze(dim=0)
        #     angle = torch.acos(torch.clamp(dot/max(torch.dot(norm_x,norm_y),epsilon),-1+epsilon,1-epsilon)+epsilon)
        #     output_lst.append(angle)
        # # pdb.set_trace()
        # output_loss = torch.stack(output_lst, dim=0)
        # output_loss=torch.nan_to_num(output_loss,posinf=None, neginf=None)
        # angle = output_loss*(180/math.pi)
        # AE_lst =list(angle.detach().cpu().numpy())

        # avg_angle = angle*(180/math.pi)
        # AE_lst =list(angle.detach().cpu().numpy())
        # pdb.set_trace()
        return avg_angle, AE_lst
    
    
def get_angular_loss(vec1,vec2):
    safe_v = 0.999999
    illum_normalized1 = torch.nn.functional.normalize(vec1,dim=1)
    illum_normalized2 = torch.nn.functional.normalize(vec2,dim=1)
    dot = torch.sum(illum_normalized1*illum_normalized2,dim=1)
    dot = torch.clamp(dot, -safe_v, safe_v)
    angle = torch.acos(dot)*(180/math.pi)
    loss = torch.mean(angle)
    return loss
'''loss_all'''

def get_loss_dict():
    loss = {}

    loss['l1_loss'] = L1_loss()
    loss['l2_loss'] = L2_loss()
    loss['cos_loss'] = Angular_loss()
    
    return loss
