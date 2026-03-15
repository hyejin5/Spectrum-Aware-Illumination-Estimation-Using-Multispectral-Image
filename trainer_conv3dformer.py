from model_utils import utils 
import h5py
import torch
import time
from math import *
import os
import numpy as np
import torch 
import torch.optim as optim
import pdb
from model_utils import visualization
import math
import shutil
import statistics
import pandas as pd
import colour
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from model_utils.utils import CosineAnnealingWarmupRestarts, statistics_AE
from torchsummary import summary

torch.autograd.set_detect_anomaly(True)

class Trainer():
    def __init__(self, args, logger, dataloader, model,  loss_all, device):
        self.model = model
        self.device=device
        # self.model = model.to(device)
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.loss = loss_all
        # self.optimizer = optim.Adam(self.model.parameters(), self.args.lr_rate)
        self.optimizer = AdamW(self.model.parameters(), self.args.lr_rate, eps=1e-6)
        self.loss_cos = self.loss['cos_loss'].to(device)

        # self.cosine_scheduler = utils.CosineAnnealingWarmUpRestarts(self.optimizer, T_0=100, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.4)
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=500, cycle_mult=1, max_lr=1e-2, min_lr=5e-6, warmup_steps=20, gamma=0.5)
    
        '''load model'''
        self.load(model_path=self.args.model_path)
        

    def load(self, model_path):
        load_model=self.args.load_model
        if (load_model):
            if (not self.args.test):
                self.logger.info('load_model_path: ' + model_path)

                model_data=torch.load(model_path)
                #self.model.module.load_state_dict(model_data['model_state_dict'])
                self.model.load_state_dict(model_data['model_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                self.args.now_epochs = model_data['epoch'] + 1
                self.args.best_val_loss = model_data['best_val_loss']
                self.scheduler.load_state_dict(model_data['scheduler_state_dict'])
                self.logger.info('model loading... rand: %s, continuing epoch: %d ' %(self.args.rand, self.args.now_epochs))

            else :
                self.logger.info('load_model_path: ' + model_path)
                model_data=torch.load(model_path)
                self.model.load_state_dict(model_data['model_state_dict'])
                self.logger.info('model loading for test... rand: %s' %(self.args.rand))
                print(self.model)
                
            
    def train(self, current_epoch=0, is_init=False):
        
        self.model.train()
        running_loss=0.0

        running_cos_loss=0.0
        running_cos_loss_xyz=0.0
        batch_cos = 0.0 
        batch_cos_xyz = 0.0
        batch_loss = 0.0
        iter_start_time=time.time()

        illum_save_path = str()
        rgb_save_path = str()
        Angular_xyz_lst = list() 

        illum_save_path = './training_illum/%s/' %self.args.rand
        rgb_save_path = './training_rgb/%s/' %self.args.rand

        if os.path.exists(illum_save_path): 
            shutil.rmtree(illum_save_path)
        if not os.path.exists(illum_save_path): 
            os.makedirs(illum_save_path)

        if os.path.exists(rgb_save_path): 
            shutil.rmtree(rgb_save_path)

        if not os.path.exists(rgb_save_path): 
            os.makedirs(rgb_save_path)
        if (not is_init):
            self.scheduler.step()
        print('Current Epoch: %d' %current_epoch)
        for batch_idx, data in enumerate(self.dataloader['train']):
            input_image,gt_spectro, gt_xyz, image_name = data

            input_image = input_image.to(self.device)
            gt_spectro=gt_spectro.view([-1,36]).to(self.device)  #batch=50; gt_L.size()=torch.Size([50, 16]
            gt_xyz = gt_xyz.view([-1,3]).to(self.device)
            
            self.optimizer.zero_grad()
            output=self.model(input_image)
            output = output.view([-1,36])
            output_norm=output.clone()

            for i in range(output.shape[0]):
                max_val = max(output[i].max(),1e-8)
                output_norm[i]=(output[i]/max_val)
            im_name = "%s" %(image_name[0])

            output_norm = output_norm.to(gt_spectro.device)
            loss_cos, loss_lst = self.loss_cos(output_norm, gt_spectro)
            
            '''calculate AE in rgb'''
            output_xyz = utils.hyper2xyz_illum_batch(output_norm.detach().cpu(), utils.cmf_36)
            output_xyz = output_xyz.detach().cpu()
            output_xyz = output_xyz/max(output_xyz.max(),1e-8)

            output_xyz = output_xyz.to(gt_xyz.device)
            Angular_error_xyz, loss_lst_xyz = self.loss_cos(output_xyz, gt_xyz)

            Angular_xyz_lst.append(Angular_error_xyz.item())

            total_loss=self.args.wc*loss_cos+self.args.wc_xyz*Angular_error_xyz
            total_loss = torch.nan_to_num(total_loss)

            total_loss.backward(retain_graph=True)
            self.optimizer.step()

            batch_cos +=loss_cos.mean().item()
            batch_cos_xyz+=Angular_error_xyz.mean().item()
            batch_loss +=total_loss.mean().item()

            running_cos_loss=batch_cos/(batch_idx+1)
            running_cos_loss_xyz = batch_cos_xyz/(batch_idx+1)
            running_loss=batch_loss/(batch_idx+1)
        
            iter_finish_time=time.time()
            duration=iter_finish_time-iter_start_time

            print('iteration: %d, 20 iter_time: %d, AE_hyper: %.4f in degree: %.4f, AE_xyz: %.4f, Total loss: %.4f' %(batch_idx+1, duration, running_cos_loss, running_cos_loss, running_cos_loss_xyz,running_loss))
            iter_start_time=time.time()
            
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))
        return running_loss, self.optimizer.param_groups[0]['lr']



    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')  
        self.model.eval()
        val_loss = 0.0
        val_loss_value = 0.0
        ang_lst = list()
        Angular_xyz_lst = list() 
        
        with torch.no_grad():
            for idx, data in enumerate(self.dataloader['eval']):
                input_image, gt_spectro, gt_xyz, image_name = data
                input_image = input_image.to(self.device)

                gt_spectro = gt_spectro.view([-1,36]).to(self.device)
                gt_xyz = gt_xyz.view([-1,3]).to(self.device)

                output =self.model(input_image)
                
                output = output.view([-1,36])
                output_norm=output.clone()

                for i in range(output.shape[0]):
                    max_val = max(output[i].max(),1e-8)
                    output_norm[i]=(output[i]/max_val)

                output_norm = output_norm.to(gt_spectro.device)
                loss_cos, loss_lst = self.loss_cos(output_norm, gt_spectro)
                
                ang_lst.append(loss_cos.item())

                '''calculate AE in rgb'''
                output_norm_560=output_norm.clone()

                for i in range(output_norm.shape[0]):
                    output_norm_560[i]=(output_norm[i]/output_norm[i][18])

                output_xyz = utils.hyper2xyz_illum_batch(output_norm_560.detach().cpu(), utils.cmf_36)
                output_xyz = output_xyz.detach().cpu()
                output_xyz = output_xyz/output_xyz.max()
                
                output_xyz = output_xyz.to(gt_xyz.device)
                Angular_error_xyz, loss_lst_xyz = self.loss_cos(output_xyz, gt_xyz)

                Angular_xyz_lst.append(Angular_error_xyz.item())

                print('Angular loss: %4f, AE_xyz: %4f'%(loss_cos, Angular_error_xyz))
                total_loss = self.args.wc*loss_cos+self.args.wc_xyz*Angular_error_xyz

                val_loss += total_loss.item()

        val_loss_value = val_loss/ (idx+1) 
        average_cos = sum(ang_lst)/len(ang_lst)   
        average_AE_xyz = sum(Angular_xyz_lst) / len(Angular_xyz_lst) 

        self.logger.info(' Validation Loss: ' + str(round(val_loss_value,4))
            +' average AE of hyperspectrum: '+str(round(average_cos, 4))+' in degree: '+str(round(average_cos, 4)) \
            +' average AE in sRGB: '+str(round(average_AE_xyz, 4))+' in XYZ: '+str(round(average_AE_xyz, 4)))  

        return val_loss_value, average_cos, average_AE_xyz
    

    def test(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' Test process...')

        self.model.eval()

        val_loss = 0.0
        val_loss_value = 0.0
        ang_lst = list()
        ang_lst_16 = list()
        Angular_xyz_lst = list() 
        im_name_lst = list()

        test_output_name = self.args.test_output_name
        test_file_path = self.args.test_output_dir+'%s/' %self.args.rand

        if not os.path.exists(test_file_path): 
            os.makedirs(test_file_path)

        if self.args.test:

            test_output_name = test_output_name+'_%s_image_%s.hdf5' %(self.args.rand, self.args.image_list)

        file = h5py.File(test_file_path+test_output_name, 'w')
        file.close()
        illum_save_path = str()
        rgb_save_path = str()

        if self.test:
            illum_save_path = './test_illum/%s/test_image_%s/' %(self.args.rand, self.args.image_list)
            rgb_save_path = './test_rgb/%s/test_image_%s/' %(self.args.rand, self.args.image_list)

        else: 
            illum_save_path = './test_illum/%s/' %self.args.rand
            rgb_save_path = './test_rgb/%s/' %self.args.rand

        if os.path.exists(illum_save_path): 
            shutil.rmtree(illum_save_path)
        if not os.path.exists(illum_save_path): 
            os.makedirs(illum_save_path)

        if os.path.exists(rgb_save_path): 
            shutil.rmtree(rgb_save_path)

        if not os.path.exists(rgb_save_path): 
            os.makedirs(rgb_save_path)

        css = np.load('./model_utils/optimized_filters_responses_IRC.npy') #16*36
        css = torch.from_numpy(css)
        css_t = css.permute(1,0).contiguous().float()

        with torch.no_grad():
            for idx, data in enumerate(self.dataloader['test']):
                torch.autograd.set_detect_anomaly(True)
                
                input_image, gt_spectro, gt_xyz, image_name = data
                input_image = input_image.to(self.device)
                image_name=list(image_name)
                im_name_lst = im_name_lst+image_name

                gt_spectro = gt_spectro.view([-1,36]).to(self.device)
                          
                gt_xyz = gt_xyz.view([-1,3])

                output =self.model(input_image)
                output = output.view([-1,36])
                output_norm=output.clone()
                
                max_values = output_norm.max(dim=1, keepdim=True)[0]
                output_norm = output_norm/max_values
                                   
                output_norm = output_norm.to(gt_spectro.device)
                loss_cos, loss_lst = self.loss_cos(output_norm, gt_spectro)                    

                ang_lst=ang_lst+loss_lst
                
                gt_illum_16 = torch.einsum('sc,cd->sd',gt_spectro,css_t.to(gt_spectro.device)) #Convert 16CH image to 36CH image with css
                output_illm_16 = torch.einsum('sc,cd->sd',output_norm,css_t.to(gt_spectro.device)) #Convert 16CH image to 36CH image with css

                gt_illum_16 = gt_illum_16/gt_illum_16.max(dim=1, keepdim=True)[0]
                output_illm_16 = output_illm_16/output_illm_16.max(dim=1, keepdim=True)[0]
                AE_cos_16, loss_lst_16 = self.loss_cos(output_illm_16, gt_illum_16)
                
                ang_lst_16=ang_lst_16+loss_lst_16

                '''calculate AE in rgb'''
                output_norm_560=output_norm.clone()

                output_xyz = utils.hyper2xyz_illum_batch(output_norm_560.detach().cpu(), utils.cmf_36)
                output_xyz=output_xyz.detach().cpu()
                output_xyz = output_xyz/output_xyz.max()
                
                output_xyz = output_xyz.to(gt_xyz.device)
                Angular_error_xyz, loss_lst_xyz = self.loss_cos(output_xyz, gt_xyz)

                Angular_xyz_lst=Angular_xyz_lst+loss_lst_xyz

                print('Angular loss: %4f, AE_xyz: %4f'%(loss_cos, Angular_error_xyz))
                total_loss = self.args.wc*loss_cos+self.args.wc_xyz*Angular_error_xyz

                file = h5py.File(test_file_path+test_output_name, 'a')
                for i in range(input_image.shape[0]):
                    im_name = image_name[i]   
                    if im_name in list(file.keys()):
                        del file[im_name]
                    group = file.create_group(im_name)
                    group.create_dataset("gt_spectro", data=gt_spectro[i].detach().cpu())
                    group.create_dataset("image", data=input_image[i].detach().cpu())                    
                    group.create_dataset("output_spectrum", data=output_norm[i].detach().cpu())
                    group.create_dataset("output_xyz", data=output_xyz[i].detach().cpu())
                    group.create_dataset("output_16", data=output_illm_16[i].detach().cpu())

                file.close()

                self.logger.info(' Angular Loss of spectrum: '+str(round(loss_cos.item(), 4))+' in degree: '+str(round(loss_cos.item(), 4))\
                    +' Angular Error in sRGB: '+str(round(Angular_error_xyz.item(), 4)))
                val_loss += total_loss.item()
        val_loss_value = val_loss/ (idx+1)
        
        dict_result = dict()

        dict_result['im_name']=im_name_lst
        dict_result['AE_hyper']=ang_lst
        dict_result['AE_multi']=ang_lst_16
        dict_result['AE_xyz']=Angular_xyz_lst


        result = pd.DataFrame(dict_result)
        result.set_index('im_name', inplace=True)
        result.to_excel(self.args.save_dir+'%s_%s_result.xlsx'%(self.args.rand, self.args.image_list))

        average_cos,med_hyper,worst25_cos,best25_cos,best75_cos,worst_cos,best_cos, std_hyper, trimean_hyper = statistics_AE(ang_lst)
        average_AE_xyz,med_xyz,worst25_AE_xyz,best25_AE_xyz, best75_AE_xyz,worst_AE_xyz,best_AE_xyz, std_xyz, trimean_xyz = statistics_AE(Angular_xyz_lst)

        self.logger.info('------------------ ALL ------------------')
        self.logger.info('AE for all dataset included in this fold')
        self.logger.info('Average hyper AE: ' + str(round(average_cos,4)) + ' Average AE in xyz: ' + str(round(average_AE_xyz,4)))
        self.logger.info('Worst AE: ' + str(round(worst_cos,4)) + ' Worst AE in xyz: ' + str(round(worst_AE_xyz,4)))
        self.logger.info('Best AE: ' + str(round(best_cos,4)) + ' Best AE in xyz: ' + str(round(best_AE_xyz,4)))
        self.logger.info('Best 25% AE: ' + str(round(best25_cos,4)) + ' Best 25% AE in xyz: ' + str(round(best25_AE_xyz,4)))
        self.logger.info('Best 75% AE: ' + str(round(best75_cos,4)) + ' Best 75% AE in xyz: ' + str(round(best75_AE_xyz,4)))
        self.logger.info('Worst 25% AE: ' + str(round(worst25_cos,4)) + ' worst 25% AE in xyz: ' + str(round(worst25_AE_xyz,4)))
        self.logger.info('Median hyper AE: ' + str(round(med_hyper,4)) + ' median AE in xyz: ' + str(round(med_xyz,4)))
        self.logger.info('Standard deviation of hyper AE: ' + str(round(std_hyper,4)) + ' std of xyz AE: ' + str(round(std_xyz,4)))
        self.logger.info('Trimean of hyper AE: ' + str(round(trimean_hyper,4)) + ' trimean of xyz AE: ' + str(round(trimean_xyz,4)))

        return val_loss_value, average_cos,average_AE_xyz, gt_spectro.detach().cpu()



    def save_best_model(self, val_loss, current_epoch=0):
        if (val_loss < self.args.best_val_loss  < self.args.best_AE_rgb) :
            if not os.path.exists(self.args.best_model_dir): 
                os.makedirs(self.args.best_model_dir)
            best_dir = self.args.best_model_dir
            self.logger.info('saving the model...')
            self.logger.info('best validation loss: '+str(round(val_loss,4)))
            self.args.best_val_loss = val_loss
            best_epoch = current_epoch
            
            torch.save({
                    'epoch': current_epoch,
                    'best_val_loss': self.args.best_val_loss,
                    'model_state_dict': self.model.state_dict(),
                    #'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict' : self.scheduler.state_dict()
                    },best_dir+'best_Model_lr_%f_Batch_%d_%s.pt' %(self.args.lr_rate,self.args.batch_size, self.args.rand))
            
            val_loss_value, average_cos,average_AE_xyz, gt_spectro = self.test(best_epoch)

            self.logger.info('Best Epoch: ' + str(best_epoch))

            return

        else: return
