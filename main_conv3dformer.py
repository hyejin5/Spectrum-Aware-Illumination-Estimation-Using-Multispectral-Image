from model_utils.option import args
from model_utils.utils import mkExpDir
# from dataloader import dataloader as dataloader
from dataloader import dataloader_v2 as dataloader
from model_utils import Loss
from trainer_conv3dformer import Trainer
import torch.distributed as dist
import os
import torch
import warnings
from tensorboardX import SummaryWriter
import random
import string
import pdb
import scipy.io as sio
import math
from importlib import import_module
warnings.filterwarnings('ignore')
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    batch_size = args.batch_size
    num_worker = args.num_workers

    if args.load_model:
        model_path = args.model_path
        rand = model_path.split('_')[-1].split('.')[0]
        args.rand = rand + '_conti'
    else:
        args.rand = rand = "".join([random.choice(string.ascii_letters) for _ in range(10)])

    rand = args.rand

    if args.test:
        args.log_file_name = 'MS_ISE_conv3dformer_%s_test_dataset_%s.log'%(args.rand, args.image_list)
        args.logger_name = 'MS_ISE_conv3dformer_%s_test_dataset_%s.log'%(args.rand, args.image_list)
    elif args.load_model:
        args.log_file_name = '%s.log' % args.rand
        args.logger_name = args.rand

    _logger = mkExpDir(args)

    writer_path = "./runs/batch%d_%s/" % (args.batch_size, args.rand)
    if not os.path.exists(writer_path): 
        os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args)

    # args.model_name
    model_name = import_module('model.' + args.model_name)

    _model  = getattr(model_name, args.model_name)()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    device = torch.device('cuda')

   # _model = nn.DataParallel(_model, output_device=1)
    _model = _model.to(device)

    ### loss
    _loss_all = Loss.get_loss_dict()
    t = Trainer(args, _logger, _dataloader, _model, _loss_all, device)
    if args.test:
        val_loss_value, average_cos,average_AE_xyz, gt_xyz = t.test()
        print("\ntest Loss: {:.4f}, \n".format(val_loss_value))
        print("\nAngular Loss: {:.4f}, \n".format(average_cos))
        print("\nAngular Loss Degree: {:.4f}, \n".format(average_cos))

        _logger.info(' Test Loss: ' + str(round(val_loss_value,4))
            +' average AE of hyperspectrum: '+str(round(average_cos, 4))+' in degree: '+str(round(average_cos, 4))\
            +' average AE of xyz: '+str(round(average_AE_xyz, 4))+' in degree: '+str(round(average_AE_xyz, 4)))


    else :
        best_epoch=0
        test_loss_value=0.00

        for epoch in range(args.now_epochs, args.num_epochs+1):
            train_loss_value, lr = t.train(current_epoch=epoch, is_init=False)
            val_loss_value, AE_hyper, AE_xyz = t.evaluate(current_epoch=epoch)
            eval_value = (0.5*AE_hyper)+(0.5*AE_xyz)
            t.save_best_model(val_loss_value, current_epoch=epoch)

            writer.add_scalar('train_loss_%s' %rand,train_loss_value, epoch)
            writer.add_scalar('validation_loss_%s' %rand, val_loss_value, epoch)
            writer.add_scalar('Learning rate_%s'%rand, lr, epoch)

            if (epoch % 5 == 0) and args.rank==0: #save every 5 epoch
                _logger.info('saving the model...')

                check_dir = args.check_dir
                if not os.path.exists(check_dir): 
                    os.makedirs(check_dir)

                torch.save({
                        'epoch': epoch,
                        #'model_state_dict': t.model.module.state_dict(),
                        'model_state_dict': t.model.state_dict(),
                        'optimizer_state_dict': t.optimizer.state_dict(),
                        'loss': train_loss_value,
                        'scheduler_state_dict' : t.scheduler.state_dict()                        
                        },check_dir+'model_lr_%f_Batch_%d_%s.pt' %(args.lr_rate, args.batch_size, args.rand))
            
