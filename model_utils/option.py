import argparse

parser = argparse.ArgumentParser(description='PyTorch MSI_ISE_conv3dformer Training')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--gpu_ids', nargs='+', default=0,
                    help='The ids of gpu when Using multi-GPU')


parser.add_argument('--world_size', type=int, default=1,
                    help='The number of gpu when Using multi-GPU')
parser.add_argument('--rank', type=int, default=0,
                    help='The number of process when Using multi-GPU')
           
parser.add_argument('--wc', default=0.5, type=float)     
parser.add_argument('--wc_xyz', default=0.5, type=float)            

parser.add_argument('--best_val_loss', default=100, type=float)
parser.add_argument('--best_AE_rgb', default=120, type=float)       

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

'''Transformer hyperparameter'''
parser.add_argument('--num_head', default=8, type=int, help='# of Multi-head for Transformer')

parser.add_argument('--encoder_depth', default=6, type=int, help='# of depth for Transformer Encoder')
''''''
parser.add_argument('--rand', default=None, type=str)

### log setting
parser.add_argument('--save_dir', type=str, default='./save_dir/',
                    help='Directory to save log, arguments, models and images')

parser.add_argument('--log_file_name', type=str, default='EMPTY.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='EMPTY',
                    help='Logger name')

parser.add_argument('--test_output_name', type=str, default='CCNet_CAB_test_output',
                    help='Logger name')

parser.add_argument('--check_dir', type=str, default='./save_dir/check_dir/',
                    help='Directory to save checkpoint')  

parser.add_argument('--test_output_dir', type=str, default='./save_dir/test_output/',
                    help='Directory to save test output of best model')    

parser.add_argument('--best_model_dir', type=str, default='./save_dir/best_model/',
                    help='Directory to save best validation model')                     

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')

### dataset setting
parser.add_argument('--image_list', '-image', nargs='+', default=None,
                     help='List of image which is going to use for training, {ECDRI_mono, SAIT_0714, samsung_indoor, samsung_outdoor}') 

parser.add_argument('--origin_dataset_name', '-origin_data', default=None,
                     help='List of illumination which is going to use for test, {samsung_daylight, samsung_mono, samsung_mixed}')


### dataloader setting
parser.add_argument('--num_workers','-num_workers', type=int, default=4,
                    help='The number of workers when loading data')

### optimizer setting
parser.add_argument('--lr_rate', type=float, default=5e-3,
                    help='Learning rate')

parser.add_argument('--step_size', type=int, default=50,
                    help='Scheduler rate step size')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay')


### training setting
parser.add_argument('--model_name', type=str, default=None,
                    help='name of model used for training')

parser.add_argument('--stat_model', type=str, default=None,
                    help='name of model used for statistical illumination estimation')

parser.add_argument('--load_model',action="store_true",
                    help='Load model')

parser.add_argument('--model_path', type=str, default='./save_dir/best_model/',
                    help='Path of model')
parser.add_argument('--batch_size','-batch_size', type=int, default=50,
                    help='Training batch size')
parser.add_argument('--train_crop_size', type=int, default=512,
                    help='Training data crop size')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='The number of training epochs')
parser.add_argument('--now_epochs', type=int, default=1,
                    help='The number of training epochs')
parser.add_argument('--print_every', type=int, default=1,
                    help='Print period')
parser.add_argument('--save_every', type=int, default=1,
                    help='Save period')
parser.add_argument('--val_every', type=int, default=999999,
                    help='Validation period')

### evaluate
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')

parser.add_argument('--test',action="store_true",
                    help='Test mode')

parser.add_argument('--reset', type=str2bool, default=False,
                    help='Delete save_dir to create a new one')

parser.add_argument('--without_gap',action="store_true",
                    help='Use GAP or not')

parser.add_argument('--CAB_r', type=int, default=15,
                    help='Channel of FC layer of CAB')                    

parser.add_argument('--CABlock', type=str, default='channel_attention_block',
                    help='Type of CAB')

args = parser.parse_args()


