args = {}
import random
import numpy as np
import torch as t

args['path'] = './checkpoints/model/'
args['pre_path'] = './checkpoints/new'
args['name'] = '18-modelv860'

seed = 72
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

learning_rate = 0.0005
dataset = "ngsim"

args['num_worker'] = 8
args['device'] = device
args['lstm_encoder_size'] = 64
args['encoder_size'] = 32
args['n_head'] = 4
args['att_out'] = 48
args['in_length'] = 30
args['in_length_stu'] = 16
args['out_length'] = 25
args['f_length'] = 5
args['traj_linear_hidden'] = 16
args['batch_size'] = 128
args['use_elu'] = True
args['dropout'] = 0
args['relu'] = 0.1
args['lat_length'] = 3
args['lon_length'] = 3
args['use_true_man'] = False
args['epoch'] = 24
args['use_spatial'] = False
args['ran_miss_continue'] = False

args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['grid_size'] = (13, 3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 3
args['train_flag'] = False
args['in_channels'] = 64
args['out_channels'] = 64
args['kernel_size'] = 3
args['nbr_max'] = 39
args['hidden_channels'] = 128
args['alpha'] = 10
args['temp'] = 3
args['use_maneuvers'] = True
args['cat_pred'] = True
args['use_mse'] = False
args['pre_epoch'] = 16
args['val_use_mse'] = True
