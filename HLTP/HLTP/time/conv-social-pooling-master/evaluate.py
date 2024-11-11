from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSETest, maskedNLLTest
from torch.utils.data import DataLoader
import time
from thop import profile
from thop import clever_format

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13, 3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = False
args['train_flag'] = False

# Evaluation metric:
metric = 'rmse'  # or rmse

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/cslstm_m.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('data/TestSet.mat')
tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=True, num_workers=0, collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()

time_all = 0
sum_nbrs = 0
all_macs = 0
all_params = 0
for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

    if metric == 'nll':
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask, use_maneuvers=False)
    else:
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man * 3 + lat_man
                fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
        else:
            # TODO
            t = time.time()
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            time_all += time.time() - t
            sum_nbrs += 128
            if (sum_nbrs > 1000000):
                print(time_all / sum_nbrs)
            l, c = maskedMSETest(fut_pred, fut, op_mask)

    lossVals += l.detach()
    counts += c.detach()

if metric == 'nll':
    print(lossVals / counts)
else:
    print(torch.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
