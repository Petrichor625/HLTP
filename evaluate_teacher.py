from __future__ import print_function
import torch
from teacher_model.teacher_model import highwayNet
from loader2 import ngsimDataset
from utils import maskedNLL, maskedMSETest, maskedNLLTest
from torch.utils.data import DataLoader
from config import *
import time


class Evaluate_teacher():
    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1

    def main(self, name, hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                 va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix):
        model_step = 1
        args['train_flag'] = True
        l_path = args['pre_path']
        net = highwayNet(args=args)
        net.load_state_dict(t.load(l_path + '/epoch' + name + '.pth', map_location=device))
        net = net.to(device)
        # net.eval()

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0
        nbrsss = 0

        hist_batch = hist_batch.to(device)
        nbrs_batch = nbrs_batch.to(device)
        mask_batch = mask_batch.to(device)
        lat_enc_batch = lat_enc_batch.to(device)
        lon_enc_batch = lon_enc_batch.to(device)
        lane_batch = lane_batch.to(device)
        nbrslane_batch = nbrslane_batch.to(device)
        class_batch = class_batch.to(device)
        nbrsclass_batch = nbrsclass_batch.to(device)
        va_batch = va_batch.to(device)
        nbrsva_batch = nbrsva_batch.to(device)
        edge_index_batch = edge_index_batch.to(device)
        ve_matrix_batch = ve_matrix_batch.to(device)
        ac_matrix_batch = ac_matrix_batch.to(device)
        man_matrix_batch = man_matrix_batch.to(device)
        view_grip_batch = view_grip_batch.to(device)
        graph_matrix = graph_matrix.to(device)

        te = time.time()
        fut_pred, lat_pred, lon_pred = net(hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                 va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix)
        all_time += time.time() - te
        return fut_pred, lat_pred, lon_pred

class Evaluate():
    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1

    def main(self, epoch_num):
        args = {}
        args['use_cuda'] = True
        args['encoder_size'] = 64
        args['decoder_size'] = 128
        args['in_length'] = 30
        args['out_length'] = 25
        args['grid_size'] = (13, 3)
        args['soc_conv_depth'] = 64
        args['conv_3x1_depth'] = 16
        args['dyn_embedding_size'] = 32
        args['input_embedding_size'] = 32
        args['num_lat_classes'] = 3
        args['num_lon_classes'] = 3
        args['use_maneuvers'] = True
        args['train_flag'] = False
        args['in_channels'] = 64
        args['out_channels'] = 64
        args['kernel_size'] = 3
        args['n_head'] = 4
        args['att_out'] = 48
        args['dropout'] = 0.2
        args['nbr_max'] = 39
        args['hidden_channels'] = 128
        batch_size = 128

        metric = 'rmse'
        model_step = 1
        net = highwayNet(args)
        net.load_state_dict(torch.load('./checkpoints/new/' + epoch_num + '.pth'))
        net = net.to(device)

        tsSet = ngsimDataset('./data/dataset_t_v_t/TestSet.mat')
        tsDataloader = DataLoader(tsSet, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                  collate_fn=tsSet.collate_fn)
        lossVals = torch.zeros(25).to(device)
        counts = torch.zeros(25).to(device)
        avg_val_loss = 0
        val_batch_count = len(tsDataloader)

        for i, data in enumerate(tsDataloader):
            st_time = time.time()
            hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu, hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix = data
            if args['use_cuda']:
                hist_batch = hist_batch.to(device)
                nbrs_batch = nbrs_batch.to(device)
                mask_batch = mask_batch.to(device)
                lat_enc_batch = lat_enc_batch.to(device)
                lon_enc_batch = lon_enc_batch.to(device)
                lane_batch = lane_batch.to(device)
                nbrslane_batch = nbrslane_batch.to(device)
                class_batch = class_batch.to(device)
                nbrsclass_batch = nbrsclass_batch.to(device)
                fut_batch = fut_batch.to(device)
                op_mask_batch = op_mask_batch.to(device)
                va_batch = va_batch.to(device)
                nbrsva_batch = nbrsva_batch.to(device)
                edge_index_batch = edge_index_batch.to(device)
                ve_matrix_batch = ve_matrix_batch.to(device)
                ac_matrix_batch = ac_matrix_batch.to(device)
                man_matrix_batch = man_matrix_batch.to(device)
                view_grip_batch = view_grip_batch.to(device)
                graph_matrix = graph_matrix.to(device)

            if metric == 'nll':
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                                                       lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                                                       va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch,
                                                       ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix)

                    l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut_batch, op_mask_batch)
                else:
                    fut_pred = net(hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch,
                                   nbrslane_batch, class_batch, nbrsclass_batch,
                                   va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch,
                                   man_matrix_batch, view_grip_batch, graph_matrix)
                    l, c, loss = maskedNLLTest(fut_pred, 0, 0, fut_batch, op_mask_batch, use_maneuvers=False)
            else:
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                                                       lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                                                       va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch,
                                                       ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix)
                    fut_pred_max = torch.zeros_like(fut_pred[0])
                    for k in range(lat_pred.shape[0]):
                        lat_man = torch.argmax(lat_pred[k, :]).detach()
                        lon_man = torch.argmax(lon_pred[k, :]).detach()
                        indx = lon_man * 3 + lat_man
                        fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
                    l, c, loss = maskedMSETest(fut_pred_max, fut_batch, op_mask_batch)
                else:
                    fut_pred = net(hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch,
                                   nbrslane_batch, class_batch, nbrsclass_batch,
                                   va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch,
                                   man_matrix_batch, view_grip_batch, graph_matrix)
                    l, c, loss = maskedMSETest(fut_pred, fut_batch, op_mask_batch)

            lossVals += l.detach()
            counts += c.detach()
            avg_val_loss += loss.item()
            if i == int(val_batch_count / 4) * model_step:
                print('process:', model_step / 4)
                model_step += 1

        if metric == 'nll':
            print('valnll:', avg_val_loss / val_batch_count)
            print(lossVals / counts)
        else:
            print('valmse:', avg_val_loss / val_batch_count)
            print(torch.pow(lossVals / counts, 0.5) * 0.3048)


if __name__ == '__main__':
    names = 'epoch18-modelv860'
    evaluate = Evaluate()
    for epoch in names:
        evaluate.main(names)
