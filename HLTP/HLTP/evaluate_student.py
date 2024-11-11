from __future__ import print_function
import loader2 as lo
from torch.utils.data import DataLoader
import pandas as pd
from config import *
import matplotlib.pyplot as plt
import os
import time
import student_model.student_model as stu_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
writer = pd.ExcelWriter('A.xlsx')


class Evaluate():

    def __init__(self):
        self.op = 0
        self.scale = 0.3048
        self.prop = 1

    def maskedMSETest(self, y_pred, y_gt, mask):
        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    def logsumexp(self, inputs, dim=None, keepdim=False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2,
                      use_maneuvers=True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim=2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            out = 0.5 * t.pow(ohr, 2) * (t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,2) - 2 * rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss

    def main(self, name, val):
        model_step = 1
        args['train_flag'] = True
        l_path = args['path']
        generator = stu_model.Generator_stu(args=args)
        gdEncoder = stu_model.GDEncoder_stu(args=args)
        generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.pth', map_location='cuda:0'))
        gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.pth', map_location='cuda:0'))
        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()
        gdEncoder.eval()

        if val:
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.ngsimDataset('./data/dataset_t_v_t/TestSet.mat')
                else:
                    t2 = lo.ngsimDataset('./data/5feature/TestSet.mat')
            else:
                t2 = lo.HighdDataset('Val')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn)
        else:
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.ngsimDataset('./data/dataset_t_v_t/TestSet.mat')
                else:
                    t2 = lo.ngsimDataset('./data/5feature/TestSet.mat')
            else:
                t2 = lo.HighdDataset('Test')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn)

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0

        val_batch_count = len(valDataloader)
        print("begin.................................")
        with(t.no_grad()):
            for idx, data in enumerate(valDataloader):
                hist, nbrs, lane, nbrslane, cls, nbrscls, va, nbrsva, fut, hist_batch, nbrs_batch, mask, lat_enc, lon_enc, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix = data
                hist = hist.to(device)
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut = fut[:args['out_length'], :, :]
                fut = fut.to(device)
                op_mask = op_mask[:args['out_length'], :, :]
                op_mask = op_mask.to(device)
                va = va.to(device)
                nbrsva = nbrsva.to(device)
                lane = lane.to(device)
                nbrslane = nbrslane.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)

                te = time.time()
                values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
                fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
                all_time += time.time() - te

                if not args['train_flag']:
                    indices = []
                    if args['val_use_mse']:
                        fut_pred_max = t.zeros_like(fut_pred[0])
                        for k in range(lat_pred.shape[0]):
                            lat_man = t.argmax(lat_enc[k, :]).detach()
                            lon_man = t.argmax(lon_enc[k, :]).detach()
                            index = lon_man * 3 + lat_man
                            indices.append(index)
                            fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                        l, c, loss = self.maskedMSETest(fut_pred_max, fut, op_mask)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                else:
                    if args['val_use_mse']:
                        l, c, loss = self.maskedMSETest(fut_pred, fut, op_mask)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                lossVals += l.detach()
                counts += c.detach()
                avg_val_loss += loss.item()
                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1
            if args['val_use_mse']:
                print('valmse:', avg_val_loss / val_batch_count)
                print(t.pow(lossVals / counts, 0.5) * 0.3048)
            else:
                print('valnll:', avg_val_loss / val_batch_count)
                print(lossVals / counts)


if __name__ == '__main__':

    names = ['1']
    evaluate = Evaluate()
    for epoch in names:
        evaluate.main(name=epoch, val=False)
