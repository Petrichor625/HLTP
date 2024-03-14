from __future__ import print_function
import torch
import loader2 as lo
from torch.utils.data import DataLoader
import pandas as pd
from config import *
import matplotlib.pyplot as plt
import os
import time
from teacher_model import teacher_model
from student_model import student_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
writer = pd.ExcelWriter('A.xlsx')


class Evaluate():
    def __init__(self):
        self.op = 0
        self.drawImg = True
        
        self.scale = 1
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
        s, _ = t.max(inputs, dim=dim,
                     keepdim=True)  
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()  
        if not keepdim:  
            outputs = outputs.squeeze(dim)
        return outputs
    
    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2,
                      use_maneuvers=True, avg_along_time=False):
        if use_maneuvers:  
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(
                device)  
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
            if avg_along_time:
                lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
                return lossVal
            else:
                lossVal = torch.sum(acc, dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts
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
            if avg_along_time:
                lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
                return lossVal
            else:
                lossVal = torch.sum(acc[:, :, 0], dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts

    def main(self, name, val):
        model_step = 1
        args['train_flag'] = True
        l_path = args['path']
        name2 = args['name']
        name3 = '1'
        l_path2 = args['pre_path']
        l_path3 = './stdan-master/checkponint/true_1'
        generator = stu_model.Generator_stu(args=args)
        gdEncoder = stu_model.GDEncoder_stu(args=args)
        state_dict = torch.load(l_path + '/epoch' + name + '_gd.pth', map_location=device)
        gdEncoder.iafsqueeze.v_mem = state_dict.get('iafsqueeze.v_mem', None)
        generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.pth', map_location='cuda:0'))
        gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.pth', map_location='cuda:0'))
        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()
        gdEncoder.eval()
        net = teacher_modelv860.highwayNet(args=args)
        net.load_state_dict(t.load(l_path2 + '/epoch' + name2 + '.pth', map_location='cuda:0'))
        net = net.to(device)
        net.eval()
        stdan_gen = student_model_original.Generator_stu(args=args)
        stdan_gde = student_model_original.GDEncoder_stu(args=args)
        stdan_gen.load_state_dict(t.load(l_path3 + '/epoch' + name3 + '_g.tar', map_location='cuda:0'))
        stdan_gde.load_state_dict(t.load(l_path3 + '/epoch' + name3 + '_gd.tar', map_location='cuda:0'))
        stdan_gen = stdan_gen.to(device)
        stdan_gde = stdan_gde.to(device)
        stdan_gde.eval()
        stdan_gen.eval()
        valSet = lo.ngsimDataset('./data/dataset_t_v_t/TestSet.mat')
        valDataloader = DataLoader(valSet, batch_size=256, shuffle=True, num_workers=8, drop_last=True,
                                   collate_fn=valSet.collate_fn, persistent_workers=True, prefetch_factor=4,
                                   pin_memory=True)
        lossVals = torch.zeros(25).to(device)
        counts = torch.zeros(25).to(device)
        avg_val_loss = 0
        val_batch_count = len(valDataloader)
        if val:  
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.ngsimDataset('./data/dataset_t_v_t/TestSet.mat')
                else:
                    t2 = lo.NgsimDataset('./data/5feature/TestSet.mat')
            else:
                t2 = lo.HighdDataset('./data/dataset_t_v_t/highd/TestSet_highd40.mat')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn, drop_last=True, persistent_workers=True,
                                       prefetch_factor=4, pin_memory=True)  
        else:  
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.ngsimDataset('./data/dataset_t_v_t/TestSet.mat')
                else:
                    t2 = lo.NgsimDataset('./data/5feature/TestSet.mat')
            else:
                t2 = lo.HighdDataset('./data/dataset_t_v_t/highd/TestSet_highd40.mat')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn, drop_last=True, persistent_workers=True,
                                       prefetch_factor=4, pin_memory=True)
        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        val_batch_count = len(valDataloader)
        with(t.no_grad()):  
            for idx, data in enumerate(valDataloader):
                hist_stu, nbrs_stu, lane_stu, nbrslane_stu, cls_stu, nbrscls_stu, va_stu, nbrsva_stu, fut_stu, \
                    hist, nbrs, mask, lat_enc, lon_enc, lane, nbrslane, cls, nbrscls, va, nbrsva, fut, op_mask, edge_index, ve_matrix, ac_matrix, man_matrix, view_grip, graph_matrix = data
                if args['ran_miss_continue']:
                    hist_stu = lo.ran_miss_continue(hist_stu)
                    nbrs_stu = lo.ran_miss_continue(nbrs_stu)
                    lane_stu = lo.ran_miss_continue(lane_stu)
                    nbrslane_stu = lo.ran_miss_continue(nbrslane_stu)
                    cls_stu = lo.ran_miss_continue(cls_stu)
                    nbrscls_stu = lo.ran_miss_continue(nbrscls_stu)
                    va_stu = lo.ran_miss_continue(va_stu)
                    nbrsva_stu = lo.ran_miss_continue(nbrsva_stu)
                hist = hist.to(device)
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut = fut[:args['out_length'], :, :]
                fut = fut.to(device)
                op_mask = op_mask[:args['out_length'], :, :]
                op_mask = op_mask.to(device)
                lane = lane.to(device)
                nbrslane = nbrslane.to(device)
                hist_stu = hist_stu.to(device)
                nbrs_stu = nbrs_stu.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)
                va = va.to(device)
                nbrsva = nbrsva.to(device)
                fut_stu = fut_stu[:args['out_length'], :, :]
                fut_stu = fut_stu.to(device)
                va_stu = va_stu.to(device)
                nbrsva_stu = nbrsva_stu.to(device)
                lane_stu = lane_stu.to(device)
                nbrslane_stu = nbrslane_stu.to(device)
                cls_stu = cls_stu.to(device)
                nbrscls_stu = nbrscls_stu.to(device)
                edge_index = edge_index.to(device)
                ve_matrix = ve_matrix.to(device)
                ac_matrix = ac_matrix.to(device)
                man_matrix = man_matrix.to(device)
                view_grip = view_grip.to(device)
                graph_matrix = graph_matrix.to(device)
                values, s = gdEncoder(hist_stu, nbrs_stu, mask, va_stu, nbrsva_stu, lane_stu, nbrslane_stu, cls_stu,
                                      nbrscls_stu)  
                fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)  

                fut_pred_tea, lat_pred_tea, lon_pred_tea = net(hist, nbrs, mask, lat_enc, lon_enc, lane, nbrslane, cls, nbrscls, va, nbrsva, edge_index, ve_matrix, ac_matrix, man_matrix, view_grip, graph_matrix)
                values_stdan = stdan_gde(hist_stu, nbrs_stu, mask, va_stu, nbrsva_stu, lane_stu, nbrslane_stu, cls_stu,
                                         nbrscls_stu)  
                fut_pred_stdan, lat_pred_stdan, lon_pred_stdan = stdan_gen(values_stdan, lat_enc, lon_enc)
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
                        l, c, loss = self.maskedMSETest(fut_pred_max, fut_stu, op_mask)
                    else:  
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut_stu, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                    if self.drawImg:  
                        lat_man = t.argmax(lat_enc, dim=-1).detach()
                        lon_man = t.argmax(lon_enc, dim=-1).detach()
                        self.draw(hist_stu, fut_stu, nbrs_stu, mask, fut_pred, fut_pred_tea, fut_pred_stdan,
                                  args['train_flag'], lon_man, lat_man, op_mask,
                                  indices)
            else:
                if args['val_use_mse']:
                    l, c, loss = self.maskedMSETest(fut_pred, fut_stu, op_mask)
                else:
                    l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut_stu, op_mask,
                                                    use_maneuvers=args['use_maneuvers'])
                if self.drawImg:
                    lat_man = t.argmax(lat_enc, dim=-1).detach()
                    lon_man = t.argmax(lon_enc, dim=-1).detach()
                    self.draw(hist_stu, fut_stu, nbrs_stu, mask, fut_pred, fut_pred_tea, fut_pred_stdan,
                              args['train_flag'], lon_man, lat_man, op_mask, None)
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

    def add_car(self, plt, x, y, alp, color0):
        plt.gca().add_patch(plt.Rectangle(
            (x, y - 2.0),  
            8.0,  
            4.0,  
            color=color0,  
            alpha=alp,  
            zorder=2
        ))

    def draw(self, hist, fut, nbrs, mask, fut_pred1, fut_pred2, fut_pred3, train_flag, lon_man, lat_man, op_mask,
             indices):
        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        op_mask = op_mask.cpu()
        IPL = 0
        for i in range(hist.size(1)):
            lon_man_i = lon_man[i].item()
            lat_man_i = lat_man[i].item()
            if lat_man_i != 0:
                marker_interval = 2
                plt.axis('on')  
                plt.ylim(-36 * self.scale, 36 * self.scale)  
                plt.xlim(-180 * self.scale * self.prop, 180 * self.scale * self.prop)  
                plt.figure(dpi=1080)
                plt.tick_params(
                    axis='both',  
                    which='both',  
                    bottom=False,  
                    top=False,  
                    left=False,  
                    right=False,  
                    labelbottom=False,  
                    labeltop=False,  
                    labelleft=False,  
                    labelright=False  
                )
                IPL_i = mask[i, :, :, :].sum().sum()  
                IPL_i = int((IPL_i / 64).item())
                for ii in range(IPL_i):
                    self.add_car(plt, nbrs[-1, IPL + ii, 1] * self.scale * self.prop,
                                 nbrs[-1, IPL + ii, 0] * self.scale,
                                 alp=1, color0=(125 / 255, 171 / 255, 207 / 255))
                    plt.plot(nbrs[:, IPL + ii, 1] * self.scale * self.prop, nbrs[:, IPL + ii, 0] * self.scale, '-',
                             color='black', linewidth=0.5, zorder=2)
                IPL = IPL + IPL_i
                self.add_car(plt, hist[-1, i, 1], hist[-1, i, 0], alp=1, color0=(240 / 255, 179 / 255, 98 / 255))
                plt.plot(hist[:, i, 1] * self.scale * self.prop, hist[:, i, 0] * self.scale, '-', color='black',
                         linewidth=0.5, zorder=2)
                plt.plot(fut[:, i, 1] * self.scale * self.prop, fut[:, i, 0] * self.scale, '-',
                         color=(2 / 255, 48 / 255, 74 / 255),
                         linewidth=0.75, zorder=2)
                if train_flag:
                    fut_pred1 = fut_pred1.detach().cpu()  
                    fut_pred2 = fut_pred2.detach().cpu()
                    fut_pred3 = fut_pred3.detach().cpu()
                    fut = fut.detach().cpu()
                    plt.scatter(fut[::marker_interval, i, 1] * self.scale * self.prop,
                                fut[::marker_interval, i, 0] * self.scale, marker='D',
                                color=(2 / 255, 48 / 255, 74 / 255), s=3, zorder=4)
                    plt.plot(fut_pred1[:, i, 1] * self.scale * self.prop, fut_pred1[:, i, 0] * self.scale, '-',
                             color=(33 / 255, 158 / 255, 188 / 255),
                             linewidth=0.75, zorder=4)
                    plt.plot(fut_pred2[:, i, 1] * self.scale * self.prop, fut_pred2[:, i, 0] * self.scale, '-',
                             color=(250 / 255, 134 / 255, 0 / 255),
                             linewidth=0.75, zorder=4)
                    plt.plot(fut_pred3[:, i, 1] * self.scale * self.prop, fut_pred3[:, i, 0] * self.scale, '-',
                             color='grey',
                             linewidth=0.75, zorder=4)
                    plt.scatter(fut_pred1[::marker_interval, i, 1] * self.scale * self.prop,
                                fut_pred1[::marker_interval, i, 0] * self.scale, marker='^',
                                color=(33 / 255, 158 / 255, 188 / 255), s=5, zorder=3)
                    plt.scatter(fut_pred2[::marker_interval, i, 1] * self.scale * self.prop * 2,
                                fut_pred2[::marker_interval, i, 0] * self.scale, marker='+',
                                color=(250 / 255, 134 / 255, 0 / 255), s=7, zorder=4)
                    plt.scatter(fut_pred3[::marker_interval, i, 1] * self.scale * self.prop,
                                fut_pred3[::marker_interval, i, 0] * self.scale, marker='x',
                                color='grey', s=5, zorder=4)
                    muX = fut_pred1[:, i, 0]  
                    muY = fut_pred1[:, i, 1]  
                    x = fut[:, i, 0]  
                    y = fut[:, i, 1]
                    max_y = y[-1] - y[0]
                    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                    acc = out * op_mask[:, i, 0]
                    loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
                else:
                    for j in range(len(fut_pred1)):
                        fut_pred_i = fut_pred1[j].detach().cpu()
                        if j == indices[i].item():
                            plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                     color='red', linewidth=1)
                            muX = fut_pred_i[:, i, 0]
                            muY = fut_pred_i[:, i, 1]
                            x = fut[:, i, 0]
                            y = fut[:, i, 1]
                            max_y = y[-1] - y[0]
                            out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                            acc = out * op_mask[:, i, 0]
                            loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
                        else:
                            plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                     color='green', linewidth=1)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig('./pic/' + str(lon_man_i + 1) + '_' + str(lat_man_i + 1) + '/' + str(self.op) + '.png')
                self.op += 1
                
                plt.close()


if __name__ == '__main__':
    names = ['7']
    evaluate = Evaluate()
    for epoch in names:
        print("epoch:", epoch)
        evaluate.main(name=epoch, val=False)
