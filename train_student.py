from torch.utils.data import DataLoader
import loader2 as lo
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
from evaluate_teacher import Evaluate_teacher
from config import *
from student_model import student_model as stu_model
from teacher_model import teacher_model as tea_model
from utils import sigmoid, maskedNLL, maskedMSE, MSELoss2, CELoss, distillation_loss
import warnings
import math

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    args['train_flag'] = True
    l_path = args['path']
    l_pre_path = args['pre_path']
    name = args['name']
    print(args['path'])
    highwayNet_name = '/epoch' + name + '_g.pth'
    highwayNet_path = os.path.join(l_pre_path, highwayNet_name)

    log_var_fut = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
    log_var_lat = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
    log_var_lon = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)

    log_var = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
    log_var_distill = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)

    evaluate_teacher = Evaluate_teacher()

    gdEncoder_stu = stu_model.GDEncoder_stu(args)
    generator_stu = stu_model.Generator_stu(args)

    gdEncoder_stu = gdEncoder_stu.to(device)
    generator_stu = generator_stu.to(device)
    gdEncoder_stu.train()
    generator_stu.train()

    trSet = lo.ngsimDataset('./data/dataset_t_v_t/TrainSet.mat')
    trDataloader = DataLoader(trSet, batch_size=args['batch_size'], shuffle=True, num_workers=8,
                              drop_last=True, persistent_workers=True, prefetch_factor=4,
                              collate_fn=trSet.collate_fn, pin_memory=True)

    params_gdEncoder = list(gdEncoder_stu.parameters()) + [log_var, log_var_distill, log_var_fut, log_var_lat, log_var_lon]
    params_generator = list(generator_stu.parameters()) + [log_var, log_var_distill, log_var_fut, log_var_lat, log_var_lon]

    optimizer_gd = optim.Adam(params_gdEncoder, lr=learning_rate)
    optimizer_g = optim.Adam(params_generator, lr=learning_rate)

    scheduler_gd = ExponentialLR(optimizer_gd, gamma=0.6)
    scheduler_g = ExponentialLR(optimizer_g, gamma=0.6)

    for epoch in range(args['epoch']):
        print("epoch:", epoch + 1, 'lr', optimizer_g.param_groups[0]['lr'])  # 打印训练轮次信息
        loss_gi1 = 0
        loss_gix = 0
        loss_gx_2i = 0
        loss_gx_3i = 0
        distil_loss = 0
        distil_loss2 = 0
        distil_loss3 = 0
        for idx, data in enumerate(tqdm(trDataloader)):
            hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu, \
                hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, \
                fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix = data

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
            hist_batch_stu = hist_batch_stu.to(device)
            nbrs_batch_stu = nbrs_batch_stu.to(device)
            lane_batch_stu = lane_batch_stu.to(device)
            nbrslane_batch_stu = nbrslane_batch_stu.to(device)
            class_batch_stu = class_batch_stu.to(device)
            nbrsclass_batch_stu = nbrsclass_batch_stu.to(device)
            fut_batch_stu = fut_batch_stu.to(device)
            va_batch_stu = va_batch_stu.to(device)
            nbrsva_batch_stu = nbrsva_batch_stu.to(device)
            edge_index_batch = edge_index_batch.to(device)
            ve_matrix_batch = ve_matrix_batch.to(device)
            ac_matrix_batch = ac_matrix_batch.to(device)
            man_matrix_batch = man_matrix_batch.to(device)
            view_grip_batch = view_grip_batch.to(device)
            graph_matrix = graph_matrix.to(device)

            fut_pred_tea, lat_pred_tea, lon_pred_tea = evaluate_teacher.main(args['name'], hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix)

            optimizer_g.zero_grad()
            optimizer_gd.zero_grad()

            values = gdEncoder_stu(hist_batch_stu, nbrs_batch_stu, mask_batch, va_batch_stu, nbrsva_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu)
            g_out, lat_pred, lon_pred = generator_stu(values, lat_enc_batch, lon_enc_batch)

            precision = torch.exp(-log_var)
            precision_distill = torch.exp(-log_var_distill)
            precision_fut = torch.exp(-log_var_fut)
            precision_lon = torch.exp(-log_var_lon)
            precision_lat = torch.exp(-log_var_lat)
            precision = precision.to(device)
            precision_fut = precision_fut.to(device)
            precision_lat = precision_lat.to(device)
            precision_lon = precision_lon.to(device)
            precision_distill = precision_distill.to(device)

            if args['use_mse']:
                loss_g1 = precision*MSELoss2(g_out, fut_batch_stu, op_mask_batch) + precision_distill*distillation_loss(g_out, fut_pred_tea)
            else:
                if epoch < args['pre_epoch']:
                    loss_g1 = precision*MSELoss2(g_out, fut_batch_stu, op_mask_batch) + precision_distill*distillation_loss(g_out, fut_pred_tea)
                else:
                    loss_g1 = precision*maskedNLL(g_out, fut_batch_stu, op_mask_batch) + precision_distill*distillation_loss(g_out, fut_pred_tea)

            loss_gx_3 = precision*CELoss(lat_pred, lat_enc_batch) + precision_distill*distillation_loss(lat_pred, lat_pred_tea)
            loss_gx_2 = precision*CELoss(lon_pred, lon_enc_batch) + precision_distill*distillation_loss(lon_pred, lon_pred_tea)
            loss_gx = precision_lat*loss_gx_3 + precision_lon*loss_gx_2
            loss_g = precision_fut*loss_g1 + loss_gx + log_var + log_var_distill + log_var_fut + log_var_lat + log_var_lon

            loss_g.backward()

            a = t.nn.utils.clip_grad_norm_(generator_stu.parameters(), 10)
            a = t.nn.utils.clip_grad_norm_(gdEncoder_stu.parameters(), 10)

            optimizer_g.step()
            optimizer_gd.step()

            loss_gi1 += loss_g1.item()
            loss_gx_2i += loss_gx_2.item()
            loss_gx_3i += loss_gx_3.item()
            loss_gix += loss_gx.item()
            distil_loss += distillation_loss(g_out, fut_pred_tea).item()
            distil_loss2 += distillation_loss(lat_pred, lat_pred_tea).item()
            distil_loss3 += distillation_loss(lon_pred, lon_pred_tea).item()

            if idx % 4000 == 3999:
                print('mse:', loss_gi1 / 4000, '|loss_gx_2:', loss_gx_2i / 4000, '|loss_gx_3', loss_gx_3i / 4000)
                print('distil_loss:', distil_loss / 4000, 'distil_loss2:', distil_loss2 / 4000, 'distil_loss3', distil_loss3 / 4000)

                loss_gi1 = 0
                loss_gix = 0
                loss_gx_2i = 0
                loss_gx_3i = 0
                distil_loss = 0
                distil_loss2 = 0
                distil_loss3 = 0

        save_model(name=str(epoch + 1), gdEncoder=gdEncoder_stu,
                   generator=generator_stu, path=args['path'])

        scheduler_gd.step()
        scheduler_g.step()


def save_model(name, gdEncoder, generator, path):
    l_path = args['path']
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    t.save(gdEncoder.state_dict(), l_path + '/epoch' + name + '_gd.pth')
    t.save(generator.state_dict(), l_path + '/epoch' + name + '_g.pth')


if __name__ == '__main__':
    main()



