from __future__ import print_function, division
import torch
import torch as t
import torch.nn.functional as F
import math
from config import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
     
    return out


def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
     
    out = 0.5 * torch.pow(ohr, 2) * (
                torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                            2) - 2 * rho * torch.pow(
            sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379

    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2, use_maneuvers=True,
                  avg_along_time=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
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
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                 
                out = -(0.5 * torch.pow(ohr, 2) * (
                            torch.pow(sigX, 2) * torch.pow(x - muX, 2) + 0.5 * torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                              2) - rho * torch.pow(
                        sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379)
                 
                 
                acc[:, :, count] = out + torch.log(wts)
                count += 1
        acc = -logsumexp(acc, dim=2)
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc, dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
         
        out = 0.5 * torch.pow(ohr, 2) * (
                    torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY,
                                                                                                2) - 2 * rho * torch.pow(
                sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379
         
         
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:, :, 0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts


def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
     
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
     
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)   
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal


def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:, :, 0], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    loss = torch.sum(acc) / torch.sum(mask)
    return lossVal, counts, loss


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return y


# MSE损失函数
def MSELoss2(g_out, fut, mask):
    acc = t.zeros_like(mask)
    muX = g_out[:, :, 0]
    muY = g_out[:, :, 1]
    x = fut[:, :, 0]
    y = fut[:, :, 1]
    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal


def CELoss(pred, target):
    value = t.log(t.sum(pred * target, dim=-1))
    return -t.sum(value) / value.shape[0]


def distillation_loss(y, teacher_scores, temp=args['temp'], alpha=args['alpha']):
    distillation_loss = t.nn.KLDivLoss()(F.log_softmax(y / temp, dim=1),
                   F.softmax(teacher_scores / temp, dim=1)) * temp * temp * 2.0 * alpha
    return distillation_loss