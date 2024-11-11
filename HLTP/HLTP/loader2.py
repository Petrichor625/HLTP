from __future__ import print_function, division
from scipy import spatial
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
from config import args
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class ngsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=25, d_s=1, t_h_stu=30, t_f_stu=50, d_s_stu=2, enc_size=64,
                 grid_size=(13, 3)):

        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']

        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s

        self.t_h_stu = t_h_stu
        self.t_f_stu = t_f_stu
        self.d_s_stu = d_s_stu

        self.enc_size = enc_size
        self.grid_size = grid_size

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx, 11:]
        neighbors = []
        neighborsclass = []
        neighborslane = []
        neighborsva = []

        neighbors_stu = []
        neighborsclass_stu = []
        neighborslane_stu = []
        neighborsva_stu = []

        hist_stu = self.getHistory_stu(vehId, t, vehId, dsId)
        fut_stu = self.getFuture_stu(vehId, t, dsId)
        lane_stu = self.getLane_stu(vehId, t, vehId, dsId)
        cclass_stu = self.getClass_stu(vehId, t, vehId, dsId)
        va_stu = self.getVA_stu(vehId, t, vehId, dsId)
        for i in grid:
            neighbors_stu.append(self.getHistory_stu(i.astype(int), t, vehId, dsId))
            neighborslane_stu.append(self.getLane_stu(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass_stu.append(self.getClass_stu(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsva_stu.append(self.getVA_stu(i.astype(int), t, vehId, dsId))

        hist = self.getHistory(vehId, t, vehId, dsId)
        fut = self.getFuture(vehId, t, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)
        va = self.getVA(vehId, t, vehId, dsId)

        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
        xy_list = self.get_xy(dsId, t, grid, vehId)
        edge_index = self.graph_xy(xy_list)
        va_list = self.get_va(dsId, t, grid, vehId)
        ve_matrix = self.graph_ve(va_list)
        ac_matrix = self.graph_ac(va_list)
        man_list = self.get_man(dsId, t, grid, vehId)
        man_matrix = self.graph_man(man_list)
        view_grip = self.mask_view(dsId, t, grid, vehId)

        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 9] - 1)] = 1

        return hist_stu, fut_stu, neighbors_stu, lane_stu, neighborslane_stu, cclass_stu, neighborsclass_stu, va_stu, neighborsva_stu, hist, fut, neighbors, lat_enc, lon_enc, lane, neighborslane, cclass, neighborsclass, va, neighborsva, edge_index, ve_matrix, ac_matrix, man_matrix, view_grip

    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])

            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getHistory_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 1:3] - refPos

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 2])
            return hist_stu

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getVA_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 3:5]

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 2])
            return hist_stu

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getLane_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 5]

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 1])
            return hist_stu

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 6]

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 1])
            return hist_stu

    def get_xy(self, dsId, t, grid, vehId):
        xy_list = np.full((len(grid), 2), np.inf)
        grid[19] = vehId
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refPos = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refPos = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()

                    if vehTrack.size != 0:
                        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
            if refPos.size != 0:
                xy_list[i] = refPos.flatten()
        return xy_list

    def graph_xy(self, xy_list):

        node1 = []
        node2 = []

        xy_list = xy_list.astype(float)
        max_num_object = 39

        neighbor_matrix = np.zeros((max_num_object, max_num_object))

        dist_xy = spatial.distance.cdist(xy_list, xy_list)

        dist_xy[np.isinf(dist_xy)] = np.inf
        dist_xy[np.isnan(dist_xy)] = np.inf

        for i in range(39):
            for j in range(i + 1, 39):
                if dist_xy[i][j] <= 100:
                    node1.append(i)
                    node2.append(j)

        node1 = torch.tensor(node1).unsqueeze(0)
        node2 = torch.tensor(node2).unsqueeze(0)
        edge_index = torch.cat((node1, node2), dim=0)

        return edge_index

    def get_man(self, dsId, t, grid, vehId):
        man_list = np.full((len(grid), 2), 0)
        grid[19] = vehId
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refMAN = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refMAN = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        refMAN = vehTrack[np.where(vehTrack[:, 0] == t)][0, 7:9]
            if refMAN.size != 0:
                man_list[i] = refMAN.flatten()
        return man_list

    def graph_man(self, man_list):

        man_list = man_list.astype(float)
        man_list[np.where(man_list == 0)] = np.nan
        man_list[np.isinf(man_list)] = np.nan
        max_num_object = 39
        man_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(39):
            for j in range(i + 1, 39):
                if (man_list[i][0] == man_list[j][0]) and (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 2

                elif (man_list[i][0] == man_list[j][0]) or (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 1

        return man_matrix

    def get_va(self, dsId, t, grid, vehId):
        va_list = np.full((len(grid), 2), 0)
        grid[19] = vehId
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refVA = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refVA = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        refVA = vehTrack[np.where(vehTrack[:, 0] == t)][0, 3:5]

            if refVA.size != 0:
                va_list[i] = refVA.flatten()

        return va_list

    def graph_ve(self, va_list):

        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 39
        ve_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(39):
            for j in range(i + 1, 39):
                ve_matrix[i][j] = (va_list[i][0] - va_list[j][0])
                ve_matrix[j][i] = -ve_matrix[i][j]

        ve_matrix = torch.tensor(ve_matrix).float()
        return ve_matrix

    def graph_ac(self, va_list):

        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 39
        ac_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(39):
            for j in range(i + 1, 39):
                ac_matrix[i][j] = (va_list[i][1] - va_list[j][1])
                ac_matrix[j][i] = -ac_matrix[i][j]

        ac_matrix = torch.tensor(ac_matrix).float()
        return ac_matrix

    def mask_view(self, dsId, t, grid, vehId):

        view_matrix = np.zeros((39))
        if vehId == 0:
            view_matrix1 = np.array(view_matrix)
            return view_matrix1.reshape(3, 13)
        else:
            if self.T.shape[1] <= vehId - 1:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(3, 13)
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            velocity = vehTrack[np.where(vehTrack[:, 0] == t)][0, 3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0 or grid.size == 0:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(3, 13)
            else:
                if velocity < 30:
                    indices = torch.tensor([4, 5, 6, 7, 8, 17, 18, 20, 21, 30, 31, 32, 33, 34])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)
                elif velocity >= 30 and velocity <= 60:
                    indices = torch.tensor([2, 3, 4, 8, 9, 10, 15, 16, 17, 18, 20, 21, 22, 23, 28, 29, 30, 34, 35, 36])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)
                elif velocity > 60:
                    indices = torch.tensor([0, 1, 2, 10, 11, 12, 13, 14, 15, 23, 24, 25, 26, 27, 28, 36, 37, 38])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)

    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    def getFuture_stu(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s_stu
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f_stu + 1)
        fut_stu = vehTrack[stpt:enpt:self.d_s_stu, 1:3] - refPos
        return fut_stu

    def collate_fn(self, samples):

        nbr_batch_size = 0
        nbr_batch_size_stu = 0
        for _, _, _, _, _, _, _, _, _, _, _, nbrs, _, _, _, _, _, _, _, _, _, _, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
        for _, _, nbrs_stu, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ in samples:
            nbr_batch_size_stu += sum([len(nbrs_stu[i]) != 0 for i in range(len(nbrs_stu))])
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        maxlen_stu = self.t_h_stu // self.d_s_stu + 1
        nbrs_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 2)
        nbrsva_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 2)
        nbrslane_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 1)
        nbrsclass_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 1)

        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)

        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        mask_batch_stu = mask_batch.bool()

        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)

        hist_batch_stu = torch.zeros(maxlen_stu, len(samples), 2)
        fut_batch_stu = torch.zeros(self.t_f_stu // self.d_s_stu, len(samples), 2)
        op_mask_batch_stu = torch.zeros(self.t_f_stu // self.d_s_stu, len(samples), 2)

        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 3)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)

        va_batch_stu = torch.zeros(maxlen_stu, len(samples), 2)
        lane_batch_stu = torch.zeros(maxlen_stu, len(samples), 1)
        class_batch_stu = torch.zeros(maxlen_stu, len(samples), 1)

        edge_index_number = 0
        for sampleId, (_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, edge_index, _, _, _, _) in enumerate(
                samples):
            edge_index_number += edge_index.shape[1]
        edge_index_batch = torch.zeros(len(samples), 2, edge_index_number)
        ve_matrix_batch = torch.zeros(len(samples), 39, 39)
        ac_matrix_batch = torch.zeros(len(samples), 39, 39)
        ve_matrix_batch = torch.zeros(len(samples), 39, 39)
        man_matrix_batch = torch.zeros(len(samples), 39, 39)
        view_grip_batch = torch.zeros(len(samples), 3, 13)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        for sampleId, (hist_stu, fut_stu, nbrs_stu, lane_stu, neighborslane_stu, cclass_stu, neighborsclass_stu, va_stu,
                       neighborsva_stu,
                       hist, fut, nbrs, lat_enc, lon_enc, lane, neighborslane, cclass, neighborsclass, va, neighborsva,
                       edge_index, ve_matrix, ac_matrix, man_matrix, view_grip) in enumerate(samples):

            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)

            hist_batch_stu[0:len(hist_stu), sampleId, 0] = torch.from_numpy(hist_stu[:, 0])
            hist_batch_stu[0:len(hist_stu), sampleId, 1] = torch.from_numpy(hist_stu[:, 1])
            fut_batch_stu[0:len(fut_stu), sampleId, 0] = torch.from_numpy(fut_stu[:, 0])
            fut_batch_stu[0:len(fut_stu), sampleId, 1] = torch.from_numpy(fut_stu[:, 1])

            va_batch_stu[0:len(va_stu), sampleId, 0] = torch.from_numpy(va_stu[:, 0])
            va_batch_stu[0:len(va_stu), sampleId, 1] = torch.from_numpy(va_stu[:, 1])
            lane_batch_stu[0:len(lane_stu), sampleId, 0] = torch.from_numpy(lane_stu)
            class_batch_stu[0:len(cclass_stu), sampleId, 0] = torch.from_numpy(cclass_stu)

            ve_matrix = torch.tensor(ve_matrix)
            ve_matrix_batch[sampleId, :] = ve_matrix
            ac_matrix = torch.tensor(ac_matrix)
            ac_matrix_batch[sampleId, :] = ac_matrix
            man_matrix = torch.tensor(man_matrix)
            man_matrix_batch[sampleId, :] = man_matrix
            view_grip = torch.tensor(view_grip)
            view_grip_batch[sampleId, :] = view_grip

            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[
                        0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(
                        self.enc_size).byte()
                    count += 1

            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count3, :] = torch.from_numpy(nbrclass)
                    count3 += 1

            for id_stu, nbr_stu in enumerate(nbrs_stu):
                if len(nbr_stu) != 0:
                    nbrs_batch_stu[0:len(nbr_stu), count5, 0] = torch.from_numpy(nbr_stu[:, 0])
                    nbrs_batch_stu[0:len(nbr_stu), count5, 1] = torch.from_numpy(nbr_stu[:, 1])
                    pos[0] = id_stu % self.grid_size[
                        0]
                    pos[1] = id_stu // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(
                        self.enc_size).byte()
                    count5 += 1

            for id_stu, nbrva_stu in enumerate(neighborsva_stu):
                if len(nbrva_stu) != 0:
                    nbrsva_batch_stu[0:len(nbrva_stu), count6, 0] = torch.from_numpy(nbrva_stu[:, 0])
                    nbrsva_batch_stu[0:len(nbrva_stu), count6, 1] = torch.from_numpy(nbrva_stu[:, 1])
                    count6 += 1

            for id_stu, nbrlane_stu in enumerate(neighborslane_stu):
                if len(nbrlane_stu) != 0:
                    nbrslane_batch_stu[0:len(nbrlane_stu), count7, :] = torch.from_numpy(nbrlane_stu)
                    count7 += 1

            for id_stu, nbrclass_stu in enumerate(neighborsclass_stu):
                if len(nbrclass_stu) != 0:
                    nbrsclass_batch_stu[0:len(nbrclass_stu), count8, :] = torch.from_numpy(nbrclass_stu)
                    count8 += 1

            buffer = edge_index.shape[1]
            edge_index_batch[sampleId, :,
            count4:count4 + buffer] = edge_index
            count4 += buffer

        max_num_object = 39

        graph_matrix = torch.zeros((256, max_num_object, 2))

        return hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu, \
            hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix


class HighdDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):

        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']

        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s
        self.enc_size = enc_size
        self.grid_size = grid_size

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx, 14:]
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []

        hist = self.getHistory(vehId, t, vehId, dsId)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
        xy_list = self.get_xy(dsId, t, grid, vehId)
        edge_index = self.graph_xy(xy_list)
        va_list = self.get_va(dsId, t, grid, vehId)
        ve_matrix = self.graph_ve(va_list)
        ac_matrix = self.graph_ac(va_list)
        man_list = self.get_man(dsId, t, grid, vehId)
        man_matrix = self.graph_man(man_list)
        view_grip = self.mask_view(dsId, t, grid, vehId)

        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 13] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 12] - 1)] = 1

        return hist, fut, neighbors, lat_enc, lon_enc, lane, neighborslane, cclass, neighborsclass, va, neighborsva, edge_index, ve_matrix, ac_matrix, man_matrix, view_grip

    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6:8]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6:8] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 8]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 8]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def get_xy(self, dsId, t, grid, vehId):
        xy_list = np.full((len(grid), 2), np.inf)
        grid[19] = vehId
        refPos = np.zeros([0, 2])
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refPos = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refPos = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        if np.argwhere(vehTrack[:, 0] == t).size == 0:
                            return np.empty([0, 2])
                        else:
                            refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
            if refPos.size != 0:
                xy_list[i] = refPos.flatten()
        return xy_list

    def graph_xy(self, xy_list):
        node1 = []
        node2 = []
        xy_list = xy_list.astype(float)
        max_num_object = 39
        neighbor_matrix = np.zeros((max_num_object, max_num_object))
        dist_xy = spatial.distance.cdist(xy_list, xy_list)
        dist_xy[np.isinf(dist_xy)] = np.inf
        dist_xy[np.isnan(dist_xy)] = np.inf

        for i in range(39):
            for j in range(i + 1, 39):
                if dist_xy.size != 0:
                    if dist_xy[i][j] <= 100:
                        node1.append(i)
                        node2.append(j)

        node1 = torch.tensor(node1).unsqueeze(0)
        node2 = torch.tensor(node2).unsqueeze(0)
        edge_index = torch.cat((node1, node2), dim=0)
        return edge_index

    def get_man(self, dsId, t, grid, vehId):
        man_list = np.full((len(grid), 2), 0)
        grid[19] = vehId
        refMAN = np.zeros([0, 2])
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refMAN = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refMAN = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        if np.argwhere(vehTrack[:, 0] == t).size == 0:
                            return np.empty([0, 2])
                        else:
                            refMAN = vehTrack[np.where(vehTrack[:, 0] == t)][0, 7:9]
            if refMAN.size != 0:
                man_list[i] = refMAN.flatten()
        return man_list

    def graph_man(self, man_list):
        man_list = man_list.astype(float)
        man_list[np.where(man_list == 0)] = np.nan
        man_list[np.isinf(man_list)] = np.nan
        max_num_object = 39
        man_matrix = np.zeros((max_num_object, max_num_object))
        if man_list.size == 0:
            return man_matrix
        for i in range(39):
            for j in range(i + 1, 39):
                if (man_list[i][0] == man_list[j][0]) and (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 2
                elif (man_list[i][0] == man_list[j][0]) or (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 1
        return man_matrix

    def get_va(self, dsId, t, grid, vehId):
        va_list = np.full((len(grid), 2), 0)
        grid[19] = vehId
        refVA = np.zeros([0, 2])
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refVA = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refVA = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        if np.argwhere(vehTrack[:, 0] == t).size == 0:
                            return np.empty([0, 2])
                        else:
                            refVA = vehTrack[np.where(vehTrack[:, 0] == t)][0, 3:5]
            if refVA.size != 0:
                va_list[i] = refVA.flatten()
        return va_list

    def graph_ve(self, va_list):
        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 39
        ve_matrix = np.zeros((max_num_object, max_num_object))
        if va_list.size == 0:
            ve_matrix = torch.tensor(ve_matrix).float()
            return ve_matrix
        for i in range(39):
            for j in range(i + 1, 39):
                ve_matrix[i][j] = (va_list[i][0] - va_list[j][0])
                ve_matrix[j][i] = -ve_matrix[i][j]
        ve_matrix = torch.tensor(ve_matrix).float()
        return ve_matrix

    def graph_ac(self, va_list):
        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 39
        ac_matrix = np.zeros((max_num_object, max_num_object))
        if va_list.size == 0:
            ac_matrix = torch.tensor(ac_matrix).float()
            return ac_matrix
        for i in range(39):
            for j in range(i + 1, 39):
                ac_matrix[i][j] = (va_list[i][1] - va_list[j][1])
                ac_matrix[j][i] = -ac_matrix[i][j]
        ac_matrix = torch.tensor(ac_matrix).float()
        return ac_matrix

    def mask_view(self, dsId, t, grid, vehId):
        view_matrix = np.zeros((39))
        if vehId == 0:
            view_matrix1 = np.array(view_matrix)
            return view_matrix1.reshape(3, 13)
        else:
            if self.T.shape[1] <= vehId - 1:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(3, 13)
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            velocity = vehTrack[np.where(vehTrack[:, 0] == t)][0, 3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0 or grid.size == 0:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(3, 13)
            else:
                if velocity < 30:
                    indices = torch.tensor([4, 5, 6, 7, 8, 17, 18, 20, 21, 30, 31, 32, 33, 34])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)
                elif velocity >= 30 and velocity <= 60:
                    indices = torch.tensor([2, 3, 4, 8, 9, 10, 15, 16, 17, 18, 20, 21, 22, 23, 28, 29, 30, 34, 35, 36])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)
                elif velocity > 60:
                    indices = torch.tensor([0, 1, 2, 10, 11, 12, 13, 14, 15, 23, 24, 25, 26, 27, 28, 36, 37, 38])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)

    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    def collate_fn(self, samples):

        nbr_batch_size = 0

        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])

        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)

        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 3)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        edge_index_number = 0
        for sampleId, (_, _, _, _, _, _, _, _, _, _, _, edge_index, _, _, _, _) in enumerate(samples):
            edge_index_number += edge_index.shape[1]
        edge_index_batch = torch.zeros(len(samples), 2, edge_index_number)
        ve_matrix_batch = torch.zeros(len(samples), 39, 39)
        ac_matrix_batch = torch.zeros(len(samples), 39, 39)
        ve_matrix_batch = torch.zeros(len(samples), 39, 39)
        man_matrix_batch = torch.zeros(len(samples), 39, 39)
        view_grip_batch = torch.zeros(len(samples), 3, 13)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, lane, neighborslane, cclass, neighborsclass, va, neighborsva,
                       edge_index, ve_matrix, ac_matrix, man_matrix, view_grip) in enumerate(samples):

            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)

            ve_matrix = torch.tensor(ve_matrix)
            ve_matrix_batch[sampleId, :] = ve_matrix
            ac_matrix = torch.tensor(ac_matrix)
            ac_matrix_batch[sampleId, :] = ac_matrix
            man_matrix = torch.tensor(man_matrix)
            man_matrix_batch[sampleId, :] = man_matrix
            view_grip = torch.tensor(view_grip)
            view_grip_batch[sampleId, :] = view_grip

            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[
                        0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(
                        self.enc_size).byte()
                    count += 1

            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count3, :] = torch.from_numpy(nbrclass)
                    count3 += 1
            edge_index = torch.tensor(edge_index)
            buffer = edge_index.shape[1]
            edge_index_batch[sampleId, :,
            count4:count4 + buffer] = edge_index
            count4 += buffer

        max_num_object = 39

        graph_matrix = torch.zeros((128, max_num_object, 2))

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix


class umDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=25, d_s=1, t_h_stu=30, t_f_stu=50, d_s_stu=2, enc_size=64,
                 grid_size=(13, 3)):

        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']

        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s

        self.t_h_stu = t_h_stu
        self.t_f_stu = t_f_stu
        self.d_s_stu = d_s_stu

        self.enc_size = enc_size
        self.grid_size = grid_size

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx, 11:]
        neighbors = []
        neighborsclass = []
        neighborslane = []
        neighborsva = []

        neighbors_stu = []
        neighborsclass_stu = []
        neighborslane_stu = []
        neighborsva_stu = []

        hist_stu = self.getHistory_stu(vehId, t, vehId, dsId)
        fut_stu = self.getFuture_stu(vehId, t, dsId)
        lane_stu = self.getLane_stu(vehId, t, vehId, dsId)
        cclass_stu = self.getClass_stu(vehId, t, vehId, dsId)
        va_stu = self.getVA_stu(vehId, t, vehId, dsId)
        for i in grid:
            neighbors_stu.append(self.getHistory_stu(i.astype(int), t, vehId, dsId))
            neighborslane_stu.append(self.getLane_stu(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass_stu.append(self.getClass_stu(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsva_stu.append(self.getVA_stu(i.astype(int), t, vehId, dsId))

        hist = self.getHistory(vehId, t, vehId, dsId)
        fut = self.getFuture(vehId, t, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)
        va = self.getVA(vehId, t, vehId, dsId)

        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
        xy_list = self.get_xy(dsId, t, grid, vehId)
        edge_index = self.graph_xy(xy_list)
        va_list = self.get_va(dsId, t, grid, vehId)
        ve_matrix = self.graph_ve(va_list)
        ac_matrix = self.graph_ac(va_list)
        man_list = self.get_man(dsId, t, grid, vehId)
        man_matrix = self.graph_man(man_list)
        view_grip = self.mask_view(dsId, t, grid, vehId)

        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 9] - 1)] = 1

        return hist_stu, fut_stu, neighbors_stu, lane_stu, neighborslane_stu, cclass_stu, neighborsclass_stu, va_stu, neighborsva_stu, hist, fut, neighbors, lat_enc, lon_enc, lane, neighborslane, cclass, neighborsclass, va, neighborsva, edge_index, ve_matrix, ac_matrix, man_matrix, view_grip

    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])

            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getHistory_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 1:3] - refPos

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 2])
            return hist_stu

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getVA_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 3:5]

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 2])
            return hist_stu

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getLane_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 5]

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 1])
            return hist_stu

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass_stu(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h_stu)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist_stu = vehTrack[stpt:enpt:self.d_s_stu, 6]

            if len(hist_stu) < self.t_h_stu // self.d_s_stu + 1:
                return np.empty([0, 1])
            return hist_stu

    def get_xy(self, dsId, t, grid, vehId):
        xy_list = np.full((len(grid), 2), np.inf)
        grid[19] = vehId
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refPos = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refPos = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()

                    if vehTrack.size != 0:
                        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
            if refPos.size != 0:
                xy_list[i] = refPos.flatten()
        return xy_list

    def graph_xy(self, xy_list):

        node1 = []
        node2 = []

        xy_list = xy_list.astype(float)
        max_num_object = 39

        neighbor_matrix = np.zeros((max_num_object, max_num_object))

        dist_xy = spatial.distance.cdist(xy_list, xy_list)

        dist_xy[np.isinf(dist_xy)] = np.inf
        dist_xy[np.isnan(dist_xy)] = np.inf

        for i in range(39):
            for j in range(i + 1, 39):
                if dist_xy[i][j] <= 100:
                    node1.append(i)
                    node2.append(j)

        node1 = torch.tensor(node1).unsqueeze(0)
        node2 = torch.tensor(node2).unsqueeze(0)
        edge_index = torch.cat((node1, node2), dim=0)

        return edge_index

    def get_man(self, dsId, t, grid, vehId):
        man_list = np.full((len(grid), 2), 0)
        grid[19] = vehId
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refMAN = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refMAN = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        refMAN = vehTrack[np.where(vehTrack[:, 0] == t)][0, 7:9]
            if refMAN.size != 0:
                man_list[i] = refMAN.flatten()
        return man_list

    def graph_man(self, man_list):

        man_list = man_list.astype(float)
        man_list[np.where(man_list == 0)] = np.nan
        man_list[np.isinf(man_list)] = np.nan
        max_num_object = 39
        man_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(39):
            for j in range(i + 1, 39):
                if (man_list[i][0] == man_list[j][0]) and (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 2

                elif (man_list[i][0] == man_list[j][0]) or (man_list[i][1] == man_list[j][1]):
                    man_matrix[i][j] = man_matrix[j][i] = 1

        return man_matrix

    def get_va(self, dsId, t, grid, vehId):
        va_list = np.full((len(grid), 2), 0)
        grid[19] = vehId
        for i, neighbor in enumerate(grid):
            if neighbor == 0:
                refVA = np.zeros([0, 2])
            else:
                if self.T.shape[1] <= neighbor - 1:
                    refVA = np.empty([0, 2])

                else:
                    neighbor = neighbor.astype(int)
                    vehTrack = self.T[dsId - 1][neighbor - 1].transpose()
                    if vehTrack.size != 0:
                        refVA = vehTrack[np.where(vehTrack[:, 0] == t)][0, 3:5]

            if refVA.size != 0:
                va_list[i] = refVA.flatten()

        return va_list

    def graph_ve(self, va_list):

        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 39
        ve_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(39):
            for j in range(i + 1, 39):
                ve_matrix[i][j] = (va_list[i][0] - va_list[j][0])
                ve_matrix[j][i] = -ve_matrix[i][j]

        ve_matrix = torch.tensor(ve_matrix).float()
        return ve_matrix

    def graph_ac(self, va_list):

        va_list = va_list.astype(float)
        va_list[np.where(va_list == 0)] = np.nan
        va_list[np.isinf(va_list)] = np.nan
        max_num_object = 39
        ac_matrix = np.zeros((max_num_object, max_num_object))
        for i in range(39):
            for j in range(i + 1, 39):
                ac_matrix[i][j] = (va_list[i][1] - va_list[j][1])
                ac_matrix[j][i] = -ac_matrix[i][j]

        ac_matrix = torch.tensor(ac_matrix).float()
        return ac_matrix

    def mask_view(self, dsId, t, grid, vehId):

        view_matrix = np.zeros((39))
        if vehId == 0:
            view_matrix1 = np.array(view_matrix)
            return view_matrix1.reshape(3, 13)
        else:
            if self.T.shape[1] <= vehId - 1:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(3, 13)
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            velocity = vehTrack[np.where(vehTrack[:, 0] == t)][0, 3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0 or grid.size == 0:
                view_matrix1 = np.array(view_matrix)
                return view_matrix1.reshape(3, 13)
            else:
                if velocity < 30:
                    indices = torch.tensor([4, 5, 6, 7, 8, 17, 18, 20, 21, 30, 31, 32, 33, 34])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)

                elif velocity >= 30 and velocity <= 60:
                    indices = torch.tensor([2, 3, 4, 8, 9, 10, 15, 16, 17, 18, 20, 21, 22, 23, 28, 29, 30, 34, 35, 36])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)

                elif velocity > 60:
                    indices = torch.tensor([0, 1, 2, 10, 11, 12, 13, 14, 15, 23, 24, 25, 26, 27, 28, 36, 37, 38])
                    all_non_zero_mask = torch.nonzero(torch.tensor(grid))
                    view_matrix[all_non_zero_mask] = 0.2
                    non_zero_mask = torch.nonzero(torch.tensor(grid[indices]))
                    view_matrix[indices[non_zero_mask]] = 1
                    view_matrix1 = np.array(view_matrix)
                    return view_matrix1.reshape(3, 13)

    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    def getFuture_stu(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s_stu
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f_stu + 1)
        fut_stu = vehTrack[stpt:enpt:self.d_s_stu, 1:3] - refPos
        return fut_stu

    def collate_fn(self, samples):

        nbr_batch_size = 0
        nbr_batch_size_stu = 0

        for _, _, _, _, _, _, _, _, _, _, _, nbrs, _, _, _, _, _, _, _, _, _, _, _, _, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])

        for _, _, nbrs_stu, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ in samples:
            nbr_batch_size_stu += sum([len(nbrs_stu[i]) != 0 for i in range(len(nbrs_stu))])

        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        maxlen_stu = self.t_h_stu // self.d_s_stu + 1
        nbrs_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 2)
        nbrsva_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 2)
        nbrslane_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 1)
        nbrsclass_batch_stu = torch.zeros(maxlen_stu, nbr_batch_size_stu, 1)

        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)

        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        mask_batch_stu = mask_batch.bool()

        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)

        hist_batch_stu = torch.zeros(maxlen_stu, len(samples), 2)
        fut_batch_stu = torch.zeros(self.t_f_stu // self.d_s_stu, len(samples), 2)
        op_mask_batch_stu = torch.zeros(self.t_f_stu // self.d_s_stu, len(samples), 2)

        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 3)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)

        va_batch_stu = torch.zeros(maxlen_stu, len(samples), 2)
        lane_batch_stu = torch.zeros(maxlen_stu, len(samples), 1)
        class_batch_stu = torch.zeros(maxlen_stu, len(samples), 1)

        edge_index_number = 0
        for sampleId, (_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, edge_index, _, _, _, _) in enumerate(
                samples):
            edge_index_number += edge_index.shape[1]
        edge_index_batch = torch.zeros(len(samples), 2, edge_index_number)
        ve_matrix_batch = torch.zeros(len(samples), 39, 39)
        ac_matrix_batch = torch.zeros(len(samples), 39, 39)
        ve_matrix_batch = torch.zeros(len(samples), 39, 39)
        man_matrix_batch = torch.zeros(len(samples), 39, 39)
        view_grip_batch = torch.zeros(len(samples), 3, 13)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        count6 = 0
        count7 = 0
        count8 = 0
        for sampleId, (hist_stu, fut_stu, nbrs_stu, lane_stu, neighborslane_stu, cclass_stu, neighborsclass_stu, va_stu,
                       neighborsva_stu,
                       hist, fut, nbrs, lat_enc, lon_enc, lane, neighborslane, cclass, neighborsclass, va, neighborsva,
                       edge_index, ve_matrix, ac_matrix, man_matrix, view_grip) in enumerate(samples):

            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)

            hist_batch_stu[0:len(hist_stu), sampleId, 0] = torch.from_numpy(hist_stu[:, 0])
            hist_batch_stu[0:len(hist_stu), sampleId, 1] = torch.from_numpy(hist_stu[:, 1])
            fut_batch_stu[0:len(fut_stu), sampleId, 0] = torch.from_numpy(fut_stu[:, 0])
            fut_batch_stu[0:len(fut_stu), sampleId, 1] = torch.from_numpy(fut_stu[:, 1])

            va_batch_stu[0:len(va_stu), sampleId, 0] = torch.from_numpy(va_stu[:, 0])
            va_batch_stu[0:len(va_stu), sampleId, 1] = torch.from_numpy(va_stu[:, 1])
            lane_batch_stu[0:len(lane_stu), sampleId, 0] = torch.from_numpy(lane_stu)
            class_batch_stu[0:len(cclass_stu), sampleId, 0] = torch.from_numpy(cclass_stu)

            ve_matrix = torch.tensor(ve_matrix)
            ve_matrix_batch[sampleId, :] = ve_matrix
            ac_matrix = torch.tensor(ac_matrix)
            ac_matrix_batch[sampleId, :] = ac_matrix
            man_matrix = torch.tensor(man_matrix)
            man_matrix_batch[sampleId, :] = man_matrix
            view_grip = torch.tensor(view_grip)
            view_grip_batch[sampleId, :] = view_grip

            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[
                        0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(
                        self.enc_size).byte()
                    count += 1

            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count3, :] = torch.from_numpy(nbrclass)
                    count3 += 1

            for id_stu, nbr_stu in enumerate(nbrs_stu):
                if len(nbr_stu) != 0:
                    nbrs_batch_stu[0:len(nbr_stu), count5, 0] = torch.from_numpy(nbr_stu[:, 0])
                    nbrs_batch_stu[0:len(nbr_stu), count5, 1] = torch.from_numpy(nbr_stu[:, 1])
                    pos[0] = id_stu % self.grid_size[
                        0]
                    pos[1] = id_stu // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(
                        self.enc_size).byte()
                    count5 += 1

            for id_stu, nbrva_stu in enumerate(neighborsva_stu):
                if len(nbrva_stu) != 0:
                    nbrsva_batch_stu[0:len(nbrva_stu), count6, 0] = torch.from_numpy(nbrva_stu[:, 0])
                    nbrsva_batch_stu[0:len(nbrva_stu), count6, 1] = torch.from_numpy(nbrva_stu[:, 1])
                    count6 += 1

            for id_stu, nbrlane_stu in enumerate(neighborslane_stu):
                if len(nbrlane_stu) != 0:
                    nbrslane_batch_stu[0:len(nbrlane_stu), count7, :] = torch.from_numpy(nbrlane_stu)
                    count7 += 1

            for id_stu, nbrclass_stu in enumerate(neighborsclass_stu):
                if len(nbrclass_stu) != 0:
                    nbrsclass_batch_stu[0:len(nbrclass_stu), count8, :] = torch.from_numpy(nbrclass_stu)
                    count8 += 1

            buffer = edge_index.shape[1]
            edge_index_batch[sampleId, :,
            count4:count4 + buffer] = edge_index
            count4 += buffer

        max_num_object = 39

        graph_matrix = torch.zeros((256, max_num_object, 2))

        return hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu, \
            hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix
