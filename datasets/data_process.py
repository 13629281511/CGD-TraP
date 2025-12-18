import math
import os

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader


class NBADataset(Dataset):
    """Dataloader for the Trajectory datasets (NBA)"""
    def __init__(self, obs_len=8, pred_len=12, training=True, validation_split=0.8):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        """

        super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        # self.norm_lap_matr = norm_lap_matr

        if training:
            mode = 'train'
            data_root = './datasets/nba/nba_train.npy'
        else:
            mode = 'test'
            data_root = './datasets/nba/nba_test.npy'

        self.trajs = np.load(data_root)  # (N,30,11,2)
        self.trajs /= (94 / 28)
        if training:
            self.trajs = self.trajs[:32500]
        else:
            self.trajs = self.trajs[:12500]

        self.batch_len = len(self.trajs)
        print("Total {} datas are: {}".format(mode, self.batch_len))
        seq_list = self.trajs
        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs - self.trajs[:, self.obs_len - 1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0, 2, 1, 3)
        self.traj_norm = self.traj_norm.permute(0, 2, 1, 3)
        self.actor_num = self.traj_abs.shape[1]  # num_agent

        # calculate the mean value of NBA trajectory
        # compress_seq = seq_list.transpose(0, 2, 1)
        # new_seq = np.concatenate(seq_list, axis=-1)
        # mean_values = new_seq.mean(axis=1)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = self.traj_abs[index, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[index, :, self.obs_len:, :]
        pre_motion_mask = torch.ones(11, self.obs_len)
        fut_motion_mask = torch.ones(11, self.pred_len)

        results = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
            'pre_motion_mask': pre_motion_mask,
            'fut_motion_mask': fut_motion_mask
        }
        return results


def initial_pos(traj_batches):
    batches = []
    for b in traj_batches:
        starting_pos = b[:,7,:].copy()/1000  # starting pos is end of past, start of future. scaled down.
        batches.append(starting_pos)

    return batches


class SDDDataset(Dataset):
    def __init__(self, data_path, obs_len, pred_len, flip_aug=True, mode='train'):
        super(SDDDataset, self).__init__()

        self.data_path = data_path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mode = mode
        self.sdd_dataset, self.split_marks = [], [0]
        s = 0

        if self.mode == 'train':
            data_path = data_path + '/' + 'sdd/train_8_12.npy'
        elif self.mode == 'test':
            data_path = data_path + '/' + 'sdd/val_8_12.npy'
        data = np.load(data_path)

        assert data.shape[1] == obs_len + pred_len
        self.sdd_dataset.append(data[:, :, 0:2])
        s += len(data)
        self.split_marks.append(s)

        if self.mode == 'train':
            if flip_aug:
                flipped_data = np.flip(data, axis=1)
                self.sdd_dataset.append(flipped_data[:, :, 0:2])
                s += len(flipped_data)
                self.split_marks.append(s)

        self.sdd_dataset = np.concatenate(self.sdd_dataset, axis=0)
        print("Total {} datas are: {}".format(self.mode, len(self.sdd_dataset)))
        assert len(self.sdd_dataset) == s
        self.len = len(self.sdd_dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        results = {
            'past_traj': self.sdd_dataset[index, :self.obs_len, :],
            'fut_traj': self.sdd_dataset[index, self.obs_len:, :],
        }
        return results
