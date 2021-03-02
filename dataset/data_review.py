# -*- coding: utf-8 -*-

import os
import numpy as np
from torch.utils.data import Dataset


class ReviewData(Dataset):

    def __init__(self, root_path, mode="Train"):
        if mode == "Train":
            path = os.path.join(root_path, 'train/npy/')
            data = np.load(path + 'Train.npy', encoding='bytes')
            rating_score = np.load(path + 'Train_Rating_Score.npy')
            is_helpful = np.load(path + 'Train_Is_Helpful.npy')
            helpful_score = np.load(path + 'Train_Helpful_Score.npy')
        elif mode == 'Val':
            path = os.path.join(root_path, 'val/npy/')
            data = np.load(path + 'Val.npy', encoding='bytes')
            rating_score = np.load(path + 'Val_Rating_Score.npy')
            is_helpful = np.load(path + 'Val_Is_Helpful.npy')
            helpful_score = np.load(path + 'Val_Helpful_Score.npy')
        else:
            path = os.path.join(root_path, 'test/npy/')
            data = np.load(path + 'Test.npy', encoding='bytes')
            rating_score = np.load(path + 'Test_Rating_Score.npy')
            is_helpful = np.load(path + 'Test_Is_Helpful.npy')
            helpful_score = np.load(path + 'Test_Helpful_Score.npy')

        self.x = list(zip(data, rating_score, is_helpful, helpful_score))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
