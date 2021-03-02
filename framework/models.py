# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import time


class Model(nn.Module):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt
        self.model_name = self.opt.model
        self.model = Net(opt)

    def forward(self, datas):
        return self.model(datas)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, epoch=None, name=None, opt=None):
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name) + '_' + str(opt) + '.pth'
        torch.save(self.state_dict(), name)
        return name
