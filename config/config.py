# -*- coding: utf-8 -*-

import numpy as np


class DefaultConfig:

    model = 'DPHP'  # prior gru double attention network
    dataset = 'Gourmet_Food_data'
    drop_out = 0.1

    # --------------optimizer---------------------#
    optimizer = 'Adam'
    weight_decay = 5e-4  # optimizer rameteri
    lr = 1e-3

    # -------------settings-----------------------#
    seed = 2019
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []
    use_gpu = True
    num_epochs = 5

    # -------------models-----------------------#
    id_emb_size = 32
    filters_num = 100
    use_uid = True
    use_iid = True
    char_num = 40
    pool = 'att'

    print_opt = 'def'

    def __repr__(self):
        return f"cfg-{self.dataset}-pool{self.pool}-lr{self.lr}-wd{self.weight_decay}-drop{self.drop_out}-id{self.id_emb_size}-hidden{self.filters_num}"

    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train/npy'
        self.w2v_path = f'{prefix}/w2v.npy'
        self.pair_review_path = f"{prefix}/pair_review_dict.npy"
        self.pair_len_path = f"{prefix}/review2len.npy"

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        self.pair_review_dict = np.load(self.pair_review_path, allow_pickle=True).tolist()
        self.pair_review2len_dict = np.load(self.pair_len_path, allow_pickle=True).tolist()

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)


class Gourmet_Food_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Gourmet_Food_data')

    vocab_size = 20002
    word_dim = 300
    r_max_len = 161  # review max length
    t_max_len = 14
    c_max_len = 7

    user_num = 13528 + 2
    item_num = 8453 + 2

    train_data_size = 52826
    test_data_size = 11931
    batch_size = 512
