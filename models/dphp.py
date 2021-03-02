# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DPHP(nn.Module):
    '''
    Personalized Attention Review
    '''
    def __init__(self, opt, uori='user'):
        super(DPHP, self).__init__()
        self.opt = opt
        self.word_embs = nn.Embedding.from_pretrained(torch.from_numpy(np.load(opt.w2v_path)).float(), freeze=False)
        self.review_rnn = nn.GRU(opt.word_dim,
                                 opt.filters_num//2,
                                 num_layers=2,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=opt.drop_out)

        self.rnn_fc = nn.Linear(opt.filters_num, opt.id_emb_size)

        self.user_embs = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.item_embs = nn.Embedding(opt.item_num, opt.id_emb_size)

        if self.opt.use_uid and self.opt.use_iid:
            feature_dim = self.opt.id_emb_size * 3
        elif not self.opt.use_uid and not self.opt.use_iid:
            feature_dim = self.opt.id_emb_size
        else:
            feature_dim = self.opt.id_emb_size * 2

        self.fc = nn.Sequential(
            nn.Dropout(opt.drop_out),
            nn.Linear(feature_dim, opt.id_emb_size),
            nn.ReLU()
        )

        self.id_fc = nn.Linear(opt.id_emb_size, opt.filters_num)
        self.fu_fc = nn.Linear(opt.filters_num, opt.filters_num)
        self.fi_fc = nn.Linear(opt.filters_num, opt.filters_num)

        self.cls = nn.Linear(opt.id_emb_size, 1)
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.reset_para()

    def forward(self, data):
        uid, iid, reviews, review2len = data
        reviews = self.word_embs(reviews)
        u_fea = self.user_embs(uid)
        i_fea = self.user_embs(iid)

        r_fea, _ = self.review_rnn(reviews)  # BS*n*2l

        uid_q = self.id_fc(u_fea)
        iid_q = self.id_fc(i_fea)
        if self.opt.pool == 'att':
            ur_fea, u_socre = self.pooling(self.opt.pool, r_fea, query=uid_q, seq_len=review2len)
            ir_fea, i_socre = self.pooling(self.opt.pool, r_fea, query=iid_q, seq_len=review2len)

            if not self.opt.use_uid:
                r_fea = self.dropout(self.fi_fc(ir_fea))
            if not self.opt.use_iid:
                r_fea = self.dropout(self.fu_fc(ur_fea))
            else:
                r_fea = self.dropout(self.fu_fc(ur_fea) + self.fi_fc(ir_fea))
        else:
            r_fea, _ = self.pooling(self.opt.pool, r_fea)

        r_fea = self.rnn_fc(r_fea)

        if self.opt.use_uid and self.opt.use_iid:
            fea = torch.cat([u_fea, r_fea, i_fea], 1)
        elif self.opt.use_uid and not self.opt.use_iid:
            fea = torch.cat([u_fea, r_fea], 1)
        elif not self.opt.use_uid and self.opt.use_iid:
            fea = torch.cat([r_fea, i_fea], 1)
        else:
            fea = r_fea

        fea = self.fc(fea)

        fea = self.dropout(fea)
        res = self.cls(fea)
        return res.squeeze(-1)

    def pooling(self, pool, H, query=None, seq_len=None):
        if pool == 'avg':
            return H.mean(1), None
        if pool == 'last':
            return H[:, -1, :], None
        if pool == 'att':
            assert query is not None
            if query.size(1) == 1:   # Vanilla attention
                weight = H.matmul(query)  # BS*n*1
            else:
                weight = H.bmm(query.unsqueeze(2))  # BS * n * 1
            if seq_len is not None:
                weight = self.Mask(weight, seq_len)
            score = F.softmax(weight, 1)
            value = (H * score).sum(1)
            return value, score
        else:
            raise "Pool Para Error"

    def sequence_mask(self, sequence_length, max_len):
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).cuda()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def Mask(self, inputs, seq_len=None):
        """
        inputs: (B, L, K), the possibilities we need to mask
        sqe_len: (B)
        """
        if seq_len is None:
            return inputs
        mask = self.sequence_mask(seq_len, self.opt.r_max_len)  # (B, L)
        mask = mask.unsqueeze(-1)      # (B, L, 1)
        outputs = inputs - ~mask * 1e12
        return outputs

    def reset_para(self):
        nn.init.xavier_normal_(self.user_embs.weight, gain=1)
        nn.init.xavier_normal_(self.item_embs.weight, gain=1)

        nn.init.uniform_(self.rnn_fc.weight, -0.2, 0.2)
        # nn.init.xavier_uniform_(self.rnn_fc.weight, gain=1)
        nn.init.uniform_(self.rnn_fc.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc[1].weight, -0.2, 0.2)
        # nn.init.xavier_uniform_(self.fc[1].weight, gain=1)
        nn.init.uniform_(self.fc[1].bias, -0.1, 0.1)
        nn.init.uniform_(self.cls.weight, -0.2, 0.2)
        # nn.init.xavier_uniform_(self.cls.weight, gain=1)
        nn.init.uniform_(self.cls.bias, -0.1, 0.1)

        nn.init.uniform_(self.fu_fc.weight, -0.2, 0.2)
        nn.init.uniform_(self.fu_fc.bias, -0.2, 0.2)

        nn.init.uniform_(self.fi_fc.weight, -0.2, 0.2)
        nn.init.uniform_(self.fi_fc.bias, -0.2, 0.2)

        nn.init.uniform_(self.id_fc.weight, -0.2, 0.2)
        nn.init.uniform_(self.id_fc.bias, -0.2, 0.2)
