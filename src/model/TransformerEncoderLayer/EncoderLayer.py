"""
====================================================
@Project:   MyBERT -> EncoderLayer
@Author:    TropicalAlgae
@Date:      2023/2/18 15:20
@Desc:
====================================================
"""
import copy

import torch
from torch import nn
from torch.nn.functional import relu
from src.model.Attention.multi_head import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, feedforward, head_num, dropout, train):
        super(EncoderBlock, self).__init__()
        self.multi_attention = MultiHeadAttention(embed_dim=embed_dim, head_num=head_num, dropout=dropout, train=train)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.activate = relu
        self.linear1 = nn.Linear(embed_dim, feedforward)
        self.linear2 = nn.Linear(feedforward, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src):  # [batch_size, len, embed_dim]
        print("get one block")
        atte = self.multi_attention(src, src, src)[0]
        src = src + self.dropout1(atte)
        src = self.norm1(src)

        src = self.activate(self.linear1(src))
        src = self.dropout2(src)
        src = self.linear2(src)
        src = self.norm2(src)
        return src


class EncoderLayer(nn.Module):
    def __init__(self, encoder_block, encoder_num: int = 6):
        super(EncoderLayer, self).__init__()
        self.encoders = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(encoder_num)])
        self.encoder_num = encoder_num

    def forward(self, src):
        out = src
        for encoder in self.encoders:
            out = encoder(out)
        return out
