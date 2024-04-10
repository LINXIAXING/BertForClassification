"""
====================================================
@Project:   MyBERT -> multi_head
@Author:    TropicalAlgae
@Date:      2023/2/13 18:14
@Desc:
====================================================
"""
import torch
from torch import nn

from Model.Attention.single import attention


# 多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_num, dropout, bias=True, train=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        self.dropout = dropout
        assert self.head_dim * self.head_num == embed_dim, "嵌入维度是head数量的整数倍"
        self.is_train = train
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value):
        return attention(query=query,
                         key=key,
                         value=value,
                         head_num=self.head_num,
                         dropout=self.dropout,
                         train=self.is_train,
                         q_linear=self.q_linear,
                         k_linear=self.q_linear,
                         v_linear=self.v_linear,
                         out_linear=self.out_linear)
