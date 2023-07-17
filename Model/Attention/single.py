"""
====================================================
@Project:   MyBERT -> single
@Author:    TropicalAlgae
@Date:      2023/2/13 18:17
@Desc:
====================================================
"""
import torch


def attention(query: torch.Tensor,  # [batch_size, len, embed_dim]
              key: torch.Tensor,  # [batch_size, len, embed_dim]
              value: torch.Tensor,  # [batch_size, len, embed_dim]
              head_num: int,
              dropout: int = 0,
              train: bool = True,
              q_linear=None,
              k_linear=None,
              v_linear=None,
              out_linear=None,
              mask=None):
    q = q_linear(query)  # [batch_size, len, k_dim * head_num] k_dim * head_num = embed_dim (k_dim = head_dim)
    k = k_linear(key)  # [batch_size, len, k_dim * head_num] k_dim * head_num = embed_dim
    v = v_linear(value)  # [batch_size, len, v_dim * head_num] v_dim * head_num = embed_dim (v_dim = head_dim)

    batch_size, ipt_len, embed_dim = query.size()
    head_dim = embed_dim // head_num
    assert head_dim * head_num == embed_dim, "嵌入维度是head数量的整数倍"

    q = q * float(head_dim) ** -0.5  # [batch_size, len, k_dim * head_num]

    q = q.reshape(batch_size * head_num, ipt_len, head_dim)  # [batch_size * head_num, len, k_dim]
    k = k.reshape(batch_size * head_num, k.shape[1], head_dim)  # [batch_size * head_num, len, k_dim]
    v = v.reshape(batch_size * head_num, v.shape[1], head_dim)  # [batch_size * head_num, len, k_dim]

    # 计算softmax(q(k^T) / (dim ** 0.5))v
    atte_output = torch.matmul(q, k.transpose(1, 2))  # [batch_size * head_num, len, len]
    # 添加mask
    if mask is not None:
        pass
    atte_output = torch.softmax(atte_output, dim=-1)  # 对dim求softmax
    atte_output = torch.dropout(atte_output, dropout, train)
    atte_output = torch.matmul(atte_output, v)  # [batch_size * head_num, len, k_dim]

    atte_output = atte_output.reshape(batch_size, ipt_len, atte_output.size(-1) * head_num)  # [batch_size, len, embed_dim]
    z = out_linear(atte_output)

    return z, atte_output.sum(dim=1) / head_num


