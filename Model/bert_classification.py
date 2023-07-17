"""
====================================================
@Project:   MyBERT -> bert_classification
@Author:    TropicalAlgae
@Date:      2023/7/12 21:46
@Desc:
====================================================
"""

from torch import nn
from transformers import BertModel


class ClassifierBert(nn.Module):
    def __init__(self,
                 out_channel: int = 3,
                 dropout: float = 0.1):
        super(ClassifierBert, self).__init__()
        self.bert_module = BertModel.from_pretrained("bert-base-chinese")

        self.bert_config = self.bert_module.config

        self.dropout_layer = nn.Dropout(dropout)
        out_dims = self.bert_config.hidden_size
        self.obj_classifier = nn.Linear(out_dims, out_channel)

    def forward(self,
                input_ids,
                attention_mask,
                segment_ids):
        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids
        )
        seq_out, pooled_out = bert_outputs[0], bert_outputs[1]
        # 对反向传播及逆行截断
        x = pooled_out.detach()
        x = self.dropout_layer(x)
        out = self.obj_classifier(x)
        return out
