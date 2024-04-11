"""
====================================================
@Project:   MyBERT -> bert_classification
@Author:    TropicalAlgae
@Date:      2023/7/12 21:46
@Desc:
====================================================
"""

from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer


class ClassifierBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# class ClassifierBert(nn.Module):
#     def __init__(self,
#                  model: str,
#                  out_channel: int = 3,
#                  dropout: float = 0.1):
#         super(ClassifierBert, self).__init__()
#
#         self.bert_module = BertModel.from_pretrained(model)
#
#         self.bert_config = self.bert_module.config
#
#         self.dropout_layer = nn.Dropout(dropout)
#         out_dims = self.bert_config.hidden_size
#         self.obj_classifier = nn.Linear(out_dims, out_channel)
#
#     def forward(self,
#                 input_ids,
#                 attention_mask,
#                 token_type_ids):
#         bert_outputs = self.bert_module(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         seq_out, pooled_out = bert_outputs[0], bert_outputs[1]
#         # 对反向传播及逆行截断
#         # x = pooled_out.detach()
#         x = self.dropout_layer(pooled_out)
#         out = self.obj_classifier(x)
#         return out


def get_classification_model(pretrain_model: str,
                             num_labels: int = 3,
                             classifier_dropout: int = 0.1):
    model = ClassifierBert.from_pretrained(pretrain_model,
                                           num_labels=num_labels,
                                           classifier_dropout=classifier_dropout)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    return model, tokenizer
