from time import sleep

import yaml
from easydict import EasyDict
from objprint import objstr
from tqdm import tqdm
from transformers import BertTokenizer
from onnxruntime import InferenceSession
import torch

from src.model.Attention.multi_head import MultiHeadAttention
from src.model.TransformerEncoderLayer.EncoderLayer import EncoderBlock, EncoderLayer


def test_multi_attention():
    a = MultiHeadAttention(embed_dim=3, head_num=1, dropout=0.1)
    query = torch.randn(4, 2, 3)
    key = torch.randn(4, 2, 3)
    value = torch.randn(4, 2, 3)
    print(query)
    print(key)
    print(value)
    z, out = a(query, key, value)
    print("z: " + str(z))
    print("out: " + str(out))


def test_encoder_block():
    a = EncoderBlock(embed_dim=8, feedforward=1024, head_num=2, dropout=0.1, train=True)
    src = torch.randn(5, 10, 8)
    out = a(src)
    print("out size: " + str(out.shape))


def test_encoder_layer():
    a = EncoderBlock(embed_dim=8, feedforward=1024, head_num=2, dropout=0.1, train=True)
    b = EncoderLayer(encoder_block=a, encoder_num=6)
    src = torch.randn(5, 10, 8)
    out = b(src)
    print("out size: " + str(out.shape))


# config = EasyDict(yaml.load(open('./config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
# data = [('真正执行collate_', 0), ('ataloaders进行for调用，后再断点_', 1), ('函数里面做', 0), ('h中不一样长的句子paddin', 0)]
# train_dataloader, eval_dataloader = build_dataloader(**config.dataset)
# for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(eval_dataloader):
#     print(len(input_ids))
#     print(len(attention_mask))


# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# session = InferenceSession("src/utils/save/onnx/pretrain_checkpoint.onnx")
# # ONNX Runtime expects NumPy arrays as input
# inputs = tokenizer("这家店的环境一言难尽", return_tensors="np")
# output = session.run(output_names=["logits"], input_feed=dict(inputs))
