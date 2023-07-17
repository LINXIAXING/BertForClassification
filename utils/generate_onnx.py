import os
import torch
import yaml
from easydict import EasyDict
from transformers import BertTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

config = EasyDict(yaml.load(open('../config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))

check_point = f'../{config.trainer.check_point}/pipeline'

save_pt = config.other.save_pt
save_onnx = config.other.save_onnx

if not os.path.isdir(save_pt):
    os.makedirs(save_pt)
if not os.path.isdir(save_onnx):
    os.makedirs(save_onnx)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = ORTModelForSequenceClassification.from_pretrained(check_point, export=True)

tokenizer.save_pretrained(save_onnx)
model.save_pretrained(save_onnx)

# 传统方法，弃用
# # save pt
# torch.save(tokenizer, save_pt + '/tokenizer.pt')
# torch.save(model, save_pt + '/bert_clz.pt')
#
# tokenizer_pt = torch.load(save_pt + '/tokenizer.pt')
# model_pt = torch.load(save_pt + '/bert_clz.pt')
# # save onnx
# text = ["这间店环境真的很差，我感觉不太行"]
# token = tokenizer(text)
# torch.onnx.export(tokenizer_pt, token, save_onnx + "/tokenizer.onnx", opset_version=10)
# torch.onnx.export(model_pt, torch.tensor(token["input_ids"]), save_onnx + "/bert_clz.onnx", opset_version=10)
