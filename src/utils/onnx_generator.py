import os
from transformers import BertTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

from src.utils.util import build_folder, get_newest_checkpoint_path


class OnnxGenerator:
    def __init__(self, checkpoint, describe: str, save_path):
        if describe.split('_v')[-1].isdigit():
            checkpoint_root = os.path.join(checkpoint, describe)
        else:
            checkpoint_root = get_newest_checkpoint_path(checkpoint, describe)
        assert checkpoint_root != "", "Non-existent model path, please check your configuration!"

        print(f"loading model from {checkpoint_root}")
        self.model = os.path.join(checkpoint_root, "model", "best")
        self.tokenizer = os.path.join(checkpoint_root, "tokenizer")
        self.save_path = build_folder(save_path, describe)

    def generate(self):
        # load model
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer)
        model = ORTModelForSequenceClassification.from_pretrained(self.model, export=True)

        # save model & tokenizer
        tokenizer.save_pretrained(self.save_path)
        print("Tokenizer has been saved.")
        model.save_pretrained(self.save_path)
        print("Model (onnx) has been saved.")

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
