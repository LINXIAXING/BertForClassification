import sys
from typing import Literal

import yaml
from easydict import EasyDict

from src.trainer import Trainer
from src.utils.onnx_generator import OnnxGenerator
from src.utils.util import get_checkpoint_path, build_folder


def launch(type: Literal["train_multi_gpu", "train_single_gpu", "init_folder", "onnx"]):
    config = EasyDict(yaml.load(open('./config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    if type == "train_multi_gpu":
        Trainer(**config.trainer, describe=config.describe, multi_gpu=True).train()
    if type == "train_single_gpu":
        Trainer(**config.trainer, describe=config.describe, multi_gpu=False).train()
    if type == "init_folder":
        checkpoint_root, is_new = get_checkpoint_path(root=config.trainer.checkpoint,
                                                      describe=config.describe,
                                                      restore_train=config.trainer.restore_train)
        if is_new:
            build_folder(checkpoint_root)
    if type == "onnx":
        describe = config.describe if config.onnx.describe == "" else config.onnx.describe
        OnnxGenerator(checkpoint=config.trainer.checkpoint,
                      describe=describe,
                      save_path=config.onnx.save_path).generate()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        launch(sys.argv[1])
    else:
        launch("train_single_gpu")
