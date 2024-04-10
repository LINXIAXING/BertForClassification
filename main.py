import yaml
from easydict import EasyDict

from src.trainer import Trainer

if __name__ == '__main__':
    config = EasyDict(yaml.load(open('./config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    Trainer(**config.trainer).train()
