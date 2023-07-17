import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, BertTokenizer

import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


# 暂时弃用
def text_tokenizer(text):
    return tokenizer(text, max_length=512, padding="max_length", truncation=True)


class SentenceDataset(Dataset):
    def __init__(self, paths: list, labels: list, ratio: float, train: bool):
        texts = []
        label_map = {}
        for i, label in enumerate(labels):
            label_map[label] = i
        for (p, l) in zip(paths, labels):
            label_index = label_map[l]
            # 划分数据集类型
            sentences = pd.read_csv(p)['sentence'].values.tolist()
            if train:
                sentences = sentences[:int(len(sentences) * ratio)]
            else:
                sentences = sentences[int(len(sentences) * ratio):]
            for sentence in sentences:
                text = {
                    'sentence': sentence,
                    'label': label_index
                }
                texts.append(text)
        self.texts = texts
        self.label_map = label_map

    def __getitem__(self, index):
        return self.texts[index]['sentence'], self.texts[index]['label']

    def __len__(self):
        return len(self.texts)


def collect_function(data):
    sentences = [d[0] for d in data]
    labels = [d[1] for d in data]

    feature = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                          padding="max_length",
                                          max_length=256,
                                          truncation=True,
                                          return_tensors="pt",
                                          return_length=True)
    input_ids = feature['input_ids']
    attention_mask = feature['attention_mask']
    token_type_ids = feature['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


def build_dataloader(paths: list,
                     labels: list,
                     batch_size: int,
                     train_ratio: float):
    train_dataset = SentenceDataset(paths, labels, train_ratio, True)
    eval_dataset = SentenceDataset(paths, labels, train_ratio, False)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collect_function,
                                  shuffle=True,
                                  drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collect_function,
                                 shuffle=True,
                                 drop_last=True)
    return train_dataloader, eval_dataloader
