"""
====================================================
@Project:   MGAT -> train
@Author:    TropicalAlgae
@Date:      2023/6/20 19:11
@Desc:
====================================================
"""
import os
import time
import logging

import evaluate
import yaml
from objprint import objstr
from accelerate import Accelerator
from accelerate.logging import get_logger
from easydict import EasyDict
from accelerate.utils.random import set_seed
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from transformers import BertForSequenceClassification

from DataLoader.data_process import build_dataloader
# from model.bert_classification import ClassifierBert
from utils import util


class Trainer:
    def __init__(self, cfg: EasyDict):
        set_seed(23)
        torch.multiprocessing.set_sharing_strategy("file_system")
        log_dir = os.path.join(cfg.trainer.log_root, time.strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.cfg = cfg
        self.restore = cfg.trainer.restore_train
        self.epochs = cfg.trainer.epochs
        self.starting_epoch = 0
        self.train_step = 0
        self.val_step = 0

        self.accelerator = Accelerator(
            log_with="tensorboard",  # type:ignore
            project_dir=log_dir,
            mixed_precision=cfg.trainer.mixed_precision,
            gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps
        )
        self.accelerator.init_trackers(__name__)  # 此方法设置跟踪训练期间的各种指标
        if self.accelerator.is_main_process:  # 这用于确保在分布式训练期间仅在一个进程上执行某些代码
            logging.basicConfig(
                level=logging.INFO,  # 日志等级
                format="[%(asctime)s] %(levelname)s %(message)s",  # 日志格式
                datefmt="%Y-%m-%d-%H:%M:%S",
                handlers=[
                    logging.StreamHandler(),  # 两个日志处理程序：一个记录到控制台，一个记录到log.txt文件中
                    logging.FileHandler(log_dir + "/log.txt"),
                ],
                force=True,
            )
        self.logger = get_logger(__name__)
        self.logger.info("Record training config:")
        self.logger.info(objstr(cfg))
        self.logger.info("Load dataset:")
        self.train_loader, self.eval_loader = build_dataloader(**cfg.dataset)
        self.logger.info("Load model:")
        # self.metrics = {
        #     "recall": torchmetrics.classification.Recall(task='binary'),
        #     "accuracy": torchmetrics.classification.accuracy.Accuracy(task='binary'),
        #     "f1": torchmetrics.classification.f_beta.FBetaScore(task='binary', beta=1.0),
        #     "precision": torchmetrics.classification.average_precision.AveragePrecision(task='binary')
        # }
        self.metric = evaluate.load("accuracy")
        self.loss_functions = {
            "CrossEntropy": torch.nn.CrossEntropyLoss()
        }
        # self.model = ClassifierBert(**cfg.model)
        self.model = BertForSequenceClassification.from_pretrained(
            "IDEA-CCNL/Taiyi-CLIP-RoBERTa-326M-ViT-H-Chinese",
            num_labels=3,
            problem_type="single_label_classification",
        )
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr=cfg.trainer.opt.lr,
                                           weight_decay=cfg.trainer.opt.weight_decay)
        # 余弦退火
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer,
                                                     T_0=cfg.trainer.scheduler.T_0,
                                                     T_mult=cfg.trainer.scheduler.T_mult)

    def _train_one_epoch(self, epoch):
        self.model.train()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.train_loader):
            with self.accelerator.accumulate(self.model):
                # 模型推理
                # total_loss = 0
                log = ''
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                loss = output.loss
                # 计算loss
                # for loss_name, loss_fn in self.loss_functions.items():
                #     loss = loss_fn(prediction, labels)
                #     total_loss += loss
                #     log += f'{loss_name}: {loss}\t'
                log = f'loss: {loss}\t'
                predictions, references = self.accelerator.gather_for_metrics(
                    (output.logits.argmax(dim=-1), labels)
                )
                self.metric.add_batch(predictions=predictions, references=references)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                # 训练日志
                self.accelerator.log({
                    'Train/Total Loss': float(loss),
                }, step=self.train_step)
                self.logger.info(
                    f'Epoch [{epoch + 1}/{self.cfg.trainer.epochs}]\tTraining [{i + 1}/{len(self.train_loader)}]\t\
                    Loss: {float(loss):1.5f}\t[{log}]')
                self.train_step += 1
        metric = self.metric.compute()
        self.logger.info(f'Epoch [{epoch + 1}/{self.cfg.trainer.epochs}] Training metric {metric}')
        self.accelerator.log(metric, step=epoch)

    @torch.no_grad()
    def _eval_one_epoch(self, epoch):
        self.model.eval()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.eval_loader):
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=labels)

            loss = output.loss
            log = f'loss: {loss}\t'
            predictions, references = self.accelerator.gather_for_metrics(
                (output.logits.argmax(dim=-1), labels)
            )
            self.metric.add_batch(predictions=predictions, references=references)

            self.accelerator.log({
                'Val/ Loss': float(loss),
            }, step=self.val_step)
            self.logger.info(
                f'Epoch [{epoch + 1}/{self.cfg.trainer.epochs}] Validation [{i + 1}/{len(self.eval_loader)}] \
                Loss: {(float(loss)):1.5f} {log}')
            self.val_step += 1

        metric = self.metric.compute()
        self.logger.info(f'Epoch [{epoch + 1}/{self.cfg.trainer.epochs}] Validation metric {metric}')
        self.accelerator.log(metric, step=epoch)
        return metric['accuracy']

    def accelerator_control(self):
        self.accelerator.wait_for_everyone()
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.eval_loader
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader, self.eval_loader  # type:ignore
        )

    def train(self):
        self.accelerator_control()
        # 尝试继续训练
        # if self.restore:
        #     self.starting_epoch, self.train_step, self.val_step = util.resume_train_state(
        #         self.cfg.trainer.finetune_save_dir, self.train_loader, self.eval_loader, self.logger, self.accelerator
        #     )
        self.logger.info("Start training")
        best_acc = 0
        for epoch in range(self.starting_epoch, self.epochs):
            # 推理 + 预测
            self._train_one_epoch(epoch=epoch)
            mean_acc = self._eval_one_epoch(epoch=epoch)
            self.accelerator.log({'lr': self.scheduler.get_last_lr()}, step=epoch)
            # 存储模型
            if best_acc < mean_acc:
                best_acc = mean_acc
                self.accelerator.save_state(
                    output_dir=f'{os.path.join(os.getcwd(), self.cfg.trainer.check_point, "model")}'
                )
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    f'{os.path.join(os.getcwd(), self.cfg.trainer.check_point, "pipeline")}')
            self.logger.info(
                "--------- Epoch [{}/{}] Verification results --- mean acc :{} --- best acc:{} --- lr = {} ---------"
                .format(epoch + 1, self.cfg.trainer.epochs, mean_acc, best_acc, self.scheduler.get_last_lr()))
        self.logger.info(f"----------End of all training----------/n")
        self.logger.info(f"Best acc: {best_acc}")
        exit(1)

    def test(self):
        self.accelerator_control()
        # 加载最优模型
        self.model.load_state_dict(torch.load(
            os.path.join(os.getcwd(), self.cfg.trainer.check_point, "best", "pytorch_model.bin")
        ))
        # 开始测试
        self.logger.info("Start testing")


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('./config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    Trainer(config).train()
