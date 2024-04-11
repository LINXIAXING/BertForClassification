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
from objprint import objstr
from accelerate import Accelerator
from accelerate.logging import get_logger
from easydict import EasyDict
from accelerate.utils.random import set_seed
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from transformers import BertForSequenceClassification

from src.dataLoader.data_process import build_dataloader
from src.model.bert_classification import get_classification_model
from src.utils.util import get_checkpoint_path, resume_train_state, build_folder, get_newest_checkpoint_path


class Trainer:
    def __init__(self,
                 seed: int,
                 model: EasyDict,
                 optim: EasyDict,
                 scheduler: EasyDict,
                 dataset: EasyDict,
                 describe: str,
                 checkpoint: str,
                 epochs: int,
                 eval_step: int,
                 restore_train: bool,
                 save_all_epoch: bool,
                 mixed_precision: str,
                 gradient_accumulation_steps: int,
                 multi_gpu: bool = False):
        assert all(char not in describe for char in '/:*"<>|,'), "describe contains illegal characters!"

        set_seed(seed)
        torch.multiprocessing.set_sharing_strategy("file_system")

        # define the path of log and checkpoint
        if multi_gpu:
            checkpoint_root = get_newest_checkpoint_path(checkpoint, describe)
        else:
            checkpoint_root, _ = get_checkpoint_path(checkpoint, describe, restore_train)
        log_dir = build_folder(checkpoint_root, "log", time.strftime("%Y-%m-%d-%H-%M-%S"))
        model_dir = build_folder(checkpoint_root, "model")
        tokenizer_dir = build_folder(checkpoint_root, "tokenizer")

        # set the handler for accelerator log
        sh = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        sh.setLevel(20)
        fh.setLevel(15)
        # set the params
        self.restore = restore_train
        self.save_all_epoch = save_all_epoch
        self.epochs = epochs
        self.eval_step = eval_step
        self.starting_epoch = 0
        self.train_step = 0
        self.val_step = 0
        self.check_point = model_dir
        self.accelerator = Accelerator(log_with="tensorboard",  # type:ignore
                                       project_dir=log_dir,
                                       mixed_precision=mixed_precision,
                                       gradient_accumulation_steps=gradient_accumulation_steps)
        self.accelerator.init_trackers(__name__)
        if self.accelerator.is_main_process:
            logging.basicConfig(level=15,
                                format="[%(asctime)s] %(filename)s -> %(funcName)s | "
                                       "line:%(lineno)d [%(levelname)s]:\t%(message)s",
                                datefmt="%Y-%m-%d-%H:%M:%S",
                                handlers=[sh, fh],
                                force=True, )
            logging.addLevelName(15, "MESG")

        # load model & dataloader
        self.logger = get_logger(__name__)
        self.logger.info("Recording training config...")
        self.logger.info(objstr({k: v for k, v in locals().items() if k != "self"}))
        self.logger.info(f"Loading model...")
        self.model, tokenizer = get_classification_model(**model)
        tokenizer.save_pretrained(tokenizer_dir)
        # self.logger.info("-------------------------")
        # self.model = BertForSequenceClassification.from_pretrained(model.pretrain_model,
        #                                                            num_labels=3,
        #                                                            problem_type="single_label_classification", )
        self.logger.info("Loading dataset...")
        self.train_loader, self.eval_loader = build_dataloader(**dataset, tokenizer=tokenizer)

        # set Evaluation indicators, loss and optimizer
        self.metric = evaluate.load("./src/metrics/accuracy")
        self.loss_functions = {
            "CrossEntropy": torch.nn.CrossEntropyLoss()
        }
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr=optim.lr,
                                           weight_decay=optim.weight_decay)

        # cosine annealing for learning rate
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer,
                                                     T_0=scheduler.T_0,
                                                     T_mult=scheduler.T_mult)

    def _train_one_epoch(self, epoch):
        self.model.train()
        with tqdm(self.train_loader, desc="Training", disable=not self.accelerator.is_local_main_process) as tl:
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tl):
                with self.accelerator.accumulate(self.model):
                    loss = 0
                    log = ''
                    logits = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
                    for loss_name, loss_fn in self.loss_functions.items():
                        loss_single = loss_fn(logits, labels)
                        loss += loss_single
                        log += f" | {loss_name}: {loss_single}"
                        self.accelerator.log({f'Train/{loss_name}': float(loss), }, step=self.train_step)
                    log = f"total_loss: {loss}" + log
                    tl.set_postfix_str(log)

                    predictions, references = self.accelerator.gather_for_metrics(
                        (logits.argmax(dim=-1), labels)
                    )

                    # Backpropagation & gradient update
                    self.metric.add_batch(predictions=predictions, references=references)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # training log
                    self.accelerator.log({'Train/Total Loss': float(loss), }, step=self.train_step)
                    self.logger.log(15, 'Epoch [{}/{}] Step [{}/{}]\tTraining: {}'
                                    .format(epoch + 1, self.epochs, i + 1, len(self.train_loader), log))
                    self.train_step += 1
                    tl.update()
        metric = self.metric.compute()
        self.logger.info(f'Epoch [{epoch + 1}/{self.epochs}] Training metric {metric}')
        for m_name, m_value in metric.items():
            self.accelerator.log({f'Train Metric/{m_name}': m_value}, step=epoch)

    @torch.no_grad()
    def _eval_one_epoch(self, epoch):
        self.model.eval()
        with tqdm(self.eval_loader, desc="Evaluating", disable=not self.accelerator.is_local_main_process) as el:
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(el):
                loss = 0
                log = ''
                logits = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
                for loss_name, loss_fn in self.loss_functions.items():
                    loss_single = loss_fn(logits, labels)
                    loss += loss_single
                    log += f" | {loss_name}: {loss_single}"
                    self.accelerator.log({f'Val/{loss_name}': float(loss), }, step=self.val_step)
                log = f"total_loss: {loss}" + log
                el.set_postfix_str(log)

                predictions, references = self.accelerator.gather_for_metrics(
                    (logits.argmax(dim=-1), labels)
                )
                self.metric.add_batch(predictions=predictions, references=references)
                self.accelerator.log({'Val/Total Loss': float(loss), }, step=self.val_step)
                self.logger.log(15, 'Epoch [{}/{}] Step [{}/{}]\tValidation: {}'
                                .format(epoch + 1, self.epochs, i + 1, len(self.eval_loader), log))
                self.val_step += 1
                el.update()
        metric = self.metric.compute()
        self.logger.info(f'Epoch [{epoch + 1}/{self.epochs}] Validation metric {metric}')
        for m_name, m_value in metric.items():
            self.accelerator.log({f'Eval Metric/{m_name}': m_value}, step=epoch)
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

        # try to restore the train step
        if self.restore:
            self.logger.info(f"Loading checkpoint from path: {self.check_point}, attempting to resume training...")
            self.starting_epoch, self.train_step, self.val_step = resume_train_state(
                self.accelerator, self.check_point, self.train_loader, self.eval_loader, self.logger
            )
        else:
            self.logger.info(f"Preparing to start a new training, checkpoint path: {self.check_point}")
        self.logger.info("Start training!")
        best_acc = 0
        for epoch in range(self.starting_epoch, self.epochs):
            # train model of one epoch
            self._train_one_epoch(epoch=epoch)

            # evaluate model for one epoch
            if epoch % self.eval_step == 0 or epoch + 1 == self.epochs:
                mean_acc = self._eval_one_epoch(epoch=epoch)
                self.accelerator.log({'lr': self.scheduler.get_last_lr()}, step=epoch)

                # save the best model
                if best_acc < mean_acc:
                    best_acc = mean_acc
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        f'{os.path.join(os.getcwd(), self.check_point, "best")}')

                # save model
                if self.save_all_epoch:
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        f'{os.path.join(os.getcwd(), self.check_point, f"epoch_{epoch + 1}")}')
                    # traditional function for model save (nn.Module)
                    # self.accelerator.save_state(
                    #     output_dir=f'{os.path.join(os.getcwd(), self.check_point, f"epoch_{epoch + 1}")}'
                    # )
                self.logger.info(
                    "--------- Epoch [{}/{}] Verification results | mean acc :{} | best acc:{} | lr = {} ---------\n"
                    .format(epoch + 1, self.epochs, mean_acc, best_acc, self.scheduler.get_last_lr())
                )
        self.logger.info(f"---------- End of all training ----------")
        self.logger.info(f"Best acc: {best_acc}")
        exit(1)

    def test(self):
        self.accelerator_control()

        # load the best model
        self.model.load_state_dict(torch.load(
            os.path.join(os.getcwd(), self.check_point, "best", "pytorch_model.bin")
        ))

        # start test
        self.logger.info("Start testing")
