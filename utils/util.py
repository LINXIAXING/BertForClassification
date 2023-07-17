"""
====================================================
@Project:   MyBert -> util
@Author:    TropicalAlgae
@Date:      2023/6/23 21:13
@Desc:
====================================================
"""

import logging
import accelerate
import torch
import os


class MetricSaver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.best_acc = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


def resume_train_state(path: str, train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader,
                       logger: logging.Logger, accelerator: accelerate.Accelerator):
    try:
        # Get the most recent checkpoint
        base_path = os.getcwd() + '/' + path
        dirs = [base_path + '/' + f.name for f in os.scandir(base_path) if
                (f.is_dir() and f.name.startswith('epoch_'))]
        dirs.sort(key=os.path.getctime)  # Sorts folders by date modified, most recent checkpoint is the last
        logger.info(f'Try to load epoch {dirs[-1]} train state')
        accelerator.load_state(dirs[-1])
        training_difference = os.path.splitext(dirs[-1])[0]
        starting_epoch = int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        if val_loader is not None:
            val_step = starting_epoch * len(val_loader)
        else:
            val_step = 0
        logger.info(f'Load train state success! Start from epoch {starting_epoch}')
        return starting_epoch, step, val_step
    except Exception as e:
        logger.error(e)
        logger.error(f'Load train state fail!')
        return 0, 0, 0


