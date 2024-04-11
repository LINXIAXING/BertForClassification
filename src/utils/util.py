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


def resume_train_state(accelerator: accelerate.Accelerator, path: str, train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader, logger: logging.Logger, ):
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


def get_checkpoint_path(root: str, describe: str, restore_train: bool) -> (str, bool):
    is_new = True

    if not os.path.isdir(root):
        check_point_root = os.path.join(root, f"{describe}_v1")
    else:
        folders = [os.path.join(root, d) for d in os.listdir(root) if describe == "_v".join(d.split("_v")[:-1])]
        if restore_train and len(folders) > 0:
            folders.sort(key=lambda x: x.split("_v")[-1])
            check_point_root = folders[-1]
            is_new = False
        else:
            train_id = max([int(x.split("_v")[-1]) for x in folders] + [0])
            check_point_root = os.path.join(root, f"{describe}_v{train_id + 1}")
    return check_point_root, is_new


def get_newest_checkpoint_path(root: str, describe: str) -> str:
    folders = [os.path.join(root, d) for d in os.listdir(root) if describe == "_v".join(d.split("_v")[:-1])]
    if len(folders) > 0:
        folders.sort(key=lambda x: x.split("_v")[-1])
        return folders[-1]
    else:
        return ""


def build_folder(*args):
    path = os.path.join(*args)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path
