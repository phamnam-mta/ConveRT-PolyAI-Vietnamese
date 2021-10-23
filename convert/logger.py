import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class TrainLogger(logging.Logger):
    def __init__(self, logger_name: str, tensorboard: SummaryWriter):
        self.tensorboard: SummaryWriter = tensorboard
        self.global_train_step = 0
        self.global_eval_step = 0
        super().__init__(logger_name)

    def log_train_step(self, epoch_id: int, step_id: int, eta: float, loss: torch.Tensor, accuracy: float):
        self.info(f"EP:{epoch_id}\tSTEP:{step_id}\t" f"loss:{loss}\tacc:{accuracy}\t" f"eta:{eta//60} min {eta%60} sec")
        self.tensorboard.add_scalar("train/acc", accuracy, global_step=self.global_train_step)
        self.tensorboard.add_scalar("train/loss", loss.item(), global_step=self.global_train_step)
        self.global_train_step += 1

    def log_eval_step(self, epoch_id: int, loss: float, accuracy: float):
        self.info(f"[EVAL] EP:{epoch_id}\t loss:{loss:.4f} eval acc: {accuracy:.4f}")
        self.tensorboard.add_scalar("eval/acc", accuracy, global_step=self.global_eval_step)
        self.tensorboard.add_scalar("eval/loss", loss, global_step=self.global_eval_step)
        self.global_eval_step += 1
