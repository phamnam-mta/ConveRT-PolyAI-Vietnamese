import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from convert.config import ConveRTModelConfig, ConveRTTrainConfig
from convert.criterion import ConveRTCosineLoss
from convert.dataset import (
    ConveRTDataset,
    convert_collate_fn,
    load_instances_from_reddit_dataset,
    load_instances_from_tsv_dataset,
)
from convert.logger import TrainLogger
from convert.model import ConveRTDualEncoder
from convert.trainer import ConveRTTrainer
from bpemb import BPEmb


def get_train_config() -> ConveRTTrainConfig:
    default_train_config = ConveRTTrainConfig()

    argparser = ArgumentParser()
    for config_key, config_value in default_train_config._asdict().items():
        argparser.add_argument(f"--{config_key}", default=config_value, type=type(config_value))
    args = argparser.parse_args()

    return ConveRTTrainConfig(**vars(args))


def logger_setup(log_dir: str) -> TrainLogger:
    date_time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_logdir = os.path.join(log_dir, f"convert.tensorboard.{date_time_string}")
    tensorboard = SummaryWriter(log_dir=tensorboard_logdir, flush_secs=60)

    logger = TrainLogger("convert-trainer", tensorboard)
    formatter = logging.Formatter("%(asctime)s\t%(message)s")

    log_output_path = os.path.join(log_dir, f"convert.train.{date_time_string}.logs")
    file_handler = logging.FileHandler(log_output_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)
    
    return logger


def main() -> int:
    train_config = get_train_config()
    model_config = ConveRTModelConfig()

    logger = logger_setup(train_config.log_dir)
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")

    tokenizer = SentencePieceProcessor()
    tokenizer.Load(train_config.sp_model_path)

    instance_load_fn = load_instances_from_reddit_dataset if train_config.is_reddit else load_instances_from_tsv_dataset
    train_instances = instance_load_fn(train_config.train_dataset_path)
    test_instances = instance_load_fn(train_config.test_dataset_path)

    pretrain_embed=None
    if train_config.is_pretrain_embed:
        pretrain_embed = BPEmb(lang="vi", vs=model_config.vocab_size, dim=model_config.num_embed_hidden, cache_dir=train_config.data_dir)
    train_dataset = ConveRTDataset(train_instances, tokenizer, pretrain_embed)
    test_dataset = ConveRTDataset(test_instances, tokenizer, pretrain_embed)
    train_dataloader = DataLoader(
        train_dataset, train_config.train_batch_size, collate_fn=convert_collate_fn, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, train_config.test_batch_size, collate_fn=convert_collate_fn, drop_last=True
    )

    model = ConveRTDualEncoder(model_config)
    criterion = ConveRTCosineLoss(split_size=train_config.split_size)

    model.to(device)
    criterion.to(device)

    if train_config.use_data_paraller and torch.cuda.is_available():
        model = nn.DataParallel(model)
        criterion = nn.DataParallel(criterion)

    # model.load_state_dict(torch.load(train_config.model_save_dir))
    # model.train()

    trainer = ConveRTTrainer(
        model=model,
        criterion=criterion,
        train_config=train_config,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        logger=logger,
        device=device,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
