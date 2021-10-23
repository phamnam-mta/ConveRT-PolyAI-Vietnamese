import pytest
import torch
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from convert.config import ConveRTModelConfig, ConveRTTrainConfig
from convert.criterion import ConveRTCosineLoss
from convert.dataset import ConveRTDataset, convert_collate_fn, load_instances_from_reddit_dataset
from convert.logger import TrainLogger
from convert.model import ConveRTDualEncoder
from convert.trainer import ConveRTTrainer


@pytest.fixture
def train_dataloader():
    config = ConveRTTrainConfig(train_batch_size=10, split_size=5)
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(config.sp_model_path)

    instances = load_instances_from_reddit_dataset("data/sample-dataset.json")[:100]
    dataset = ConveRTDataset(instances, tokenizer)
    data_loader = DataLoader(dataset, batch_size=config.train_batch_size, collate_fn=convert_collate_fn)
    return data_loader


@pytest.fixture
def convert_model():
    # small model config for testing
    model_config = ConveRTModelConfig(
        num_embed_hidden=32,
        feed_forward1_hidden=64,
        feed_forward2_hidden=32,
        num_attention_project=16,
        vocab_size=10000,
        num_encoder_layers=2,
        dropout_rate=0.1,
    )
    return ConveRTDualEncoder(model_config)


def test_init_trainer(convert_model: ConveRTDualEncoder, train_dataloader: DataLoader):
    train_config = ConveRTTrainConfig(train_batch_size=10, split_size=5)
    tensorboard = SummaryWriter()
    logger = TrainLogger("test-logger", tensorboard)
    device = torch.device("cpu")

    criterion = ConveRTCosineLoss(split_size=train_config.split_size)
    trainer = ConveRTTrainer(
        model=convert_model,
        criterion=criterion,
        train_config=train_config,
        train_dataloader=train_dataloader,
        test_dataloader=train_dataloader,
        logger=logger,
        device=device,
    )
    assert trainer is not None


def test_train_one_epoch(convert_model: ConveRTDualEncoder, train_dataloader: DataLoader):
    train_config = ConveRTTrainConfig(train_batch_size=10, split_size=5)
    tensorboard = SummaryWriter()
    logger = TrainLogger("test-logger", tensorboard)
    device = torch.device("cpu")

    criterion = ConveRTCosineLoss(split_size=train_config.split_size)
    trainer = ConveRTTrainer(
        model=convert_model,
        criterion=criterion,
        train_config=train_config,
        train_dataloader=train_dataloader,
        test_dataloader=train_dataloader,
        logger=logger,
        device=device,
    )
    trainer.train()
