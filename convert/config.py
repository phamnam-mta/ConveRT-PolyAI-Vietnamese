from typing import NamedTuple


class ConveRTModelConfig(NamedTuple):
    num_embed_hidden: int = 300
    feed_forward1_hidden: int = 2048
    feed_forward2_hidden: int = 1024
    num_attention_project: int = 64
    vocab_size: int = 10000
    num_encoder_layers: int = 6
    dropout_rate: float = 0.1


class ConveRTTrainConfig(NamedTuple):
    data_dir: str = "data"
    sp_model_path: str = "data/vi.wiki.bpe.vs10000.model"
    train_dataset_path: str = "data/train.json"
    test_dataset_path: str = "data/test.json"

    model_save_dir: str = "models/convert"
    log_dir: str = "logs"
    device: str = "cuda:0"
    use_data_paraller: bool = False

    is_reddit: bool = True
    is_pretrain_embed: bool = True

    train_batch_size: int = 64
    test_batch_size: int = 128

    split_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 10
