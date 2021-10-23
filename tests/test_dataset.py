import pytest
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader

from convert.config import ConveRTTrainConfig
from convert.dataset import ConveRTDataset, convert_collate_fn, load_instances_from_reddit_dataset

REDDIT_SAMPLE_DATA = {
    "context_author": "Needs_Mega_Magikarp",
    "context/8": "*She giggles at his giggle.* Yay~",
    "context/5": "*He rests his head on yours.*\n\nYou aaaaare. You're the cutest.",
    "context/4": "Pfffft. *She playfully pokes his stomach.* Shuddup.",
    "context/7": "*He hugs you.*\n\nOhmigooods, you're so cute.",
    "context/6": "*She giggles again.* No I'm noooot.",
    "context/1": "*He snorts a laugh*\n\nD'aww. Cute.",
    "context/0": "Meanie.",
    "response_author": "Ironic_Remorse",
    "subreddit": "PercyJacksonRP",
    "thread_id": "2vcitx",
    "context/3": "*He shrugs.*\n\nBut I dun wanna lie!",
    "context": "Cutie.\n\n*He jokes, rubbing your arm again. Vote Craig for best brother 2k15.*",
    "context/2": "*She sticks her tongue out.*",
    "response": "Meanieee. *She pouts.*",
}


@pytest.fixture
def tokenizer() -> SentencePieceProcessor:
    config = ConveRTTrainConfig()
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(config.sp_model_path)
    return tokenizer


def test_load_reddit_instances():
    instances = load_instances_from_reddit_dataset("data/sample-dataset.json")
    assert len(instances) == 1000


def test_encoding_using_sp_model(tokenizer: SentencePieceProcessor):
    assert tokenizer.EncodeAsIds("welcome home") == [3441, 4984, 1004]


def test_dataset_get_item(tokenizer: SentencePieceProcessor):
    instances = load_instances_from_reddit_dataset("data/sample-dataset.json")
    dataset = ConveRTDataset(instances, tokenizer)
    assert len(dataset) == 1000


def test_dataset_batching(tokenizer: SentencePieceProcessor):
    instances = load_instances_from_reddit_dataset("data/sample-dataset.json")
    dataset = ConveRTDataset(instances, tokenizer)
    data_loader = DataLoader(dataset, batch_size=3, collate_fn=convert_collate_fn)

    for batch in data_loader:
        print(batch)
