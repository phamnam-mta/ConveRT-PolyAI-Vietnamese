import json
from dataclasses import dataclass
from typing import List, NamedTuple

import torch
from sentencepiece import SentencePieceProcessor
from torch.nn.functional import pad
from torch.utils.data import Dataset

INPUT_ATTRIBUTES = ["input_ids", "attention_mask", "position_ids", "input_lengths"]


class DatasetInstance(NamedTuple):
    context: List[str]
    response: str


@dataclass
class EncoderInputFeature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    input_lengths: torch.Tensor

    def pad_sequence(self, seq_len: int):
        self.input_ids = pad(self.input_ids, [0, seq_len - self.input_ids.size(0)], "constant", 0)
        self.attention_mask = pad(self.attention_mask, [0, seq_len - self.attention_mask.size(0)], "constant", 0)
        self.position_ids = pad(self.position_ids, [0, seq_len - self.position_ids.size(0)], "constant", 0)

    def to(self, device: torch.device) -> "EncoderInputFeature":
        return EncoderInputFeature(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device),
            input_lengths=self.input_lengths.to(device),
        )


@dataclass
class ContextReplyFeaturePair:
    context: EncoderInputFeature
    reply: EncoderInputFeature

    def to(self, device: torch.device) -> "ContextReplyFeaturePair":
        return ContextReplyFeaturePair(context=self.context.to(device), reply=self.reply.to(device))


class ConveRTDataset(Dataset):
    def __init__(self, instances: List[DatasetInstance], sp_processor: SentencePieceProcessor):
        super().__init__()
        self.instances: List[DatasetInstance] = instances
        self.sp_processor: SentencePieceProcessor = sp_processor

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, item) -> ContextReplyFeaturePair:
        context_str = " ".join(self.instances[item].context)
        reply_str = " ".join(self.instances[item].response)
        context_input = self._convert_instance_to_feature(context_str)
        reply_input = self._convert_instance_to_feature(reply_str)
        return ContextReplyFeaturePair(context=context_input, reply=reply_input)

    def _convert_instance_to_feature(self, input_str: str) -> EncoderInputFeature:
        input_ids = self.sp_processor.EncodeAsIds(input_str)
        attention_mask = [1 for _ in range(len(input_ids))]
        position_ids = [i for i in range(len(input_ids))]

        return EncoderInputFeature(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
            position_ids=torch.tensor(position_ids),
            input_lengths=torch.tensor(len(input_ids)),
        )


def batching_input_features(encoder_inputs: List[EncoderInputFeature]) -> EncoderInputFeature:
    max_seq_len = max([int(encoder_input.input_lengths.item()) for encoder_input in encoder_inputs])
    for encoder_input in encoder_inputs:
        encoder_input.pad_sequence(max_seq_len)

    batch_features = {
        feature_name: torch.stack([getattr(encoder_input, feature_name) for encoder_input in encoder_inputs], dim=0)
        for feature_name in INPUT_ATTRIBUTES
    }
    return EncoderInputFeature(**batch_features)


def convert_collate_fn(features: List[ContextReplyFeaturePair]) -> ContextReplyFeaturePair:
    return ContextReplyFeaturePair(
        context=batching_input_features([feature.context for feature in features]),
        reply=batching_input_features([feature.reply for feature in features]),
    )


def load_instances_from_reddit_dataset(dataset_path: str) -> List[DatasetInstance]:
    instances: List[DatasetInstance] = []
    # dataset_file = open(dataset_path)
    # for line in dataset_file:
    #     example = json.loads(line)
    #     context_keys = sorted([key for key in example.keys() if "context" in key])
    #     instance = DatasetInstance(context=[example[key] for key in context_keys], response=example["response"],)
    #     instances.append(instance)
    with open(dataset_path, "r") as dataset_file:
        data = json.load(dataset_file)
        for d in data:
            thread_name = d.get("thread_name")
            quotes = d.get("quotes")
            replies = d.get("replies")
            if quotes:
                instance = DatasetInstance(context=quotes, response=replies)
            else:
                instance = DatasetInstance(context=[thread_name], response=replies)
            instances.append(instance)
    return instances


def load_instances_from_tsv_dataset(dataset_path: str) -> List[DatasetInstance]:
    instances: List[DatasetInstance] = []
    dataset_file = open(dataset_path)
    for line in dataset_file:
        splited_lines = line.strip().split("\t")
        instance = DatasetInstance(context=splited_lines[:-1], response=splited_lines[-1])
        instances.append(instance)
    dataset_file.close()
    return instances
