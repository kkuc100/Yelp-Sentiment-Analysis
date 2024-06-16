import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer


def collate_fn(
    data, feature_col, label_col, tokenizer, max_length, device
):
    features = [i[feature_col] for i in data]
    labels = torch.tensor([i[label_col] for i in data]).to(device)
    encoded_features = tokenizer.batch_encode_plus(
        features,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encoded_labels = tokenizer.batch_encode_plus(
        [str(i[label_col]) for i in data],
        truncation=True,
        padding="max_length",
        max_length=2,
        return_tensors="pt",
    ).input_ids
    encoded_features = encoded_features.to(device)
    encoded_labels = encoded_labels.to(device)
    return features, encoded_features, labels, encoded_labels


class YelpDataset:
    def __init__(
        self,
        trainset_path: str = None,
        testset_path: str = None,
        feature_col: str = None,
        label_col: str = None,
        tokenizer_name: str = None,
        max_length: int = 400,
        batch_size: int = 32,
        test_size: float = 0.2,
        seed: int = 3407,
        device: str = "cuda",
        stratify_col: str = None,
    ) -> None:

        self.train_path = trainset_path
        self.valid_path = testset_path
        self.feature_col = feature_col
        self.label_col = label_col
        self.max_length = max_length
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed
        self.device = device
        self.stratify_col = stratify_col

        if "t5" in tokenizer_name:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.setup()

    def setup(self):
        # import the parquet
        self.train_df = pd.read_parquet(self.train_path).reset_index(drop=True)
        self.valid_df = pd.read_parquet(self.valid_path).reset_index(drop=True)

        self.train_dataset = Dataset.from_pandas(self.train_df, preserve_index=True)
        self.valid_dataset = Dataset.from_pandas(self.valid_df, preserve_index=True)

        self.train_dataset = self.train_dataset.class_encode_column(self.label_col)
        self.valid_dataset = self.valid_dataset.class_encode_column(self.label_col)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda data: collate_fn(
                data=data,
                feature_col=self.feature_col,
                label_col=self.label_col,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=self.device,
            ),
            shuffle=True,
            drop_last=True,
        )

    def valid_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda data: collate_fn(
                data=data,
                feature_col=self.feature_col,
                label_col=self.label_col,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                device=self.device,
            ),
        )

