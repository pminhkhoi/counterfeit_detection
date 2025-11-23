import pandas as pd
from torch.utils.data import Dataset
import torch


class CSVDataset(Dataset):
    def __init__(self, data, text_col, label_col=None, tokenizer=None, max_length=256):
        """
        data: path to CSV file OR a pandas DataFrame
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Data must be a path (str) or a pandas DataFrame")

        self.text_col = text_col
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure text is string
        text = str(self.data.iloc[idx][self.text_col])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.label_col and self.label_col in self.data.columns:
            item["label"] = torch.tensor(int(self.data.iloc[idx][self.label_col]), dtype=torch.long)

        return item