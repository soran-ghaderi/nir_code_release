import random

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset

import configs
from configs import MAX_LENGTH, SUBSET_SIZE

from utils import set_seed, logger
# from rich import print


logger = logger()


class MathDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        max_length=MAX_LENGTH,
        subset_size=SUBSET_SIZE,
        seed=42,
    ):

        self.dataset = load_dataset("hendrycks/competition_math", split=split)
        if subset_size is not None:
            self.dataset = self.dataset.select(
                range(min(subset_size, len(self.dataset)))
            )
        self.tokenizer = tokenizer
        self.max_length = max_length
        # tokenizer.pad_token = tokenizer.eos_token
        set_seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            question_type = item.get("type", "")
            input_text = f"Problem: {problem}\nSolution: {solution}"

            # encoded = self.tokenizer.encode_plus(
            #     input_text,
            #     max_length=self.max_length,
            #     padding="max_length",
            #     truncation=True,
            #     return_tensors="pt",
            # )
            print(f"dataset input {idx}: ", len(input_text), input_text, "\n\n")

            encoded = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
            }
        except Exception as e:
            print(f"Error processing item at index {idx}: {str(e)}")
            # Return a dummy item in case of error
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            }


class GSM8KDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        max_length=configs.MAX_LENGTH,
        subset_size=configs.SUBSET_SIZE,
        seed=42,
    ):
        self.dataset = load_dataset("gsm8k", "main", split=split)
        if subset_size is not None:
            set_seed(seed)
            self.dataset = self.dataset.select(
                random.sample(
                    range(len(self.dataset)), min(subset_size, len(self.dataset))
                )
            )
        self.tokenizer = tokenizer
        self.max_length = max_length
        set_seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            question = item.get("question", "")
            answer = item.get("answer", "")
            input_text = f"Problem: {question}\nSolution: {answer}"
            # input_text = f"Solution: {answer}"
            print(
                idx, "input_text: ", f"Problem: {question}\nSolution: {answer}", "\n\n"
            )
            encoded = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            logger.debug(f"Processed item {idx}: {input_text[:100]}...")

            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
            }
        except Exception as e:
            logger.error(f"Error processing item at index {idx}: {str(e)}")
            # Return a dummy item in case of error
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            }


class MBPPDataset(Dataset):
    def __init__(self, dataset, num_examples=-1):
        self.dataset = dataset
        self.num_examples = num_examples if num_examples != -1 else len(dataset)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples:
            raise IndexError("Index out of bounds")

        instance = self.dataset[idx]
        print(f"Accessing index {idx}, instance keys: {instance.keys()}")

        return {
            "id": idx,
            "query": self._get_first_or_default(instance, "query", ""),
            "context": self._get_first_or_default(instance, "context", ""),
            "input_final_prompts": self._get_first_or_default(
                instance, "input_final_prompts", ""
            ),
        }

    @staticmethod
    def _get_first_or_default(instance, key, default):
        value = instance.get(key, default)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return value[0]
        return value
