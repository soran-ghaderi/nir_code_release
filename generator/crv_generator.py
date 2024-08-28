from typing import Union, Optional, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from configs import SUBSET_SIZE
from data_processor.data_loader import GSM8KDataset
from utils import logger

logger = logger()


class CRVGenerator:
    def __init__(self, model, tokenizer, max_length=512, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.logger = logger
        self.max_length = max_length

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_crvs(
        self,
        inputs: Union[str, Dataset],
        output_file: str = "data/new_stack.pt",
        crv_layers: Optional[Union[int, List[int]]] = None,
        batch_size: int = 32,
        num_contexts: int = 1000,
        subset_size: int = None,
        crv_save_batch: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.set_seed(self.seed)

        if inputs == "dataset" or isinstance(inputs, Dataset):
            return self._generate_crvs_from_dataset(
                output_file,
                crv_layers,
                batch_size,
                num_contexts,
                subset_size,
                crv_save_batch,
                inputs,
            )
        elif isinstance(inputs, str):
            return self._generate_crvs_from_query(inputs, output_file, crv_layers)
        else:
            raise ValueError("Input must be either 'dataset' or a query string")

    def _generate_crvs_from_dataset(
        self,
        output_file,
        crv_layers,
        batch_size,
        num_contexts,
        subset_size,
        crv_save_batch,
        dataset=None,
    ):
        if dataset is None:
            self.logger.info("'dataset' is None. Load GSM8K dataset.")
            dataset = GSM8KDataset(
                self.tokenizer, split="train", subset_size=subset_size
            )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        crvs = []
        seq_lengths = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_contexts // batch_size:
                    break

                inputs = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                outputs = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                prompt_stacked_crv = self._process_hidden_states(
                    outputs.hidden_states, crv_layers
                )
                crvs.append(prompt_stacked_crv)

                # Calculate and store the original sequence lengths
                batch_seq_lengths = attention_mask.sum(dim=1)
                seq_lengths.append(batch_seq_lengths)

                if (batch_idx + 1) % crv_save_batch == 0:
                    self.logger.info(
                        f"Processed {(batch_idx + 1) * batch_size} contexts"
                    )

        crvs_tensor = torch.cat(crvs, dim=0)  # (layers, b, seq_len, dim)
        crvs_tensor = crvs_tensor.transpose(0, 1)  # (b, layers, seq_len, dim)

        seq_lengths_tensor = torch.cat(seq_lengths, dim=0)  # (b,)

        torch.save(
            {"crvs": crvs_tensor, "seq_lengths": seq_lengths_tensor}, output_file
        )
        self.logger.info(f"CRVs and sequence lengths saved to {output_file}")
        return crvs_tensor, seq_lengths_tensor

    def _generate_crvs_from_query(self, inputs, output_file, crv_layers):
        self.logger.info("The input received is a query")
        with torch.no_grad():
            encoded_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = encoded_inputs["input_ids"].to(self.model.device)
            attention_mask = encoded_inputs["attention_mask"].to(self.model.device)
            outputs = self.model(
                inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            prompt_stacked_crv = self._process_hidden_states(
                outputs.hidden_states, crv_layers
            )
            seq_length = attention_mask.sum(dim=1)

        return prompt_stacked_crv, seq_length

    @staticmethod
    def _process_hidden_states(hidden_states, crv_layers):
        if crv_layers is None:
            prompt_stacked_crv = torch.stack(
                [layer for layer in hidden_states], dim=0
            ).squeeze(1)
        elif isinstance(crv_layers, list):
            prompt_stacked_crv = torch.stack(
                [layer for idx, layer in enumerate(hidden_states) if idx in crv_layers],
                dim=0,
            ).squeeze(1)
        elif isinstance(crv_layers, int):
            prompt_stacked_crv = hidden_states[crv_layers].squeeze(1)
        else:
            raise ValueError(
                "crv_layers must be None, a list of integers, or an integer"
            )
        return prompt_stacked_crv


class CRVGenerator1:
    def __init__(self, model, tokenizer, max_length=512, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.logger = logger
        self.max_length = max_length

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_crvs(
        self,
        inputs: Union[str, Dataset],
        output_file: str = "data/new_stack.pt",
        crv_layers: Optional[Union[int, List[int]]] = None,
        batch_size: int = 32,
        num_contexts: int = 1000,
        subset_size: int = SUBSET_SIZE,
        crv_save_batch: int = 10,
    ) -> torch.Tensor:
        self.set_seed(self.seed)

        if inputs == "dataset" or isinstance(inputs, Dataset):
            # dataset = GSM8KDataset(
            #     self.tokenizer, split="train", subset_size=subset_size
            # )

            return self._generate_crvs_from_dataset(
                output_file,
                crv_layers,
                batch_size,
                num_contexts,
                subset_size,
                crv_save_batch,
                inputs,
            )
        elif isinstance(inputs, str):
            return self._generate_crvs_from_query(inputs, output_file, crv_layers)
        else:
            raise ValueError("Input must be either 'dataset' or a query string")

    def _generate_crvs_from_dataset(
        self,
        output_file,
        crv_layers,
        batch_size,
        num_contexts,
        subset_size,
        crv_save_batch,
        dataset=None,
    ):
        print("is dataset? ", isinstance(dataset, Dataset))
        if dataset is None:
            logger.info("'dataset' is None. Load GSM8K dataset.")
            dataset = GSM8KDataset(
                self.tokenizer, split="train", subset_size=subset_size
            )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        crvs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_contexts // batch_size:
                    break

                inputs = batch["input_ids"].to(self.model.device)
                outputs = self.model(
                    inputs, output_hidden_states=True, return_dict=True
                )

                prompt_stacked_crv = self._process_hidden_states(
                    outputs.hidden_states, crv_layers
                )
                crvs.append(prompt_stacked_crv)

                if (batch_idx + 1) % crv_save_batch == 0:
                    self.logger.info(
                        f"Processed {(batch_idx + 1) * batch_size} contexts"
                    )

        crvs_tensor = torch.cat(crvs, dim=0)  # (layers, b, seq_len, dim)
        crvs_tensor = crvs_tensor.transpose(0, 1)  # (b, layers, seq_len, dim)

        torch.save(crvs_tensor, output_file)
        self.logger.info(f"CRVs saved to {output_file}")
        return crvs_tensor

    def _generate_crvs_from_query(self, inputs, output_file, crv_layers):
        self.logger.info("The input received is a query")
        with torch.no_grad():
            inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = inputs["input_ids"].to(self.model.device)
            outputs = self.model(inputs, output_hidden_states=True, return_dict=True)

            prompt_stacked_crv = self._process_hidden_states(
                outputs.hidden_states, crv_layers
            )
        return prompt_stacked_crv

    @staticmethod
    def _process_hidden_states(hidden_states, crv_layers):
        if crv_layers is None:
            prompt_stacked_crv = torch.stack(
                [layer for layer in hidden_states], dim=0
            ).squeeze(
                1
            )  # (layers, seq_len, d_model)
            return prompt_stacked_crv
            # return torch.stack(hidden_states, dim=0).squeeze(1)
        elif isinstance(crv_layers, list):
            prompt_stacked_crv = torch.stack(
                [layer for idx, layer in enumerate(hidden_states) if idx in crv_layers],
                dim=0,
            ).squeeze(
                1
            )  # (len(crv_layers), seq_len, d_model)
            return prompt_stacked_crv
            # return torch.stack(
            #     [hidden_states[idx] for idx in crv_layers], dim=0
            # ).squeeze(1)
        elif isinstance(crv_layers, int):
            prompt_stacked_crv = hidden_states[crv_layers].squeeze(
                1
            )  # (1, seq_len, d_model), the 1 is the len(crv_layers)
            return prompt_stacked_crv
            # return hidden_states[crv_layers].squeeze(1)
        else:
            raise ValueError(
                "crv_layers must be None, a list of integers, or an integer"
            )
