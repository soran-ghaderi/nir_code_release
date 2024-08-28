import io
from typing import Union, List, Dict, Tuple

import torch
from sympy.polys.polyconfig import query
from torch.nn import functional as F

from generator.crv_generator import CRVGenerator
from utils import set_seed, logger

logger = logger()


class CRVRetriever:
    def __init__(self, model, tokenizer, crv_layers, seed=42, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.crv_layers = crv_layers
        self.seed = seed
        self.set_seed()
        self.max_length = max_length
        self.crv_generator = CRVGenerator(model, tokenizer, max_length=max_length)
        self.logger = logger

    def set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def load_crvs(
        self,
        crvs_file: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(
            crvs_file, str
        ):  # if it's a file path todo: may not work as expected
            print("crvs_file type: ", type(crvs_file))
            data = torch.load(crvs_file)
            return data[0], data[1]  # data["crvs"], data["seq_lengths"]
        elif (
            isinstance(crvs_file, dict)
            and "crvs" in crvs_file
            and "seq_lengths" in crvs_file
        ):
            return crvs_file["crvs"], crvs_file["seq_lengths"]

        if isinstance(crvs_file, tuple):
            print("crvs_file type: ", type(crvs_file))
            return crvs_file[0], crvs_file[1]  # data["crvs"], data["seq_lengths"]
        else:
            raise ValueError(
                f"crvs_file must be either a string (file path) or a dictionary with 'crvs' and 'seq_lengths' keys. "
                f"Type detected: {type(crvs_file)}"
            )

    def generate_crvs(
        self, inputs, output_file="data/new_stack.pt"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_crv, query_seq_length = self.crv_generator.generate_crvs(
            inputs, crv_layers=self.crv_layers
        )
        self.logger.info(f"Generated query CRV shape: {query_crv.shape}")
        self.logger.info(f"Generated query sequence length: {query_seq_length}")
        return query_crv, query_seq_length

    def compute_similarities(
        self, query_crv: torch.Tensor, crvs: torch.Tensor
    ) -> torch.Tensor:
        self.logger.info(f"Query CRV shape: {query_crv.shape}")
        self.logger.info(f"CRVs shape: {crvs.shape}")
        query_crv_last = query_crv[0]  # shape: (seq_len, d_model)
        crvs_last = crvs[:, 1, :, :]  # shape: (b, seq_len, d_model)
        crvs_last = crvs_last.to(query_crv_last.device)
        query_crv_norm = F.normalize(query_crv_last.reshape(1, -1), p=2, dim=1)
        crvs_norm = F.normalize(crvs_last.reshape(crvs_last.shape[0], -1), p=2, dim=1)
        self.logger.info(f"Query norm CRV shape: {query_crv_norm.shape}")
        self.logger.info(f"CRVs norm shape: {crvs_norm.shape}")
        return torch.cosine_similarity(query_crv_norm, crvs_norm, -1)

    def get_top_k_indices(self, similarities: torch.Tensor, k: int = 5) -> torch.Tensor:
        return torch.topk(similarities, k).indices

    def retrieve_best_crv(
        self, query: str, crvs_file: Union[str, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        crvs, seq_lengths = self.load_crvs(crvs_file)
        self.logger.info(f"Loaded CRVs shape: {crvs.shape}")
        self.logger.info(f"Loaded sequence lengths shape: {seq_lengths.shape}")

        query_crv, query_seq_length = self.generate_crvs(
            inputs=query, output_file="data/new_stack.pt"
        )
        similarities = self.compute_similarities(query_crv, crvs)

        if torch.isnan(similarities).any() or torch.isinf(similarities).any():
            self.logger.warning("Warning: NaN or Inf values in similarities")

        self.logger.info(f"Max similarity: {similarities.max().item()}")
        self.logger.info(f"Min similarity: {similarities.min().item()}")
        self.logger.info(f"Mean similarity: {similarities.mean().item()}")

        best_indices = self.get_top_k_indices(similarities)
        self.logger.info(f"Top 5 indices: {best_indices}")
        best_crv_index = best_indices[0].item()
        self.logger.info(f"Best CRV index: {best_crv_index}")

        return crvs[best_crv_index], seq_lengths[best_crv_index]

    def __call__(
        self, query: str, crvs_file: Union[str, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.retrieve_best_crv(query, crvs_file)


class CRVRetriever1:
    def __init__(self, model, tokenizer, crv_layers, seed=42, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.crv_layers = crv_layers
        self.seed = seed
        self.set_seed()
        self.max_length = max_length
        self.crv_generator = CRVGenerator(model, tokenizer, max_length=max_length)

    def set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def load_crvs(self, crvs_file):
        if isinstance(crvs_file, str):
            return torch.load(crvs_file)
        elif isinstance(crvs_file, torch.Tensor):
            logger.info("CRVs is an instance of torch.Tensor.")
            return crvs_file
        else:
            raise ValueError(
                "crvs_file must be either a string (file path) or a torch.Tensor"
            )

    def generate_crvs(self, inputs, output_file="data/new_stack.pt"):

        query_crv = self.crv_generator.generate_crvs(
            inputs, crv_layers=self.crv_layers
        )  # shape: (crv_layers, seq_len, d_model)

        print(f"Generated query CRV shape: {query_crv.shape}")

        return query_crv

    def compute_similarities(self, query_crv, crvs):
        print(f"Query CRV shape: {query_crv.shape}")
        print(f"CRVs shape: {crvs.shape}")

        query_crv_last = query_crv[0]  # shape: (seq_len, d_model)
        crvs_last = crvs[:, 1, :, :]  # shape: (b, seq_len, d_model)

        crvs_last = crvs_last.to(query_crv_last.device)

        # query_crv_flat = query_crv.reshape(1, -1)
        # crvs_flat = crvs.reshape(crvs.shape[0], -1)

        query_crv_norm = F.normalize(query_crv_last.reshape(1, -1), p=2, dim=1)
        crvs_norm = F.normalize(crvs_last.reshape(crvs_last.shape[0], -1), p=2, dim=1)

        print(f"Query norm CRV shape: {query_crv_norm.shape}")
        print(f"CRVs norm shape: {crvs_norm.shape}")

        return torch.cosine_similarity(query_crv_norm, crvs_norm, -1)
        # return torch.mm(query_crv_norm, crvs_norm.t()).squeeze()

    def get_top_k_indices(self, similarities, k=5):
        return torch.topk(similarities, k).indices

    def retrieve_best_crv(self, query, crvs_file):
        crvs = self.load_crvs(crvs_file)
        print(f"Loaded CRVs shape: {crvs.shape}")
        query_crv = self.generate_crvs(inputs=query, output_file="data/new_stack.pt")

        similarities = self.compute_similarities(query_crv, crvs)

        if torch.isnan(similarities).any() or torch.isinf(similarities).any():
            print("Warning: NaN or Inf values in similarities")

        print(f"Max similarity: {similarities.max().item()}")
        print(f"Min similarity: {similarities.min().item()}")
        print(f"Mean similarity: {similarities.mean().item()}")

        best_indices = self.get_top_k_indices(similarities)
        print(f"Top 5 indices: {best_indices}")

        best_crv_index = best_indices[0].item()
        print(f"Best CRV index: {best_crv_index}")

        return crvs[best_crv_index]

    def __call__(self, query, crvs_file):
        return self.retrieve_best_crv(query, crvs_file)
