import os
from typing import List, Union
import logging
from zlib import compressobj

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, TextStreamer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import zstandard as zstd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from moc_layers import LlamaSdpaAttention, LlamaForCausalLM

# Constants
MAX_LENGTH = 512
BATCH_SIZE = 4
NUM_CONTEXTS = 20
CRV_DIM = 4096  # Assuming LLaMA hidden size
SUBSET_SIZE = 20
CRV_SAVE_BATCH = 20


class MathDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=MAX_LENGTH):
        self.dataset = load_dataset("hendrycks/competition_math", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            problem = item.get("problem", "")
            solution = item.get("solution", "")

            input_text = f"Problem: {problem}\nSolution: {solution}"
            encoded = self.tokenizer.encode_plus(
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


def load_custom_transformer(pretrained_model_path, tokenizer_path, config, hf_token):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=hf_token)

    # Load model in 16-bit precision
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        use_auth_token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16,  # Use 16-bit precision
    )

    model.eval()
    return model, tokenizer


class OptimizedCRVHandler:
    def __init__(self, compression_ratio=0.1, precision=16):
        self.compression_ratio = compression_ratio
        self.precision = precision
        self.compressor = zstd.ZstdCompressor(level=22)
        self.dtype_map = {
            "torch.float16": np.float16,
            "torch.float32": np.float32,
            "torch.int8": np.int8,
        }

    def compress_crv(self, crv):
        crv_cpu = crv.cpu().detach()
        original_shape = crv_cpu.shape
        crv_flat = crv_cpu.reshape(-1)

        threshold = np.quantile(np.abs(crv_flat.numpy()), 1 - self.compression_ratio)
        mask = torch.abs(crv_flat) > threshold
        pruned_crv = crv_flat * mask

        if self.precision == 16:
            pruned_crv = pruned_crv.half()
        elif self.precision == 8:
            pruned_crv = (pruned_crv * 127).to(torch.int8)

        compressed_data = self.compressor.compress(pruned_crv.numpy().tobytes())

        return {
            "data": compressed_data,
            "shape": original_shape,
            "dtype": str(pruned_crv.dtype),
        }

    def decompress_crv(self, compressed_crv):
        decompressor = zstd.ZstdDecompressor()

        # Convert PyTorch dtype string to NumPy dtype
        torch_dtype = compressed_crv["dtype"]
        if torch_dtype.startswith("torch."):
            np_dtype = self.dtype_map.get(torch_dtype)
            if np_dtype is None:
                raise ValueError(f"Unsupported dtype: {torch_dtype}")
        else:
            np_dtype = np.dtype(torch_dtype)

        crv_array = np.frombuffer(
            decompressor.decompress(compressed_crv["data"]), dtype=np_dtype
        )

        crv = torch.from_numpy(crv_array.reshape(compressed_crv["shape"]))

        if crv.dtype == torch.int8:
            crv = crv.float() / 127
        return crv.float()  # Always return float32 for consistency

    def load_crvs(self, file_path):
        compressed_crvs = torch.load(file_path, map_location="cpu")
        return self.decompress_crv(compressed_crvs)


class CRVGenerator:
    def __init__(self, model, save_dir="crv_batches", target_layer=10):
        self.model = model
        self.save_dir = save_dir
        self.target_layer = target_layer
        self.crv_handler = OptimizedCRVHandler()
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def generate_crvs(self, dataloader):
        self.model.eval()
        crvs = []
        total_saved = 0

        for i, batch in enumerate(tqdm(dataloader, desc="Generating CRVs")):
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)

            outputs = self.model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            hidden_state = outputs.hidden_states[self.target_layer]

            crvs.append(hidden_state.cpu())

            if len(crvs) * hidden_state.shape[0] >= CRV_SAVE_BATCH:
                self.save_crvs(
                    torch.cat(crvs, dim=0), f"crvs_batch_{total_saved + 1}.pt"
                )
                total_saved += len(crvs) * hidden_state.shape[0]
                crvs = []

        if crvs:
            self.save_crvs(torch.cat(crvs, dim=0), f"crvs_batch_final.pt")
            total_saved += len(crvs) * hidden_state.shape[0]

        logger.info(f"Total CRVs saved: {total_saved}")

    def save_crvs(self, crvs, filename):
        compressed_crvs = self.crv_handler.compress_crv(crvs)
        file_path = os.path.join(self.save_dir, filename)
        torch.save(compressed_crvs, file_path)
        logger.info(f"Saved compressed CRVs to {file_path}")

        # Verify saved file
        try:
            self.load_crvs(file_path)
            logger.info(f"Successfully verified saved CRVs in {file_path}")
        except Exception as e:
            logger.error(f"Error verifying saved CRVs in {file_path}: {str(e)}")

    def load_crvs(self, file_path):
        compressed_crvs = torch.load(file_path, map_location="cpu")
        return self.crv_handler.decompress_crv(compressed_crvs)


class CRVIntegrator:
    def __init__(self, model, crv_source: Union[str, List[str]]):
        self.model = model
        self.crv_files = self._get_crv_files(crv_source)
        self.current_crvs = None
        self.current_file_idx = 0
        self.crv_handler = OptimizedCRVHandler()
        if not self.crv_files:
            logger.warning(
                "No CRV files found. CRVIntegrator may not function as expected."
            )
        else:
            self.load_next_crvs()

    def _get_crv_files(self, crv_source: Union[str, List[str]]) -> List[str]:
        if isinstance(crv_source, str):
            if os.path.isdir(crv_source):
                files = [
                    os.path.join(crv_source, f)
                    for f in os.listdir(crv_source)
                    if f.startswith("crvs_batch_") and f.endswith(".pt")
                ]
                logger.info(f"Found {len(files)} CRV files in directory: {crv_source}")
                return files
            elif os.path.isfile(crv_source) and crv_source.endswith(".pt"):
                logger.info(f"Using single CRV file: {crv_source}")
                return [crv_source]
            else:
                logger.error(f"Invalid crv_source: {crv_source}")
                raise ValueError(f"Invalid crv_source: {crv_source}")
        elif isinstance(crv_source, list):
            valid_files = [
                f for f in crv_source if f.endswith(".pt") and os.path.isfile(f)
            ]
            logger.info(f"Using {len(valid_files)} valid CRV files from provided list")
            return valid_files
        else:
            logger.error(
                "crv_source must be either a string (directory or file path) or a list of file paths"
            )
            raise TypeError(
                "crv_source must be either a string (directory or file path) or a list of file paths"
            )

    def load_next_crvs(self):
        if self.current_file_idx < len(self.crv_files):
            file_path = self.crv_files[self.current_file_idx]
            logger.info(f"Attempting to load CRV file: {file_path}")
            try:
                if not os.path.exists(file_path):
                    logger.error(f"File does not exist: {file_path}")
                    raise FileNotFoundError(f"No such file or directory: '{file_path}'")

                self.current_crvs = self.crv_handler.load_crvs(file_path)

                self.current_file_idx += 1
                logger.info(f"Successfully loaded CRV file: {file_path}")
                logger.info(
                    f"Loaded CRVs shape: {self.current_crvs.shape}, dtype: {self.current_crvs.dtype}"
                )
                return True
            except Exception as e:
                logger.error(f"Error loading CRV file {file_path}: {str(e)}")
                self.current_file_idx += 1
                return self.load_next_crvs()
        logger.info("No more CRV files to load")
        return False

    def find_similar_crv(self, query):
        if self.current_crvs is None:
            logger.warning("No CRVs loaded. Cannot find similar CRV.")
            return None

        logger.info(f"Query shape: {query.shape}")
        logger.info(f"Current CRVs shape: {self.current_crvs.shape}")

        # Ensure query and CRVs are 2D tensors
        if query.dim() > 2:
            query = query.view(query.size(0), -1)
        if self.current_crvs.dim() > 2:
            self.current_crvs = self.current_crvs.view(self.current_crvs.size(0), -1)

        logger.info(f"Reshaped query shape: {query.shape}")
        logger.info(f"Reshaped CRVs shape: {self.current_crvs.shape}")

        # Ensure the last dimension matches
        if query.size(-1) != self.current_crvs.size(-1):
            logger.warning(
                f"Dimension mismatch. Query: {query.size(-1)}, CRVs: {self.current_crvs.size(-1)}"
            )
            # Option 1: Truncate the larger dimension
            min_dim = min(query.size(-1), self.current_crvs.size(-1))
            query = query[..., :min_dim]
            self.current_crvs = self.current_crvs[..., :min_dim]
            logger.info(f"Truncated to match dimensions. New shape: {min_dim}")

        try:
            similarities = torch.nn.functional.cosine_similarity(
                query.unsqueeze(1), self.current_crvs
            )
            most_similar_idx = torch.argmax(similarities)
            logger.info(f"Found most similar CRV at index: {most_similar_idx}")
            return self.current_crvs[most_similar_idx]
        except RuntimeError as e:
            logger.error(f"Error computing similarities: {str(e)}")
            return None

    def integrate_crv(self, hidden_states, crv, layer_idx):
        integrated = list(hidden_states)
        integrated[layer_idx] = torch.cat(
            [hidden_states[layer_idx], crv.to(hidden_states[layer_idx].device)], dim=-1
        )
        return tuple(integrated)


def main():
    # Load model and tokenizer
    hf_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_token = "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"
    config = AutoConfig.from_pretrained(hf_model_path, use_auth_token=hf_token)
    model, tokenizer = load_custom_transformer(
        hf_model_path, hf_tokenizer_path, config=config, hf_token=hf_token
    )

    # Load dataset
    full_dataset = MathDataset(tokenizer, split="train")
    subset_indices = torch.randperm(len(full_dataset))[:SUBSET_SIZE].tolist()
    subset_dataset = Subset(full_dataset, subset_indices)
    train_dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Mode 1: Generate CRVs
    crv_generator = CRVGenerator(model)
    # crv_generator.generate_crvs(train_dataloader)

    # Mode 2: Integrate CRVs
    crv_dir = "crv_batches"
    logger.info(f"Initializing CRVIntegrator with directory: {crv_dir}")
    crv_integrator = CRVIntegrator(model, crv_dir)

    # Test integration
    test_dataset = MathDataset(tokenizer, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

            query = hidden_states[-1].mean(dim=1)
            similar_crv = crv_integrator.find_similar_crv(query)

            if similar_crv is not None:
                integrated_hidden_states = crv_integrator.integrate_crv(
                    hidden_states, similar_crv, layer_idx=10
                )

                logits = model.lm_head(integrated_hidden_states[-1])
                next_token = torch.argmax(logits[:, -1, :])
                generated_text = tokenizer.decode(next_token.item())

                logger.info(f"Input: {tokenizer.decode(input_ids[0])}")
                logger.info(f"Generated: {generated_text}")
            else:
                logger.warning("No similar CRV found.")

        break  # Just for demonstration, remove this in actual use


if __name__ == "__main__":
    main()
