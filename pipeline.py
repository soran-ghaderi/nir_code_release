import os

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, TextStreamer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import zstandard as zstd

from moc_layers import LlamaSdpaAttention, LlamaForCausalLM

# Constants
MAX_LENGTH = 512
BATCH_SIZE = 4
NUM_CONTEXTS = 20
CRV_DIM = 4096  # Assuming LLaMA hidden size
SUBSET_SIZE = 100
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
    def __init__(self, compression_ratio=0.1, precision=16, use_sparse=True):
        self.compression_ratio = compression_ratio
        self.precision = precision
        self.use_sparse = use_sparse
        self.compressor = zstd.ZstdCompressor(level=22)  # High compression level

    def compress_crv(self, crv):
        # Ensure we're working with a float tensor for quantile operation
        crv_float = crv.float()

        # Prune small values
        threshold = torch.quantile(torch.abs(crv_float), 1 - self.compression_ratio)
        mask = torch.abs(crv_float) > threshold
        pruned_crv = crv_float * mask

        # Convert to lower precision after pruning
        if self.precision == 16:
            pruned_crv = pruned_crv.half()
        elif self.precision == 8:
            pruned_crv = pruned_crv.to(torch.int8)

        if self.use_sparse:
            # Convert to sparse format
            indices = torch.nonzero(pruned_crv).cpu().numpy()
            values = pruned_crv[pruned_crv != 0].cpu().numpy()
            shape = pruned_crv.shape
            sparse_crv = csr_matrix(
                (values, (indices[:, 0], indices[:, 1])), shape=shape
            )

            # Compress the sparse data
            compressed_data = self.compressor.compress(sparse_crv.data.tobytes())
            compressed_indices = self.compressor.compress(sparse_crv.indices.tobytes())
            compressed_indptr = self.compressor.compress(sparse_crv.indptr.tobytes())

            return {
                "data": compressed_data,
                "indices": compressed_indices,
                "indptr": compressed_indptr,
                "shape": shape,
                "dtype": str(pruned_crv.dtype),
            }
        else:
            # Compress the pruned tensor directly
            return {
                "data": self.compressor.compress(pruned_crv.cpu().numpy().tobytes()),
                "shape": pruned_crv.shape,
                "dtype": str(pruned_crv.dtype),
            }

    def decompress_crv(self, compressed_crv):
        decompressor = zstd.ZstdDecompressor()

        if self.use_sparse:
            data = np.frombuffer(
                decompressor.decompress(compressed_crv["data"]),
                dtype=np.dtype(compressed_crv["dtype"]),
            )
            indices = np.frombuffer(
                decompressor.decompress(compressed_crv["indices"]), dtype=np.int32
            )
            indptr = np.frombuffer(
                decompressor.decompress(compressed_crv["indptr"]), dtype=np.int32
            )

            sparse_crv = csr_matrix(
                (data, indices, indptr), shape=compressed_crv["shape"]
            )
            crv = torch.from_numpy(sparse_crv.toarray())
        else:
            crv_array = np.frombuffer(
                decompressor.decompress(compressed_crv["data"]),
                dtype=np.dtype(compressed_crv["dtype"]),
            )
            crv = torch.from_numpy(crv_array.reshape(compressed_crv["shape"]))

        return crv.float()  # Convert back to float32 for compatibility


class CRVGenerator:
    def __init__(self, model):
        self.model = model
        self.crv_handler = OptimizedCRVHandler()

    @torch.no_grad()
    def generate_crvs(self, dataloader):
        self.model.eval()
        crvs = []

        for i, batch in enumerate(tqdm(dataloader, desc="Generating CRVs")):
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)

            outputs = self.model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

            # Stack hidden states from all layers
            batch_crvs = torch.stack(
                hidden_states, dim=1
            )  # [batch_size, num_layers, seq_len, hidden_size]
            crvs.append(batch_crvs.cpu())  # Move to CPU to save GPU memory

            # Save every CRV_SAVE_BATCH
            if (i + 1) % (CRV_SAVE_BATCH // BATCH_SIZE) == 0:
                self.save_crvs(torch.cat(crvs, dim=0), f"data/crvs_batch_{i + 1}.pt")
                crvs = []  # Clear the list to free up memory

        # Save any remaining CRVs
        if crvs:
            self.save_crvs(torch.cat(crvs, dim=0), f"data/crvs_batch_final.pt")

    def save_crvs(self, crvs, filename):
        compressed_crvs = self.crv_handler.compress_crv(crvs)
        file_path = os.path.join(self.save_dir, filename)
        torch.save(compressed_crvs, file_path)
        print(f"Saved compressed CRVs to {file_path}")


class CRVIntegrator:
    def __init__(self, model, crv_files):
        self.model = model
        self.crv_files = crv_files
        self.current_crvs = None
        self.current_file_idx = 0
        self.load_next_crvs()
        self.crv_handler = OptimizedCRVHandler()

        self.knn = NearestNeighbors(n_neighbors=1, metric="cosine")
        self.update_knn()

    def load_next_crvs(self):
        if self.current_file_idx < len(self.crv_files):
            file_path = os.path.join(
                self.crv_dir, self.crv_files[self.current_file_idx]
            )
            compressed_crvs = torch.load(file_path)
            self.current_crvs = self.crv_handler.decompress_crv(compressed_crvs)
            self.current_file_idx += 1
            return True
        return False

    def update_knn(self):
        if self.current_crvs is not None:
            self.knn.fit(self.current_crvs.view(-1, CRV_DIM).cpu().numpy())

    def find_similar_crv(self, query):
        if self.current_crvs is None:
            return None

        _, indices = self.knn.kneighbors(query.cpu().numpy().reshape(1, -1))
        return self.current_crvs[indices[0][0]]

    def integrate_crv(self, hidden_states, crv, layer_idx):
        integrated = list(hidden_states)
        integrated[layer_idx] = torch.cat(
            [
                hidden_states[layer_idx],
                crv[layer_idx].to(hidden_states[layer_idx].device),
            ],
            dim=-1,
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
    crv_generator.generate_crvs(train_dataloader)

    # Mode 2: Integrate CRVs
    crv_files = [
        f
        for f in os.listdir(".")
        if f.startswith("data/crvs_batch_") and f.endswith(".pt")
    ]
    crv_integrator = CRVIntegrator(model, crv_files)

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

            # Find similar CRV
            query = hidden_states[-1].mean(dim=1)
            similar_crv = crv_integrator.find_similar_crv(query)

            if similar_crv is not None:
                # Integrate CRV at a specific layer (e.g., layer 10)
                integrated_hidden_states = crv_integrator.integrate_crv(
                    hidden_states, similar_crv, layer_idx=10
                )

                # Generate output using integrated hidden states
                logits = model.lm_head(integrated_hidden_states[-1])
                next_token = torch.argmax(logits[:, -1, :])
                generated_text = tokenizer.decode(next_token.item())

                print(f"Input: {tokenizer.decode(input_ids[0])}")
                print(f"Generated: {generated_text}")
            else:
                print("No similar CRV found.")

            break  # Just for demonstration, remove this in actual use


if __name__ == "__main__":
    main()
