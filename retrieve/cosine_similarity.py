import torch
from torch.nn import functional as F

from generator.crv_generator import CRVGenerator
from utils import set_seed


# from new_pipline import set_seed, generate_crvs


def retrieve_best_crv(query, crvs_file, model, tokenizer, crv_layers, seed=42):
    """
    other similarities to try: pη (z|x) ∝ exp(d(z)T q(x)) from Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive
        nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474.
    :param query:
    :param crvs_file:
    :param model:
    :param tokenizer:
    :param crv_layers:
    :return:
    """
    # Load pre-trained model and tokenizer
    set_seed(seed)
    # Load CRVs
    if isinstance(crvs_file, str):
        crvs = torch.load(crvs_file)
    elif isinstance(crvs_file, torch.Tensor):
        crvs = crvs_file  # (b, num_layers, seq_len, d_model)

    query_crv = generate_crvs(
        model,
        tokenizer,
        input=query,
        output_file="data/new_stack.pt",
        crv_layers=crv_layers,
    )  # shape: (crv_layers, seq_len, d_model)

    print(f"query_crv shape: {query_crv.shape}")
    print(f"crvs shape: {crvs.shape}")

    # Extract last layer
    query_crv_last = query_crv[0]  # shape: (seq_len, d_model)
    crvs_last = crvs[:, 1, :, :]  # shape: (b, seq_len, d_model)

    print(f"query_crv_last shape: {query_crv_last.shape}")
    print(f"crvs_last shape: {crvs_last.shape}")

    # Move tensors to the same device
    crvs_last = crvs_last.to(query_crv_last.device)

    # Flatten and normalize
    query_crv_norm = F.normalize(query_crv_last.reshape(1, -1), p=2, dim=1)
    crvs_norm = F.normalize(crvs_last.reshape(crvs_last.shape[0], -1), p=2, dim=1)

    # Compute similarities
    # similarities = torch.mm(query_crv_norm, crvs_norm.t()).squeeze()
    similarities = torch.cosine_similarity(query_crv_norm, crvs_norm, -1)

    # Check for NaN or Inf
    if torch.isnan(similarities).any() or torch.isinf(similarities).any():
        print("Warning: NaN or Inf values in similarities")

    # Print similarity statistics
    print(f"Max similarity: {similarities.max().item()}")
    print(f"Min similarity: {similarities.min().item()}")
    print(f"Mean similarity: {similarities.mean().item()}")

    # Get top-k indices
    k = 5
    best_indices = torch.topk(similarities, k).indices
    print(f"Top {k} indices: {best_indices}")

    best_crv_index = best_indices[0].item()
    print(f"Best CRV index: {best_crv_index}")

    # print(f"query_crv shape: {query_crv.shape}")
    # print(f"crvs shape: {crvs.shape}")

    # Extract last layer
    # query_crv_last = query_crv[-1]  # shape: (seq_len, d_model)
    # crvs_last = crvs[:, -1, :, :]  # shape: (b, seq_len, d_model)
    #
    # # crvs = crvs.to(query_crv.device)
    # # Move tensors to the same device
    # crvs_last = crvs_last.to(query_crv_last.device)
    #
    # query_crv_norm = F.normalize(query_crv_last.reshape(1, -1), p=2, dim=1)
    # crvs_norm = F.normalize(crvs.reshape(crvs_last.shape[0], -1), p=2, dim=1)
    # similarities = torch.mm(query_crv_norm, crvs_norm.t()).squeeze()

    # Compute similarities
    # similarities = torch.cosine_similarity(
    #     query_crv.reshape(1, -1), crvs.reshape(crvs.shape[0], -1)
    # )
    # similarities = torch.cosine_similarity(
    #     query_crv_last.unsqueeze(0), crvs_last, dim=-1
    # )
    # similarities = similarities.mean(dim=1)  # Average over sequence length
    # print("similarities: ", similarities.shape)
    # Get the index of the most similar CRV
    # best_crv_index = similarities.argmax().item()
    best_crv_index = torch.argmax(similarities)
    # best_crv_index1 = similarities1.argmax().item()

    print(f"best_crv_index: {best_crv_index}")

    return crvs[best_crv_index]


class CRVRetriever:
    def __init__(self, model, tokenizer, crv_layers, seed=42, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.crv_layers = crv_layers
        self.seed = seed
        self.set_seed()
        self.max_length = max_length
        self.crv_generator = CRVGenerator(model, tokenizer, max_length=max_length)
        self.input_crv = self.crv_generator.generate_crvs(
            "dataset", crv_layers=crv_layers
        )

    def set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def load_crvs(self, crvs_file):
        if isinstance(crvs_file, str):
            return torch.load(crvs_file)
        elif isinstance(crvs_file, torch.Tensor):
            return crvs_file
        else:
            raise ValueError(
                "crvs_file must be either a string (file path) or a torch.Tensor"
            )

    def generate_crvs(self, input, output_file="data/new_stack.pt"):

        query_crv = self.crv_generator.generate_crvs(
            input, crv_layers=self.crv_layers
        )  # shape: (crv_layers, seq_len, d_model)

        return query_crv

    def compute_similarities(self, query_crv, crvs):
        query_crv_last = query_crv[0]  # shape: (seq_len, d_model)
        crvs_last = crvs[:, 1, :, :]  # shape: (b, seq_len, d_model)

        crvs_last = crvs_last.to(query_crv_last.device)

        query_crv_norm = F.normalize(query_crv_last.reshape(1, -1), p=2, dim=1)
        crvs_norm = F.normalize(crvs_last.reshape(crvs_last.shape[0], -1), p=2, dim=1)

        return torch.cosine_similarity(query_crv_norm, crvs_norm, -1)

    def get_top_k_indices(self, similarities, k=5):
        return torch.topk(similarities, k).indices

    def retrieve_best_crv(self, query, crvs_file):
        crvs = self.load_crvs(crvs_file)

        query_crv = self.generate_crvs(input=query, output_file="data/new_stack.pt")

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
