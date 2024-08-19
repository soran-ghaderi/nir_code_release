import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from blib2to3.pgen2.driver import Optional
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, TextStreamer


from moc_layers import LlamaForCausalLM

from main import write_results_to_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH = 128
BATCH_SIZE = 4
NUM_CONTEXTS = 20
CRV_DIM = 4096  # Assuming LLaMA hidden size
SUBSET_SIZE = 20
CRV_SAVE_BATCH = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MathDataset(Dataset):
    def __init__(
        self, tokenizer, split="train", max_length=MAX_LENGTH, subset_size=None, seed=42
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
        max_length=MAX_LENGTH,
        subset_size=SUBSET_SIZE,
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


def load_custom_transformer(
    model_path, tokenizer_path, hf_token=None, load_in_8bit=False, seed=42
):
    set_seed(seed)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        use_auth_token=hf_token,
        device_map="auto",
        # load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
    )
    model.eval()  # Set the model to evaluation mode

    return model, tokenizer


def write_results_to_file(
    file_path,
    results,
    prompt,
    max_length=50,
    max_new_tokens=100,
    num_return_sequences=1,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=1,
):
    """
    Writes the results to a file in an organized and tabular format.

    Args:
        file_path (str): The path to the output file.
        results (list): A list of dictionaries containing the results.
    """
    context = (
        "The Apollo 11 mission was the first manned mission to land on the Moon. Launched by NASA on July 16, "
        "1969, it carried astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. On July 20, 1969, Neil "
        "Armstrong and Buzz Aldrin landed the lunar module Eagle on the Moon while Michael Collins remained in "
        "lunar orbit in the command module Columbia. Armstrong became the first person to step onto the lunar "
        "surface, followed by Aldrin. They spent about two and a quarter hours outside the spacecraft, "
        "collecting samples and conducting experiments. The mission returned to Earth on July 24, 1969."
    )

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Context: {context}\n\n")
        file.write(f"Model Parameters\n")
        file.write(f"Max Length: {max_length}\n")
        file.write(f"Max New Tokens: {max_new_tokens}\n")
        file.write(f"Num Return Sequences: {num_return_sequences}\n")
        file.write(f"Temperature: {temperature}\n")
        file.write(f"Top K: {top_k}\n")
        file.write(f"Top P: {top_p}\n")
        file.write(f"Repetition Penalty: {repetition_penalty}\n")
        file.write(f"No Repeat N-gram Size: {no_repeat_ngram_size}\n\n\n")
        file.write(f"Generated text by concatenated layer index\n\n")

        # writer.writerow(result)
        # for key, value in results.items():
        file.write(f"{results}\n")

        file.write(f"\n{'==' * 80}\n")

    print(f"Results written to {file_path}")


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    max_new_tokens=100,
    num_return_sequences=1,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=1,
    cross_attend=False,
    config=None,
    crv_layer_idx=None,
    output_file="results/concat_different_layers.csv",
    seed=42,
):
    """

    :param model:
    :param tokenizer:
    :param prompt:
    :param max_length:
    :param max_new_tokens:
    :param num_return_sequences:
    :param temperature:
    :param top_k:
    :param top_p:
    :param repetition_penalty:
    :param no_repeat_ngram_size:
    :param cross_attend:
    :param config:
    :param crv_layer_idx:
    :param output_file:
    :return:
    """
    set_seed(seed)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    results = []

    # Generate text using the model
    if not crv_layer_idx == None:
        print(f"concatenate at layer {crv_layer_idx}:")

        # model.model.set_layers_to_concat(layer_idx=crv_layer_idx)
        # model.model.set_is_crv_concatenated(is_crv_concatenated=False)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streamer=streamer,
            )

        generated_text = outputs[0][input_ids.shape[-1] :]
        decoded_text = tokenizer.decode(generated_text, skip_special_tokens=True)

        results = decoded_text
        write_results_to_file(
            output_file,
            results,
            prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streamer=streamer,
            )

        generated_text = outputs[0][input_ids.shape[-1] :]
        decoded_text = tokenizer.decode(generated_text, skip_special_tokens=True)

    return decoded_text


def generate_crvs(
    model,
    tokenizer,
    input,
    output_file,
    crv_layers: Optional[tuple[int, list]] = None,
    seed=42,
):  # crv_layers: layers to save their hidden states
    set_seed(seed)
    if input == "dataset":
        # Create dataset and dataloader
        # dataset = MathDataset(tokenizer, split="train", subset_size=SUBSET_SIZE)
        dataset = GSM8KDataset(tokenizer, split="train", subset_size=SUBSET_SIZE)

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        crvs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= NUM_CONTEXTS // BATCH_SIZE:
                    break

                inputs = batch["input_ids"].to(model.device)
                # attention_mask = batch["attention_mask"].to(model.device)

                # generate embeds
                # print("input types: ", type(inputs))
                # print("input shape: ", inputs.shape)
                outputs = model(
                    inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

                if crv_layers == None:
                    print(
                        "outputs.hidden_states[crv_layers].shape: ",
                        outputs.hidden_states[batch_idx].shape,
                    )
                    prompt_stacked_crv = (
                        torch.stack([output for output in outputs.hidden_states], dim=0)
                        .squeeze(1)
                        .transpose(0, 1)
                    )  # (b, layers, seq_len, d_model)
                elif isinstance(crv_layers, list):
                    # if more than one layer specified
                    prompt_stacked_crv = (
                        torch.stack(
                            [
                                output
                                for idx, output in enumerate(outputs.hidden_states)
                                if idx in crv_layers
                            ],
                            dim=0,
                        )
                        .squeeze(1)
                        .transpose(0, 1)
                    )  # (b, len(crv_layers), seq_len, d_model)
                elif isinstance(crv_layers, int):
                    # if saving one layer
                    print(
                        "int - outputs.hidden_states[crv_layers].shape: ",
                        outputs.hidden_states[batch_idx].shape,
                    )
                    prompt_stacked_crv = outputs.hidden_states[crv_layers].squeeze(
                        1
                    )  # (b, seq_len, d_model)

                print("prompt_stacked_crv: ", prompt_stacked_crv.shape)
                # Use the last hidden state as the CRV
                # batch_crvs = outputs.hidden_states[-1].mean(dim=1)  # Average pooling
                crvs.append(prompt_stacked_crv)

                if (batch_idx + 1) % CRV_SAVE_BATCH == 0:
                    print(f"Processed {(batch_idx + 1) * BATCH_SIZE} contexts")

        # Stack all CRVs
        crvs_tensor = torch.cat(crvs, dim=0)

        # Save CRVs
        torch.save(crvs_tensor, output_file)
        print(f"CRVs saved to {output_file}")

    elif isinstance(input, str):
        logger.info("The input received is a query")
        with torch.no_grad():
            # inputs = tokenizer(input, return_tensors="pt").to(model.device)
            inputs = tokenizer(
                input,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = inputs["input_ids"]
            # generate embeds
            outputs = model(
                inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            if crv_layers == None:
                print(
                    "outputs.hidden_states[crv_layers].shape: ",
                    # outputs.hidden_states[batch_idx].shape,
                )
                prompt_stacked_crv = torch.stack(
                    [output for output in outputs.hidden_states], dim=0
                ).squeeze(
                    1
                )  # (layers, seq_len, d_model)
            elif isinstance(crv_layers, list):
                # if more than one layer specified
                prompt_stacked_crv = torch.stack(
                    [
                        output
                        for idx, output in enumerate(outputs.hidden_states)
                        if idx in crv_layers
                    ],
                    dim=0,
                ).squeeze(
                    1
                )  # (len(crv_layers), seq_len, d_model)
            elif isinstance(crv_layers, int):
                # if saving one layer
                prompt_stacked_crv = outputs.hidden_states[crv_layers].squeeze(
                    1
                )  # (1, seq_len, d_model), the 1 is the len(crv_layers)
            print("prompt_stacked_crv: ", prompt_stacked_crv.shape)

        crvs_tensor = prompt_stacked_crv

    return crvs_tensor


def main():
    seed = 42
    set_seed(seed)
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_token = "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"
    config = AutoConfig.from_pretrained(model_path, use_auth_token=hf_token)
    model, tokenizer = load_custom_transformer(
        model_path, tokenizer_path, hf_token=hf_token
    )

    crv_layers = [1, 10, 15, 20, 32]
    # crv_layers = 32

    prompt = "elaborate more."
    print("model type: ", type(model))
    print("config.hidden_size: ", config.num_hidden_layers)
    print("config._attn_implementation: ", config._attn_implementation)

    crvs_file = generate_crvs(
        model,
        tokenizer,
        input="dataset",
        output_file="data/new_stack.pt",
        crv_layers=crv_layers,
    )  # shape: (subset_size, crv_layers, seq_len, d_model)

    # query = "Solve the following problem: If x + y = 10 and x - y = 4, what are the values of x and y?"
    query = (
        "Problem: Grant has four times as many vacations as Kelvin has classes. If Kelvin has 90 classes, "
        "how many vacations and classes do Grant and Kelvin have altogether?"
    )
    # query = "Problem: Find the center of the circle with equation $x^2 - 6x + 5y = 11$. Solution:"

    # Input query

    # Retrieve the best CRV
    best_crv = retrieve_best_crv(
        query, crvs_file, model, tokenizer, crv_layers=crv_layers
    )
    print("best_crv.shape: ", best_crv.shape)
    #
    # reduced_crv = best_crv.mean(dim=-1)  # (layers/len(crv_layers), seq_len)
    # print("Reduced CRV shape:", reduced_crv.shape)
    # print("Reduced CRV:", str(reduced_crv[0]))

    # # Set the CRV in the model (e.g., integrate at layer 5)
    # model.model.set_crv(best_crv, layer_idx=10, crv_layers=crv_layers)

    generated_text = generate_text(
        model,
        tokenizer,
        prompt=query,
        # max_length=100,
        max_new_tokens=300,
        num_return_sequences=1,
        temperature=1.0,
        top_k=1,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        cross_attend=False,
        config=config,
        crv_layer_idx=[5, 10],
    )
    print(generated_text)


if __name__ == "__main__":

    main()
