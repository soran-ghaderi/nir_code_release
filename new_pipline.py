import csv
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from blib2to3.pgen2.driver import Optional
from transformers import AutoConfig, AutoTokenizer, TextStreamer
import configs
from generator.crv_generator import CRVGenerator
from moc_layers import LlamaForCausalLM

from retrieve.cosine_similarity import CRVRetriever
from utils import set_seed, logger

logger = logger()
# Constants


class DNMemory(nn.Module):
    def __init__(self, input_size, memory_size, word_size, num_reads, num_writes):
        super(DNMemory, self).__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = num_writes

        # Initialize memory
        self.memory = nn.Parameter(torch.zeros(memory_size, word_size))

        # Read and write heads
        self.read_heads = nn.ModuleList([ReadHead(word_size) for _ in range(num_reads)])
        self.write_heads = nn.ModuleList(
            [WriteHead(word_size) for _ in range(num_writes)]
        )

        # Controller
        self.controller = nn.LSTM(input_size + num_reads * word_size, 256, num_layers=1)

        # Output layer
        self.output = nn.Linear(256 + num_reads * word_size, input_size)

    def forward(self, x, prev_state=None):
        batch_size = x.size(0)
        if prev_state is None:
            prev_state = self.init_state(batch_size)

        controller_state, prev_reads = prev_state

        # Read from memory
        reads = [head(self.memory) for head in self.read_heads]
        read_vectors = torch.cat(reads, dim=1)

        # Controller input
        controller_input = torch.cat([x, read_vectors], dim=1)
        controller_output, controller_state = self.controller(
            controller_input.unsqueeze(0), controller_state
        )
        controller_output = controller_output.squeeze(0)

        # Write to memory
        for head in self.write_heads:
            self.memory = head(self.memory, controller_output)

        # Output
        output = self.output(torch.cat([controller_output, read_vectors], dim=1))

        return output, (controller_state, reads)

    def init_state(self, batch_size):
        controller_state = (
            torch.zeros(1, batch_size, 256),
            torch.zeros(1, batch_size, 256),
        )
        reads = [torch.zeros(batch_size, self.word_size) for _ in range(self.num_reads)]
        return (controller_state, reads)


class ReadHead(nn.Module):
    def __init__(self, word_size):
        super(ReadHead, self).__init__()
        self.word_size = word_size
        self.attention = nn.Linear(word_size, 1)

    def forward(self, memory):
        attention = F.softmax(self.attention(memory), dim=0)
        read_vector = torch.sum(attention * memory, dim=0)
        return read_vector


class WriteHead(nn.Module):
    def __init__(self, word_size):
        super(WriteHead, self).__init__()
        self.word_size = word_size
        self.erase_vector = nn.Linear(word_size, word_size)
        self.write_vector = nn.Linear(word_size, word_size)
        self.attention = nn.Linear(word_size, 1)

    def forward(self, memory, controller_output):
        attention = F.softmax(self.attention(memory), dim=0)
        erase = torch.sigmoid(self.erase_vector(controller_output))
        write = self.write_vector(controller_output)
        memory = memory * (1 - attention * erase) + attention * write
        return memory


# Usage with CRVs
class CRVMemoryManager:
    def __init__(self, crv_size, memory_size=100, num_reads=4, num_writes=1):
        self.dnc = DNMemory(crv_size, memory_size, crv_size, num_reads, num_writes)
        self.state = None

    def save_crv(self, crv):
        output, self.state = self.dnc(crv, self.state)
        return output

    def retrieve_best_crv(self, query_crv):
        output, self.state = self.dnc(query_crv, self.state)
        return output


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


class TextGenerator:
    def __init__(self, model, tokenizer, seed=42):
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.streamer = TextStreamer(
            tokenizer, skip_special_tokens=True, skip_prompt=True
        )

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_text(
        self,
        prompt: str,
        max_length: int = 50,
        max_new_tokens: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 1,
        crv_layer_idx: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> Union[str, list]:
        """
        Generate text based on the given prompt and parameters.

        :param prompt: The input prompt for text generation
        :param max_length: Maximum length of the generated text
        :param max_new_tokens: Maximum number of new tokens to generate
        :param num_return_sequences: Number of alternative sequences to generate
        :param temperature: Temperature for controlling randomness in generation
        :param top_k: Number of highest probability vocabulary tokens to keep for top-k-filtering
        :param top_p: Cumulative probability for top-p-filtering
        :param repetition_penalty: Penalty for repeating tokens
        :param no_repeat_ngram_size: Size of n-grams to avoid repeating
        :param crv_layer_idx: Index of the layer to concatenate for CRV (if applicable)
        :param output_file: File to write results (if provided)
        :return: Generated text or list of generated texts
        """
        self.set_seed(self.seed)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )

        generate_kwargs = {
            "input_ids": input_ids,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "num_return_sequences": num_return_sequences,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "streamer": self.streamer,
        }

        if crv_layer_idx is not None:
            print(f"Concatenate at layer {crv_layer_idx}:")
            # Uncomment these lines if you need to set CRV concatenation
            # self.model.model.set_layers_to_concat(layer_idx=crv_layer_idx)
            # self.model.model.set_is_crv_concatenated(is_crv_concatenated=False)

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        generated_texts = [
            self.tokenizer.decode(
                output[input_ids.shape[-1] :], skip_special_tokens=True
            )
            for output in outputs
        ]

        if output_file:
            self._write_results_to_file(
                output_file, generated_texts, prompt, **generate_kwargs
            )

        return generated_texts[0] if len(generated_texts) == 1 else generated_texts

    def _write_results_to_file(self, output_file, results, prompt, **kwargs):
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [prompt] + results + [f"{k}={v}" for k, v in kwargs.items()]
            )


def main():
    seed = 42
    set_seed(seed)
    model_urls = {
        "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    model_path = model_urls["llama31"]
    tokenizer_path = model_path
    # tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
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

    crv_generator = CRVGenerator(model, tokenizer, max_length=configs.MAX_LENGTH)
    crvs_file = crv_generator.generate_crvs(
        "dataset", "data/new_stack.pt", crv_layers=crv_layers
    )

    # crvs_file = generate_crvs(
    #     model,
    #     tokenizer,
    #     input="dataset",
    #     output_file="data/new_stack.pt",
    #     crv_layers=crv_layers,
    # )  # shape: (subset_size, crv_layers, seq_len, d_model)

    # query = "Solve the following problem: If x + y = 10 and x - y = 4, what are the values of x and y?"
    query = (
        "Problem: Grant has four times as many vacations as Kelvin has classes. If Kelvin has 90 classes, "
        "how many vacations and classes do Grant and Kelvin have altogether?"
    )
    query = (
        "Aaron pays his actuary membership fees each year. The membership fee increases yearly by $10. If he "
        "pays $80 in the first year, how much does his membership cost, in dollars, in the sixth year?"
    )
    # query = "Problem: Find the center of the circle with equation $x^2 - 6x + 5y = 11$. Solution:"

    # Input query

    # Retrieve the best CRV
    # best_crv = retrieve_best_crv(
    #     query, crvs_file, model, tokenizer, crv_layers=crv_layers
    # )

    retriever = CRVRetriever(
        model, tokenizer, crv_layers, max_length=configs.MAX_LENGTH
    )
    best_crv = retriever(query, crvs_file)
    print("best_crv.shape: ", best_crv.shape)
    #
    # reduced_crv = best_crv.mean(dim=-1)  # (layers/len(crv_layers), seq_len)
    # print("Reduced CRV shape:", reduced_crv.shape)
    # print("Reduced CRV:", str(reduced_crv[0]))

    # # Set the CRV in the model (e.g., integrate at layer 5)
    # model.model.set_crv(best_crv, layer_idx=1, crv_layers=crv_layers)

    text_generator = TextGenerator(model, tokenizer)
    generated_text = text_generator.generate_text(
        "Once upon a time", output_file="data/results.csv"
    )

    # generated_text = generate_text(
    #     model,
    #     tokenizer,
    #     prompt=query,
    #     # max_length=100,
    #     max_new_tokens=300,
    #     num_return_sequences=1,
    #     temperature=1.0,
    #     top_k=1,
    #     top_p=0.95,
    #     repetition_penalty=1.2,
    #     no_repeat_ngram_size=3,
    #     cross_attend=False,
    #     config=config,
    #     crv_layer_idx=[5, 10],
    # )
    print(generated_text)


if __name__ == "__main__":

    main()
