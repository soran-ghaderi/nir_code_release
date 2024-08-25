import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
import configs
from generator.crv_generator import CRVGenerator
from generator.text_generator import TextGenerator

from retrieve.cosine_similarity import CRVRetriever
from utils import set_seed, logger, CustomTransformerLoader

logger = logger()


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

    loader = CustomTransformerLoader()

    model, tokenizer = loader.load_model(
        model_path=model_path, tokenizer_path=tokenizer_path, hf_token=hf_token
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
    model.model.set_crv(best_crv, layer_idx=1, crv_layers=crv_layers)

    text_generator = TextGenerator(model, tokenizer)
    generated_text = text_generator.generate_text(
        "Once upon a time", output_file="data/results.csv"
    )

    print(generated_text)


if __name__ == "__main__":

    main()
