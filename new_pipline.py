import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
import configs
from data_processor.data_loader import GSM8KDataset
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

    crv_layers = configs.CRV_LAYERS

    print("model type: ", type(model))
    print("config.hidden_size: ", config.num_hidden_layers)
    print("config._attn_implementation: ", config._attn_implementation)

    dataset = GSM8KDataset(tokenizer, split="train", subset_size=configs.SUBSET_SIZE)

    crv_generator = CRVGenerator(model, tokenizer, max_length=configs.MAX_LENGTH)
    crvs_file = crv_generator.generate_crvs(
        dataset, "data/new_stack.pt", crv_layers=crv_layers
    )

    print("piple crvs_file: ", type(crvs_file))
    print("piple crvs_file: ", crvs_file[0].shape)
    print("piple crvs_file: ", crvs_file[1].shape, crvs_file[1])

    # query = "Solve the following problem: If x + y = 10 and x - y = 4, what are the values of x and y?"
    query = (
        "Problem: Ines had $20 in her purse. She bought 3 pounds of peaches, which are $2 per pound at the local "
        "farmers’ market. How much did she have left? "
        "Solution: Ines bought 3 pounds of peaches for 3 peaches * $2/peach = $<<3*2=6>>6. "
        "Ines has $20 - $6 = $<<20-6=14>>14 left. "
        "#### 14 "
    )

    query = """Problem: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
        Solution: There are 144/12 = <<144/12=12>>12 sets of 12 cans that the family collected.
        So, the family would receive $0.50 x 12 = $<<0.50*12=6>>6 for the cans."""
    # query = "this is test:"

    query = """Problem: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and twice as many as Natalie. They used their strawberries to make jam. One jar of jam used 7 strawberries and they sold each jar at $4. How much money were they able to make from the strawberries they picked?
        Solution: Matthew picked 16 + 20 = <<16+20=36>>36 strawberries.
        Natalie picked 36/2 = <<36/2=18>>18 strawberries.
        All together, they have 16 + 36 + 18 = <<16+36+18=70>>70 strawberries.
        They can make 70/7 = <<70/7=10>>10 jars of strawberries.
        They earn 10 x $4 = $<<10*4=40>>40 from the strawberries they picked.
        #### 40 """
    query = """Problem: James dumps his whole collection of 500 Legos on the floor and starts building a castle out of them.  He uses half the pieces before finishing and is told to put the rest away.  He puts all of the leftover pieces back in the box they came from, except for 5 missing pieces that he can't find.  How many Legos are in the box at the end?
        Solution"""
    query = """Problem: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?
Solution: There are 960 pages because 80 x 12 ="""

    # Input query
    retriever = CRVRetriever(
        model, tokenizer, crv_layers, max_length=configs.MAX_LENGTH
    )
    print("data loaded")

    # best_crv = retriever(query, crvs_file)
    best_crv, best_seq_length = retriever(query, crvs_file)
    print("best_crv.shape: ", best_crv.shape)
    print("best_seq_length: ", best_seq_length)

    #
    # reduced_crv = best_crv.mean(dim=-1)  # (layers/len(crv_layers), seq_len)
    # print("Reduced CRV shape:", reduced_crv.shape)
    # print("Reduced CRV:", str(reduced_crv[0]))

    sliced_best_crv = best_crv[:, :best_seq_length, :]
    print("sliced_best_crv.shape: ", sliced_best_crv.shape)

    # Set the CRV in the model (e.g., integrate at layer 1)
    model.model.set_crv(sliced_best_crv, layer_idx=1, crv_layers=crv_layers)
    model.model.set_post_concat_crv(True)
    text_generator = TextGenerator(model, tokenizer)
    generated_text = text_generator.generate_text(query, output_file="data/results.csv")

    # print(generated_text)


if __name__ == "__main__":

    main()
