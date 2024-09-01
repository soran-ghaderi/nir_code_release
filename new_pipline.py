from transformers import AutoConfig
import configs
from controller.memory_manager import MemoryManager
from data_processor.data_loader import GSM8KDataset
from generator.crv_generator import CRVGenerator
from generator.text_generator import TextGenerator

from retrieve.cosine_similarity import CRVRetriever
from retrieve.dnc import DNMemory
from utils import set_seed, logger
from utils.loading_model import CustomTransformerLoader

from rich import print
from rich.console import Console

console = Console()

logger = logger()


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


from rich.console import Console


def main():
    console = Console()
    seed = 42
    set_seed(seed)

    model_urls = {
        "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    }
    model_path = model_urls["llama31"]
    tokenizer_path = model_path
    hf_token = "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"
    config = AutoConfig.from_pretrained(model_path, use_auth_token=hf_token)

    console.rule("[bold red]Loading the Model")

    loader = CustomTransformerLoader()

    model, tokenizer = loader.load_model(
        model_path=model_path, tokenizer_path=tokenizer_path, hf_token=hf_token
    )

    crv_layers = configs.CRV_LAYERS

    print(":warning: model type: ", type(model))
    print("config.hidden_size: ", config.num_hidden_layers)
    print("config._attn_implementation: ", config._attn_implementation)

    console.rule("[bold red]Loading the dataset and generating the CRVs")

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
    # query = """Rectilinear grid does not allow Sequences as inputs ### Describe the bug, what's wrong, and what you expected. Rectilinear grid gives an error when `Sequence`s are passed in, but `ndarray` are ok. ### Steps to reproduce the bug. This doesn't work ```python import pyvista as pv pv.RectilinearGrid([0, 1], [0, 1], [0, 1]) ``` This works ```py import pyvista as pv import numpy as np pv.RectilinearGrid(np.ndarray([0, 1]), np.ndarray([0, 1]), np.ndarray([0, 1])) ``` ### System Information ```shell -------------------------------------------------------------------------------- Date: Wed Apr 19 20:15:10 2023 UTC OS : Linux CPU(s) : 2 Machine : x86_64 Architecture : 64bit Environment : IPython GPU Vendor : Mesa/X.org GPU Renderer : llvmpipe (LLVM 11.0.1, 256 bits) GPU Version : 4.5 (Core Profile) Mesa 20.3.5 Python 3.11.2 (main, Mar 23 2023, 17:12:29) [GCC 10.2.1 20210110] pyvista : 0.38.5 vtk : 9.2.6 numpy : 1.24.2 imageio : 2.27.0 scooby : 0.7.1 pooch : v1.7.0 matplotlib : 3.7.1 IPython : 8.12.0 -------------------------------------------------------------------------------- ``` ### Screenshots _No response_"""
    # query = """write a python code to print hello world."""
    query = """Provide relevant context to solve the following programming problem: obtains the id of an object and returns it as a string . if cancreate is true it will try to create a new id for the object if it has none . concode_field_sep Logger LOG concode_elem_sep Class MYCLASS concode_elem_sep String id concode_field_sep String readObjectID concode_elem_sep String createObjectID concode_elem_sep String generateID concode_elem_sep String toString. Relevant context:"""

    console.rule("[bold red]Retrieving the best CRV")

    memory_manager = MemoryManager(model, max_memories=5)

    # Input query
    retriever = CRVRetriever(
        model, tokenizer, crv_layers, max_length=configs.MAX_LENGTH
    )
    print("data loaded")

    # best_crv = retriever(query, crvs_file)
    best_crv, best_seq_length, similarities = retriever(query, crvs_file)

    # todo: not that we have the similarities, based on them, create a new router to decide which contexts to append

    print("best_crv.shape: ", best_crv.shape)
    print("best_seq_length: ", best_seq_length)

    sliced_best_crv = best_crv[:, :best_seq_length, :]
    print("sliced_best_crv.shape: ", sliced_best_crv.shape)

    layer_idx = 10
    memory_manager.add_memory(
        best_crv, best_seq_length, layer_idx=layer_idx, crv_layers=crv_layers
    )

    console.rule(f"[bold red]Concat the CRV and the hidden state at layer {layer_idx}")

    memory_manager.set_concat_positions(0, start_pos=0, end_pos=best_seq_length)
    # memory_manager.apply_memory_to_model(0)

    # Set the CRV in the model (e.g., integrate at layer 1)
    # model.model.set_crv(sliced_best_crv, layer_idx=5, crv_layers=crv_layers)
    # model.model.set_post_concat_crv(True)

    console.rule(f"[bold red]Generating the outputs")

    text_generator = TextGenerator(model, tokenizer)
    generated_text = text_generator.generate_text(
        query,
        max_new_tokens=500,
        output_file="data/results.csv",
        stop_sequences=["The end", ".\n\n"],
    )


if __name__ == "__main__":

    main()
