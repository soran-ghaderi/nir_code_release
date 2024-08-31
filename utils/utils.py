import logging
import random
from typing import Optional, Tuple

import numpy as np
import torch
from rich.logging import RichHandler
from transformers import AutoTokenizer


import configs
from moc_layers import LlamaForCausalLM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logger():
    if configs.USE_RICH:

        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
        logger = logging.getLogger("rich")
    else:

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    return logger


class CustomTransformerLoader:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self.tokenizer = None

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(
        self,
        model_path: str,
        tokenizer_path: str,
        hf_token: Optional[str] = None,
        load_in_8bit: bool = False,
    ) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
        """
        Load a custom transformer model and tokenizer.
        """
        self.set_seed(self.seed)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_auth_token=hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            use_auth_token=hf_token,
            device_map="auto",
            # load_in_8bit=load_in_8bit,
            # torch_dtype=torch.float16,
        )
        self.model.eval()  # Set the model to evaluation mode

        return self.model, self.tokenizer

    def get_model(self) -> Optional[LlamaForCausalLM]:
        return self.model

    def get_tokenizer(self) -> Optional[AutoTokenizer]:
        return self.tokenizer
