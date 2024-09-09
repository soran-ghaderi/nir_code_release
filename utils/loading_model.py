from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer

from moc_layers import LlamaForCausalLM


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

        # Set pad_token_id to eos_token_id
        self.model.model.config.pad_token_id = self.model.model.config.eos_token_id
        self.model.eval()  # Set the model to evaluation mode

        return self.model, self.tokenizer

    def get_model(self) -> Optional[LlamaForCausalLM]:
        return self.model

    def get_tokenizer(self) -> Optional[AutoTokenizer]:
        return self.tokenizer
