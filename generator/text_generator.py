import csv
from typing import Optional, Union, List

import torch
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from utils import logger

logger = logger()


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops=None, encounters=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops or []
        self.encounters = encounters
        self.count = 0

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0][-1:])
        if any(stop in decoded for stop in self.stops):
            self.count += 1
            if self.count >= self.encounters:
                return True
        return False


class TextGenerator:
    def __init__(self, model, tokenizer, seed=None, device=None, streamer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.streamer = streamer
        self.device = device

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
        min_p: float = 0.05,
        dry_run=True,
        dry_multiplier=0.8,
        dry_base=0.1,
        dry_allowed_length=32,
        dry_sequence_breakers='"\\n", ":", "\\"", "*"',
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 1,
        crv_layer_idx: Optional[Union[int, list]] = None,
        output_file: Optional[str] = None,
        use_eos_token: bool = True,
        stop_sequences: Optional[List[str]] = None,
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
        if not self.seed is None:
            self.set_seed(self.seed)

        if not self.device is None:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.device
            )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                self.model.device
            )

        # def update_display(text):
        #     return Panel(text, title="Generated Text", border_style="cyan")

        if stop_sequences:
            stopping_criteria = StoppingCriteriaList(
                [CustomStoppingCriteria(tokenizer=self.tokenizer, stops=stop_sequences)]
            )
        else:
            stopping_criteria = None
        generate_kwargs = {
            "input_ids": input_ids,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            # "top_k": top_k,
            # "top_p": top_p,
            # "min_p": min_p,
            # "dry_run": dry_run,
            # "dry_multiplier": dry_multiplier,
            # "dry_base": dry_base,
            # "dry_allowed_length": dry_allowed_length,
            # "dry_sequence_breakers": dry_sequence_breakers,
            "do_sample": True,
            "num_return_sequences": num_return_sequences,
            # "repetition_penalty": repetition_penalty,
            # "no_repeat_ngram_size": no_repeat_ngram_size,
            "streamer": self.streamer,
            "stopping_criteria": stopping_criteria,
        }

        eos_token_id = self.tokenizer.eos_token_id
        if use_eos_token:
            generate_kwargs["eos_token_id"] = eos_token_id

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

    def generate_text_batch(self, prompts: List[str], **kwargs):
        if not self.seed is None:
            self.set_seed(self.seed)

        input_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        generate_kwargs = {
            "input_ids": input_ids.input_ids,
            "attention_mask": input_ids.attention_mask,
            "max_length": kwargs.get("max_length", 50),
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
            "temperature": kwargs.get("temperature", 0.8),
            "do_sample": kwargs.get("do_sample", True),
            "num_return_sequences": 1,
        }

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        generated_texts = [
            self.tokenizer.decode(
                output[input_ids.input_ids.shape[1] :], skip_special_tokens=True
            )
            for output in outputs
        ]

        return generated_texts
