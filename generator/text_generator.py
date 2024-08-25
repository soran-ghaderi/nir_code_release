import csv
from typing import Optional, Union

import torch
from transformers import TextStreamer


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
        crv_layer_idx: Optional[Union[int, list]] = None,
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
