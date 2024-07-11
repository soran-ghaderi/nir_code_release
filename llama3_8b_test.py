# import os
#
# # import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# import torch
#
# os.environ['HUGGINGFACE_HUB_TOKEN'] = "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"
model_name = "meta-llama/Meta-Llama-3-8B"
#
token_s = "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"
#
# # pipeline = transformers.pipeline(
# #     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# # )
# # pipeline("Hey how are you doing today?")
#
#
# # model_name = "meta-llama/Meta-Llama-2-7B"
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token_s)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token_s)

import transformers
import torch

from transformers import AutoTokenizer, LlamaForCausalLM

# model = LlamaForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=model_name, token=token_s
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     pretrained_model_name_or_path=model_name, token=token_s
# )
#
# prompt = """hello world!"""
# inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
# # Generate
# generate_ids = model.generate(
#     inputs.input_ids, attention_mask=inputs.attention_mask, max_length=50
# )
# output = tokenizer.batch_decode(
#     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )[0]
#
# print(output)
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json

# import matplotlib.pyplot as plt

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_4|>",
    "<|eot_id|>",  # end of turn
] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
# mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
# tokenizer = tiktoken.Encoding(
#     name=Path(tokenizer_path).name,
#     pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
#     mergeable_ranks=mergeable_ranks,
#     special_tokens={
#         token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)
#     },
# )

special_tokens_dict = {"additional_special_tokens": special_tokens}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# Print the number of special tokens added
print(f"Number of special tokens added: {num_added_toks}")
encoded_input = tokenizer.encode("hello world!", add_special_tokens=False)
decoded_output = tokenizer.decode(encoded_input)
print(f"Encoded input: {encoded_input}")
print(f"Decoded output: {decoded_output}")

state_dict = model.state_dict()

# Print the first 20 layer names
print("dict: ", json.dumps(list(state_dict.keys())[:20], indent=4))
print(state_dict.keys())
prompt = "once upon a time"
