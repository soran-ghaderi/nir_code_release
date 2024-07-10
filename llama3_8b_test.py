# import os
#
# # import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer
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
# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token_s)
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token_s)

import transformers
import torch

from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, token=token_s)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, token=token_s)

prompt = """hello world!"""
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
# Generate
generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=50)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output)