import csv
from http.client import responses

import torch
from transformers import AutoConfig, AutoTokenizer, TextStreamer, LlamaConfig
from transformers.models.paligemma.convert_paligemma_weights_to_hf import device

from llama3_8b_test import (
    MoCSdpaAttention,
)
from moc_layers import LlamaSdpaAttention, LlamaForCausalLM


def load_custom_transformer(
    pretrained_model_path, tokenizer_path, config, hf_token=None, cross_attend=False
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    # Load the configuration
    # config = LlamaConfig.from_pretrained(pretrained_model_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=hf_token)

    # causal:
    # model = MoCLlamaForCausalLM.from_pretrained(
    #     pretrained_model_path,
    #     use_auth_token=hf_token,
    #     device_map="auto",
    #     # load_in_8bit=True,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     pretrained_model_path,
    #     use_auth_token=hf_token,
    #     device_map="auto",
    # )
    # question answering:
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        use_auth_token=hf_token,
        device_map="auto",
        # load_in_8bit=True,
    )

    # with init_empty_weights():
    #     model = MoCLlamaForCausalLM.from_config(config)
    #
    # model = load_checkpoint_and_dispatch(
    #     model,
    #     pretrained_model_path,
    #     device_map="auto",
    #     no_split_module_classes=["LlamaDecoderLayer"]
    # )

    model.eval()  # Set the model to evaluation mode

    return model, tokenizer


def write_results_to_file(
    file_path,
    results,
    prompt,
    max_length=50,
    max_new_tokens=100,
    num_return_sequences=1,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=1,
):
    """
    Writes the results to a file in an organized and tabular format.

    Args:
        file_path (str): The path to the output file.
        results (list): A list of dictionaries containing the results.
    """
    context = (
        "The Apollo 11 mission was the first manned mission to land on the Moon. Launched by NASA on July 16, "
        "1969, it carried astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. On July 20, 1969, Neil "
        "Armstrong and Buzz Aldrin landed the lunar module Eagle on the Moon while Michael Collins remained in "
        "lunar orbit in the command module Columbia. Armstrong became the first person to step onto the lunar "
        "surface, followed by Aldrin. They spent about two and a quarter hours outside the spacecraft, "
        "collecting samples and conducting experiments. The mission returned to Earth on July 24, 1969."
    )

    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Context: {context}\n\n")
        file.write(f"Model Parameters\n")
        file.write(f"Max Length: {max_length}\n")
        file.write(f"Max New Tokens: {max_new_tokens}\n")
        file.write(f"Num Return Sequences: {num_return_sequences}\n")
        file.write(f"Temperature: {temperature}\n")
        file.write(f"Top K: {top_k}\n")
        file.write(f"Top P: {top_p}\n")
        file.write(f"Repetition Penalty: {repetition_penalty}\n")
        file.write(f"No Repeat N-gram Size: {no_repeat_ngram_size}\n\n\n")
        file.write(f"Generated text by concatenated layer index\n\n")

        # writer = csv.DictWriter(
        #     file, fieldnames=["Prompt", "Generated Output", "Concatenated Layer Index"]
        # )
        # writer.writeheader()
        for result in results:
            # writer.writerow(result)
            for key, value in result.items():
                file.write(f"{key}: {value}\n")

            file.write(f"\n{'==' * 80}\n")

    print(f"Results written to {file_path}")


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    max_new_tokens=100,
    num_return_sequences=1,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=1,
    cross_attend=False,
    config=None,
    test_layers=[],
    output_file="results/concat_different_layers.csv",
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Encode the input prompt
    if type(prompt) == list:
        input_ids = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Create a TextStreamer object
    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    # test:
    #

    #
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    results = []

    # Generate text using the model
    if len(test_layers) > 0:
        for idx in test_layers:
            print(f"concatenate at layer {idx}:")

            model.model.set_layers_to_concat(layers_to_concat=[idx])
            model.model.set_is_crv_concatenated(is_crv_concatenated=False)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    streamer=streamer,
                )

            generated_text = outputs[0][input_ids.shape[-1] :]
            decoded_text = tokenizer.decode(generated_text, skip_special_tokens=True)

            results.append(
                {
                    "Layer idx": idx,
                    "Generated output": decoded_text,
                }
            )
        write_results_to_file(
            output_file,
            results,
            prompt,
            max_length=50,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=1,
        )
    else:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streamer=streamer,
            )

        generated_text = outputs[0][input_ids.shape[-1] :]
        decoded_text = tokenizer.decode(generated_text, skip_special_tokens=True)

    return decoded_text


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Path to the pretrained model and tokenizer
    # hf_model_path = "meta-llama/Meta-Llama-3-8B"
    hf_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    # hf_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # hf_tokenizer_path = "meta-llama/Meta-Llama-3-8B"
    hf_tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    # hf_tokenizer_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    hf_token = "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"
    config = AutoConfig.from_pretrained(hf_model_path, use_auth_token=hf_token)
    print(config)
    # Load the custom transformer model and tokenizer

    # todo: try to add crvs to the k and v instead of cross attention
    # todo: try to concat them instead of cross attention if it didn't work
    model, tokenizer = load_custom_transformer(
        hf_model_path, hf_tokenizer_path, config=config, hf_token=hf_token
    )

    # Define the input prompt

    instruction_prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant who always responds with respect to the given information!",
        },
        {
            "role": "user",
            "content": "write something",
        },
    ]
    prompt = "introduce yourself."
    # prompt = (
    #     "The Apollo 11 mission was the first manned mission to land on the Moon. Launched by NASA on July 16, "
    #     "1969, it carried astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. On July 20, 1969, Neil "
    #     "Armstrong and Buzz Aldrin landed the lunar module Eagle on the Moon while Michael Collins remained in "
    #     "lunar orbit in the command module Columbia. Armstrong became the first person to step onto the lunar "
    #     "surface, followed by Aldrin. They spent about two and a quarter hours outside the spacecraft, "
    #     "collecting samples and conducting experiments. The mission returned to Earth on July 24, 1969."
    # )
    prompt = "Who were the astronauts involved in the Apollo 11?"
    prompt = "elaborate more."
    # Generate text
    # generated_text = generate_text(model, tokenizer, prompt, max_length=50)
    # print(f"Generated text: {generated_text}")
    print("loaded ... ")
    print("model type: ", type(model))
    print("config.hidden_size: ", config.num_hidden_layers)
    # print("num layers: ", len(model.model.layers))

    print("config._attn_implementation: ", config._attn_implementation)
    # print(model.model)

    # input_ids = tokenizer(prompt, return_tensors="pt")
    generated_text = generate_text(
        model,
        tokenizer,
        prompt=prompt,
        max_length=100,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=1.0,
        top_k=1,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        cross_attend=False,
        config=config,
        test_layers=[2],
    )

    # print(f"Generated text after: {generated_text}")

    # Define hooks to capture hidden states and Q, K, V matrices
    # hidden_states = []
    # qkv_states = {"q": [], "k": [], "v": []}
    #
    # def hook_hidden_states(module, input, output):
    #     hidden_states.append(output)
    #
    # def hook_q(module, input, output):
    #     qkv_states["q"].append(output)
    #
    # def hook_k(module, input, output):
    #     qkv_states["k"].append(output)
    #
    # def hook_v(module, input, output):
    #     qkv_states["v"].append(output)

    # Register hooks to the q_proj, k_proj, and v_proj layers of each decoder layer
    # for layer in model.model.layers:
    #     layer.register_forward_hook(hook_hidden_states)
    #     layer.self_attn.q_proj.register_forward_hook(hook_q)
    #     layer.self_attn.k_proj.register_forward_hook(hook_k)
    #     layer.self_attn.v_proj.register_forward_hook(hook_v)

    # Run the sample input through the model again
    # outputs = model(**input_ids, output_hidden_states=True)
    #
    # logits = outputs.logits
    # predicted_ids = torch.argmax(logits, dim=-1)
    # predicted_text = tokenizer.decode(predicted_ids[0])
    # print("Predicted text:", predicted_text)

    # The qkv_states dictionary now contains the query, key, and value matrices
    # print(
    #     "Query states shape:", len(qkv_states["q"]), [q.shape for q in qkv_states["q"]]
    # )
    # print("Key states shape:", len(qkv_states["k"]), [k.shape for k in qkv_states["k"]])
    # print(
    #     "Value states shape:", len(qkv_states["v"]), [v.shape for v in qkv_states["v"]]
    # )
    # print("Shape of hidden states tensor:", crvs.shape)


if __name__ == "__main__":

    main()
