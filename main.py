import torch
from transformers import AutoConfig
from transformers.models.paligemma.convert_paligemma_weights_to_hf import device

from llama3_8b_test import (
    load_custom_transformer,
    MoCSdpaAttention,
    generate_text,
)
from base import LlamaSdpaAttention


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
    prompt = "introduce yourself."
    # prompt = (
    #     "The Apollo 11 mission was the first manned mission to land on the Moon. Launched by NASA on July 16, "
    #     "1969, it carried astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. On July 20, 1969, Neil "
    #     "Armstrong and Buzz Aldrin landed the lunar module Eagle on the Moon while Michael Collins remained in "
    #     "lunar orbit in the command module Columbia. Armstrong became the first person to step onto the lunar "
    #     "surface, followed by Aldrin. They spent about two and a quarter hours outside the spacecraft, "
    #     "collecting samples and conducting experiments. The mission returned to Earth on July 24, 1969."
    # )
    prompt = "Who were the astronauts involved in the Apollo 11 mission and what were their roles?"
    # Generate text
    # generated_text = generate_text(model, tokenizer, prompt, max_length=50)
    # print(f"Generated text: {generated_text}")
    print("loaded ... ")
    print("model type: ", type(model))
    print("config.hidden_size: ", config.num_hidden_layers)
    # print("num layers: ", len(model.model.layers))

    print("config._attn_implementation: ", config._attn_implementation)
    # print(model.model)

    input_ids = tokenizer(prompt, return_tensors="pt")
    generated_text = generate_text(model, tokenizer, prompt, max_length=100)
    print(f"Generated text after: {generated_text}")

    # Define hooks to capture hidden states and Q, K, V matrices
    hidden_states = []
    qkv_states = {"q": [], "k": [], "v": []}

    def hook_hidden_states(module, input, output):
        hidden_states.append(output)

    def hook_q(module, input, output):
        qkv_states["q"].append(output)

    def hook_k(module, input, output):
        qkv_states["k"].append(output)

    def hook_v(module, input, output):
        qkv_states["v"].append(output)

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
    # from transformers import pipeline
    #
    # hf_token = "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"
    # model_id = "meta-llama/Meta-Llama-3-8B"
    #
    # pipe = pipeline(
    #     "text-generation",
    #     model=model_id,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    #     token=hf_token,
    #     device=0,
    # )
    # output = pipeline(
    #     "Who were the astronauts involved in the Apollo 11 mission and what were their roles?"
    # )
    #
    # print(output)
