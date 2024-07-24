from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.cache_utils import Cache

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from base import (
    logger,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaAttention,
)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class ModifiedLlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float).to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        position_ids = torch.arange(seq_len, device=x.device)
        inv_freq_expanded = self.inv_freq.unsqueeze(0)  # [1, dim/2]
        position_ids_expanded = position_ids.unsqueeze(1).float()  # [seq_len, 1]

        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.matmul(
                position_ids_expanded, inv_freq_expanded
            )  # [seq_len, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
            cos = emb.cos()
            sin = emb.sin()

        return cos.unsqueeze(0).to(dtype=x.dtype), sin.unsqueeze(0).to(dtype=x.dtype)


# LLAMA_ATTENTION_CLASSES = {
#     "eager": LlamaAttention,
#     "sdpa": LlamaSdpaAttention,
#     "moc":
# }


class MoCSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        # loading crvs
        filename = "data/crvs.pt"
        loaded_crvs = torch.load(filename)
        print("loaded crvs from moc layer: ", len(loaded_crvs), loaded_crvs.shape)
        layer_crv = loaded_crvs[self.layer_idx]
        print(
            "layer crv from moc layer: ",
            self.layer_idx,
            len(layer_crv),
            layer_crv.shape,
        )
        print("original hidden state", len(hidden_states), hidden_states.shape)

        self.cross_attend = False
        # get qlen and klen and handling self- or cross-attend
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)

        if not self.cross_attend:
            k_len = q_len
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            _, k_len, _ = layer_crv.size()
            key_states = self.k_proj(layer_crv)  # use crv
            value_states = self.v_proj(layer_crv)  # use crv

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, k_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, k_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if not self.cross_attend:
            cos, sin = self.rotary_emb(
                value_states, position_ids
            )  # original implementation
            query_states, key_states = apply_rotary_pos_emb(
                query_states, query_states, cos, sin
            )
        else:
            cos_q, sin_q = self.rotary_emb(query_states, seq_len=query_states.shape[-2])
            cos_k, sin_k = self.rotary_emb(key_states, seq_len=key_states.shape[-2])

            query_states, _ = apply_rotary_pos_emb(
                query_states, query_states, cos_q, sin_q
            )
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos_k, sin_k)

        # fixme: RuntimeError: The size of tensor a (8) must match the size of tensor b (32) at non-singleton dimension 1
        # query_states, key_states = apply_rotary_pos_emb(
        #     query_states, key_states, cos, sin, position_ids=position_ids
        # ) # original implementation

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if not self.cross_attend:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }  # original
            else:
                cache_kwargs = {
                    "sin_q": sin_q,
                    "cos_q": cos_q,
                    "sin_k": sin_k,
                    "cos_k": cos_k,
                    "cache_position": cache_position,
                }  # modified separated q and k
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def load_custom_transformer(pretrained_model_path, tokenizer_path, hf_token=None):
    # Load the configuration
    # config = LlamaConfig.from_pretrained(pretrained_model_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path, use_auth_token=hf_token
    )

    model.eval()  # Set the model to evaluation mode

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    cross_attend=False,
):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
        )

    # Decode the generated token IDs back to text
    # print("outputs: ", outputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# class CRVExtractionPipeline(TokenClassificationPipeline):
#     def __init__(self, model, *args, **kwargs):
#         super().__init__(
#             model=AutoModelForTokenClassification.from_pretrained(model_path),
#             tokenizer=AutoTokenizer.from_pretrained(model),
#             *args,
#             **kwargs,
#         )
#
#     def _forward(self, model_inputs):
#         # Forward
#         special_tokens_mask = model_inputs.pop("special_tokens_mask")
#         offset_mapping = model_inputs.pop("offset_mapping", None)
#         sentence = model_inputs.pop("sentence")
#
#         outputs = self.model(**model_inputs)
#         logits = outputs[0]
#
#         hidden_state = outputs[1]
#
#         return {
#             "logits": logits,
#             "special_tokens_mask": special_tokens_mask,
#             "offset_mapping": offset_mapping,
#             "sentence": sentence,
#             "hidden_state": hidden_state,  # Add hidden state to the returned dictionary
#             **model_inputs,
#         }
#
#     def postprocess(self, model_outputs):
#         results = super().postprocess(
#             model_outputs=model_outputs,
#             aggregation_strategy=AggregationStrategy.SIMPLE,
#         )
#         return {
#             "keywords": np.unique([result.get("word").strip() for result in results]),
#             "hidden_state": model_outputs["hidden_state"],
#         }
