from audioop import cross
from http.client import responses

from accelerate import Accelerator, accelerator
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from sympy.categories import Object
from torch.nn import functional as F, CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    LlamaConfig,
)
from transformers.cache_utils import Cache
from transformers import TextStreamer

from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from base import (
    logger,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaAttention,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaModel,
    LLAMA_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


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

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        cross_attend: Optional[bool] = False,
        attention_option: str = "default",
    ):
        super().__init__(config, layer_idx)
        self.attention_option = attention_option
        self.cross_attend = cross_attend
        if self.cross_attend:
            filename = "data/crvs.pt"
            loaded_crvs = torch.load(filename)
            # print("loaded crvs from moc layer: ", len(loaded_crvs), loaded_crvs.shape)
            self.layer_crv = loaded_crvs[self.layer_idx]

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

        if self.cross_attend:
            print("concating ... ")
            # print("hidden_states.device: ", hidden_states.device, hidden_states.shape)
            self.layer_crv = self.layer_crv.to(hidden_states.device)
            # print(
            #     "self.layer_crv.device: ", self.layer_crv.device, self.layer_crv.shape
            # )
            hidden_states = torch.cat([hidden_states, self.layer_crv], dim=1)
        # print("original hidden state", len(hidden_states), hidden_states.shape)

        # get qlen and klen and handling self- or cross-attend
        bsz, q_len, _ = hidden_states.size()
        v_len = q_len
        query_states = self.q_proj(hidden_states)

        if not self.cross_attend:
            k_len = q_len
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            # filename = "data/crvs.pt"
            # loaded_crvs = torch.load(filename)
            # print("loaded crvs from moc layer: ", len(loaded_crvs), loaded_crvs.shape)
            # self.layer_crv = loaded_crvs[self.layer_idx]
            # self.layer_crv = hidden_states  # todo: remove this, just for testing to see if the self-attention works
            # with the modified rotary emb

            _, k_len, _ = self.layer_crv.size()
            key_states = self.k_proj(self.layer_crv)  # use crv
            value_states = self.v_proj(self.layer_crv)  # use crv
            # value_states = self.v_proj(
            #     hidden_states
            # )  # use hidden state # todo: remove this as well

            _, v_len, _ = self.layer_crv.size()

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if not self.cross_attend:
            # cos, sin = self.rotary_emb(
            #     value_states, position_ids
            # )  # original implementation

            cos, sin = self.rotary_emb(
                value_states, seq_len=value_states.shape[-2]
            )  # modified
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        else:
            cos_q, sin_q = self.rotary_emb(query_states, seq_len=query_states.shape[-2])
            cos_k, sin_k = self.rotary_emb(key_states, seq_len=key_states.shape[-2])

            query_states, _ = apply_rotary_pos_emb(
                query_states, query_states, cos_q, sin_q
            )
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos_k, sin_k)

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

        # following options are exclusively for tesing the implementation if it works or not
        if self.attention_option == "self_appended":
            # print("using self_append option")
            # Append self-attention vectors to themselves
            key_states = torch.cat([key_states, key_states], dim=2)
            value_states = torch.cat([value_states, value_states], dim=2)
        elif self.attention_option == "self_with_average":
            # print("using self_with_average option")
            # Append average of consecutive self-attention vectors
            avg_key = (key_states[:, :, :-1] + key_states[:, :, 1:]) / 2
            avg_value = (value_states[:, :, :-1] + value_states[:, :, 1:]) / 2
            key_states = torch.cat([key_states, avg_key, key_states[:, :, -1:]], dim=2)
            value_states = torch.cat(
                [value_states, avg_value, value_states[:, :, -1:]], dim=2
            )

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


class MoCLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, cross_attend=False):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.config = config
        self.cross_attend = cross_attend
        self.replace_layers(attention_option="default")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {
                "input_ids": input_ids.contiguous()
            }  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def replace_layers(
        self,
        attention_option: str = "default",
        layer_object: Object = None,
        layer_idx: Union[int, list] = None,
    ):

        for i in range(self.config.num_hidden_layers):
            self.model.layers[i].self_attn = MoCSdpaAttention(
                self.config,
                layer_idx=i,
                cross_attend=False,
                attention_option=attention_option,
            )

        idx = 10
        self.model.layers[idx].self_attn = MoCSdpaAttention(
            self.config,
            layer_idx=idx,
            cross_attend=False,
            attention_option=attention_option,
        )
        return


class MoCLlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.replace_layers(attention_option="default")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def replace_layers(
        self,
        attention_option: str = "default",
        layer_object: Object = None,
        layer_idx: Union[int, list] = None,
    ):

        for i in range(self.config.num_hidden_layers):
            self.transformer.layers[i].self_attn = MoCSdpaAttention(
                self.config,
                layer_idx=i,
                cross_attend=False,
                attention_option=attention_option,
            )

        idx = 10
        self.transformer.layers[idx].self_attn = MoCSdpaAttention(
            self.config,
            layer_idx=idx,
            cross_attend=False,
            attention_option=attention_option,
        )
        return


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


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    num_return_sequences=1,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=1,
    cross_attend=False,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Create a TextStreamer object
    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    # test:
    #
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a pirate chatbot who always responds in pirate speak!",
    #     },
    #     {"role": "user", "content": "Who are you?"},
    # ]
    #
    # input_ids = tokenizer.apply_chat_template(
    #     messages, add_generation_prompt=True, return_tensors="pt"
    # ).to(model.device)
    #
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    # Generate text using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            streamer=streamer,
        )

    # outputs = model.generate(
    #     input_ids,
    #     max_new_tokens=256,
    #     eos_token_id=terminators,
    #     do_sample=True,
    #     temperature=0.6,
    #     top_p=0.9,
    # )
    response = outputs[0][input_ids.shape[-1] :]
    # print(tokenizer.decode(response, skip_special_tokens=True))
    generated_text = responses
    # Decode the generated token IDs back to text
    # print("outputs: ", outputs)
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

    # try:
    #     # Generate text using the model
    #     with torch.no_grad():
    #         outputs = model.generate(
    #             input_ids,
    #             max_new_tokens=max_length,
    #             temperature=temperature,
    #             top_k=top_k,
    #             top_p=top_p,
    #             do_sample=True,
    #             num_return_sequences=1,
    #             eos_token_id=terminators,
    #         )
    #
    #     response = outputs[0][input_ids.shape[-1] :]
    #     generated_text = tokenizer.decode(response, skip_special_tokens=True)
    #     print("Generated text:", generated_text)
    #     return generated_text

    # except RuntimeError as e:
    #     print(f"Runtime Error: {e}")
    #     print("Model config:")
    #     print(f"Hidden size: {model.config.hidden_size}")
    #     print(f"Num attention heads: {model.config.num_attention_heads}")
    #     print(
    #         f"Head dim: {model.config.hidden_size // model.config.num_attention_heads}"
    #     )
    #     raise


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
