import copy
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import NLLLoss
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5ForConditionalGeneration,
    T5LayerNorm,
    T5PreTrainedModel,
    T5Stack,
    __HEAD_MASK_WARNING_MSG
)
from transformers.utils import logging

from ..twt_generation_utils import TWTGenerationMixin

logger = logging.get_logger(__name__)


@dataclass
class TWTModelOutput(BaseModelOutput):
    input_ids_output: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TWTModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    cross_attention_outputs: Optional[Tuple[torch.FloatTensor]] = None
    embedding_output: Optional[Tuple[torch.FloatTensor]] = None
    input_ids_output: Optional[Tuple[torch.FloatTensor]] = None


# TWTT5Block is mainly from the forward method in T5Block, return cross_attention_output
class TWTT5Block(T5Block):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values,
                "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            if torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # !!! Save cross attention output
            cross_attention_output = hidden_states.clone()

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        # !!! Return the attention output (context), shape: (batch_size, output_seq_length, hidden_size), e.g. (1, 128, 768)
        if self.is_decoder:
            outputs = outputs + (cross_attention_output,)
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


# TWTT5PreTrainedModel
# _init_weights is mainly from the _init_weights method in T5PreTrainedModel
# init weight for column and row embeddings
class TWTT5PreTrainedModel(TWTGenerationMixin, T5PreTrainedModel):
    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, TWTT5ForConditionalGeneration):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            module.shared_types.weight.data.normal_(mean=0.0, std=factor * 1.0)
            module.shared_rows.weight.data.normal_(mean=0.0, std=factor * 1.0)
            module.shared_cols.weight.data.normal_(mean=0.0, std=factor * 1.0)
        else:
            T5PreTrainedModel._init_weights(self, module)


# TWTT5Stack is mainly from T5Stack, add col, row, type embedding
class TWTT5Stack(TWTT5PreTrainedModel, T5Stack):
    def __init__(self, config, embed_tokens=None, embed_types=None, embed_rows=None, embed_cols=None):
        PreTrainedModel.__init__(self, config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        # !!! Add type, row, column embeddings
        if not self.is_decoder:
            self.embed_types = embed_types
            self.embed_rows = embed_rows
            self.embed_cols = embed_cols

        self.block = nn.ModuleList(
            [TWTT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        raise NotImplementedError

    def deparallelize(self):
        raise NotImplementedError

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        type_ids=None,
        row_ids=None,
        col_ids=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
            if not self.is_decoder:
                self.embed_types = self.embed_types.to(self.first_device)
                self.embed_rows = self.embed_rows.to(self.first_device)
                self.embed_cols = self.embed_cols.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # !!! Clone input_ids
        input_ids_output = input_ids.clone()

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            # !!! Add type, row, column embeddings
            token_embeds = self.embed_tokens(input_ids)
            inputs_embeds = token_embeds
            if type_ids is not None:
                assert self.embed_types is not None, "You have to initialize the model with valid type embeddings"
                type_embeds = self.embed_types(type_ids)
                inputs_embeds = inputs_embeds + type_embeds

            if row_ids is not None:
                assert self.embed_rows is not None, "You have to initialize the model with valid row embeddings"
                row_embeds = self.embed_rows(row_ids)
                inputs_embeds = inputs_embeds + row_embeds

            if col_ids is not None:
                assert self.embed_cols is not None, "You have to initialize the model with valid column embeddings"
                col_embeds = self.embed_cols(col_ids)
                inputs_embeds = inputs_embeds + col_embeds

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        all_cross_attention_outputs = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if encoder_layer_head_mask is not None:
                    encoder_layer_head_mask = encoder_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                encoder_layer_head_mask=encoder_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
                    all_cross_attention_outputs = all_cross_attention_outputs + (layer_outputs[6],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    all_cross_attention_outputs,
                    inputs_embeds,
                    input_ids_output,
                ]
                if v is not None
            )
        return TWTModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            cross_attention_outputs=all_cross_attention_outputs,
            embedding_output=inputs_embeds,
            input_ids_output=input_ids_output
        )


class TWTT5ForConditionalGeneration(TWTT5PreTrainedModel, T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"encoder\.embed_types\.weight",
        r"encoder\.embed_rows\.weight",
        r"encoder\.embed_cols\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]

    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        # !!! Add type, row, column embeddings
        self.shared_types = nn.Embedding(config.type_vocab_size, config.d_model)
        self.shared_rows = nn.Embedding(config.max_row_embeddings, config.d_model)
        self.shared_cols = nn.Embedding(config.max_col_embeddings, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = TWTT5Stack(encoder_config, self.shared, self.shared_types, self.shared_rows, self.shared_cols)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TWTT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # !!! Add copy layer
        self.p_copy_linear = nn.Linear(self.config.hidden_size * 3, 1)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    # !!! Add type_ids, row_ids, col_ids, cross_attention_mask, decoder_copy_mask
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        type_ids=None,
        row_ids=None,
        col_ids=None,
        decoder_copy_mask=None,
        cross_attention_mask=None,
        decoder_prefix_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            # !!! Add type_ids, row_ids, col_ids
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                type_ids=type_ids,
                row_ids=row_ids,
                col_ids=col_ids,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = TWTModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                input_ids_output=encoder_outputs['input_ids_output'] if 'input_ids_output' in encoder_outputs else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            # !!! Shift decoder attention mask
            if decoder_attention_mask is not None:
                shifted_decoder_attention_mask = decoder_attention_mask.new_zeros(decoder_attention_mask.shape)
                shifted_decoder_attention_mask[..., 1:] = decoder_attention_mask[..., :-1].clone()
                shifted_decoder_attention_mask[..., 0] = 1
                decoder_attention_mask = shifted_decoder_attention_mask

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # !!! Check cross attention mask
        if hasattr(self.config, 'no_cross_attention_mask') and self.config.no_cross_attention_mask:
            cross_attention_mask = attention_mask
        else:
            if cross_attention_mask is None:
                cross_attention_mask = attention_mask

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
            # !!! Add cross attention mask
            if cross_attention_mask is not None:
                cross_attention_mask = cross_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=cross_attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # !!! Embedding output of the decoder input, shape: (batch_size, output_seq_length, hidden_size), e.g. (1, 128, 768)
        embedding_output = decoder_outputs.embedding_output
        # !!! Cross attention of the last decoder layer, shape: (batch_size, num_attention_heads, output_seq_length, input_seq_length), e.g. (1, 12, 128, 512)
        last_cross_attention = decoder_outputs.cross_attentions[-1]
        # !!! Attention output (context) of the last decoder layer, shape: (batch_size, output_seq_length, hidden_size), e.g. (1, 128, 768)
        last_cross_attention_output = decoder_outputs.cross_attention_outputs[-1]
        # !!! Hidden state of the last decoder layer, shape: (batch_size, output_seq_length, hidden_size), e.g. (1, 128, 768)
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
            # !!! Follow model parallel
            last_cross_attention_output = last_cross_attention_output.to(self.lm_head.weight.device)
            last_cross_attention = last_cross_attention.to(self.lm_head.weight.device)
            embedding_output = embedding_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        # !!! Add prefix mask
        if decoder_prefix_mask is not None:
            decoder_prefix_mask = decoder_prefix_mask.to(lm_logits.device)
            lm_logits[0][-1] += -1e4 * (1 - decoder_prefix_mask[0].cuda())

        # !!! Check no_copy config
        if not (hasattr(self.config, 'no_copy') and self.config.no_copy):
            # !!! Calculate p_copy based on Few-Shot NLG
            p_copy_input = torch.cat([last_cross_attention_output, sequence_output, embedding_output], dim=-1)
            p_copy = self.p_copy_linear(p_copy_input)
            p_copy = torch.sigmoid(p_copy)
            p_gen = 1 - p_copy
            # !!! Calculate prediction scores with p_gen
            lm_scores = p_gen * F.softmax(lm_logits, dim=-1)
            # !!! Take average of attention scores from 12 heads, dim=1 is the head dimention
            last_cross_attention_mean = p_copy * torch.mean(last_cross_attention, dim=1)
            # !!! Expand input_ids to (1, 128, 512)
            encoder_input_ids = encoder_outputs['input_ids_output'].unsqueeze(1).expand(last_cross_attention_mean.size(0), last_cross_attention_mean.size(1), -1)
            # !!! Calculate final prediction scores with input ids
            lm_scores.scatter_add_(2, encoder_input_ids, last_cross_attention_mean)
            lm_scores = lm_scores + 1e-8
        else:
            lm_scores = F.softmax(lm_logits, dim=-1)

        loss = None
        if labels is not None:
            loss_fct = NLLLoss(ignore_index=-100)
            loss = loss_fct(torch.log(lm_scores.view(-1, lm_scores.size(-1))), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            # !!! Check no_copy config
            if not (hasattr(self.config, 'no_copy') and self.config.no_copy):
                if not (hasattr(self.config, 'gen_loss_rate') and self.config.gen_loss_rate == 0):
                    # Save copy score for each example
                    copy_gate_scores = torch.zeros([decoder_input_ids.size(0), 1]).to(labels.device)
                    for batch, dec_labels in enumerate(labels):
                        # !!! Find same input_ids between decoder input and encoder input
                        # enc_dec_same_tokens = (dec_labels[:,None]==input_ids[batch][None,:])*decoder_attention_mask[batch][:,None]
                        # enc_dec_same_tokens = torch.nonzero(enc_dec_same_tokens)
                        # !!! Get unique positions from the decoder inputs, 0 is the decoder input position
                        # dec_token_matched_positions = torch.unique(enc_dec_same_tokens[:, 0])
                        dec_token_matched_positions = torch.nonzero(decoder_copy_mask[batch] == 1).squeeze(1)
                        same_token_copy_score = torch.sum(p_gen[batch][dec_token_matched_positions, :])
                        # !!! Save current copy gate score
                        copy_gate_scores[batch, 0] = same_token_copy_score
                    loss = loss + (self.config.gen_loss_rate if hasattr(self.config, 'gen_loss_rate') else 1.0) * torch.mean(copy_gate_scores)

        if not return_dict:
            output = (lm_scores,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_scores,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cross_attention_mask": kwargs["decoder_cross_attention_mask"],
        }
