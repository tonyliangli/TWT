
import torch
from torch import nn
from torch.nn import NLLLoss
from torch.nn.functional import softmax
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert_generation.modeling_bert_generation import (
    BertGenerationDecoder, BertGenerationEmbeddings, BertGenerationEncoder,
    BertGenerationOnlyLMHead)
from transformers.utils import logging

from ..t5.modeling_twt_t5 import TWTModelOutputWithPastAndCrossAttentions
from ..twt_generation_utils import TWTGenerationMixin
from .modeling_twt_bert import TWTBertEncoder

logger = logging.get_logger(__name__)

class TWTBertGenerationEmbeddings(BertGenerationEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.is_decoder = config.is_decoder
        if not self.is_decoder:
            self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.row_embeddings = nn.Embedding(config.max_row_embeddings, config.hidden_size)
            self.col_embeddings = nn.Embedding(config.max_col_embeddings, config.hidden_size)


    def forward(self, input_ids=None, position_ids=None, type_ids=None, row_ids=None, col_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        position_embeddings = self.position_embeddings(position_ids)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds + position_embeddings

        if not self.is_decoder: 
            if type_ids is None:
                type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            type_embeddings = self.type_embeddings(type_ids)

            if row_ids is None:
                row_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            row_embeddings = self.row_embeddings(row_ids)

            if col_ids is None:
                col_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            col_embeddings = self.col_embeddings(col_ids)

            embeddings = embeddings + type_embeddings + row_embeddings + col_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TWTBertGenerationEncoder(TWTGenerationMixin, BertGenerationEncoder):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.config = config
        # !!! Use TWTBertGenerationEmbeddings
        self.embeddings = TWTBertGenerationEmbeddings(config)
        self.encoder = TWTBertEncoder(config)

        self.init_weights()

    # Add type_ids, row_ids, col_ids
    def forward(
        self,
        input_ids=None,
        type_ids=None,
        row_ids=None,
        col_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # !!! Clone input_ids
        input_ids_output = input_ids.clone()

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = None
        if not use_cache:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape, device
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # !!! Add type_ids, row_ids, col_ids
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            type_ids=type_ids,
            row_ids=row_ids,
            col_ids=col_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        # !!! Add Embedding Output
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:] + (embedding_output,)

        return TWTModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            cross_attention_outputs=encoder_outputs.cross_attention_outputs,
            embedding_output=embedding_output,
            input_ids_output=input_ids_output,
        )


class TWTBertGenerationDecoder(TWTGenerationMixin, BertGenerationDecoder):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        if not config.is_decoder:
            logger.warn("If you want to use `BertGenerationDecoder` as a standalone, add `is_decoder=True.`")

        self.bert = TWTBertGenerationEncoder(config)
        self.lm_head = BertGenerationOnlyLMHead(config)
        self.p_copy_linear = nn.Linear(self.config.hidden_size * 3, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        copy_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        Returns:

        Example::

            >>> from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
            >>> import torch

            >>> tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
            >>> config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            >>> config.is_decoder = True
            >>> model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder', config=config)

            >>> inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # !!! Embedding output of the decoder input, shape: (batch_size, output_seq_length, hidden_size), e.g. (1, 128, 768)
        embedding_output = outputs.embedding_output
        # !!! Cross attention of the last decoder layer, shape: (batch_size, num_attention_heads, output_seq_length, input_seq_length), e.g. (1, 12, 128, 512)
        last_cross_attention = outputs.cross_attentions[-1]
        # !!! Attention output (context) of the last decoder layer, shape: (batch_size, output_seq_length, hidden_size), e.g. (1, 128, 768)
        last_cross_attention_output = outputs.cross_attention_outputs[-1]
        # !!! Hidden state of the last decoder layer, shape: (batch_size, output_seq_length, hidden_size), e.g. (1, 128, 768)
        sequence_output = outputs[0]

        prediction_scores = self.lm_head(sequence_output)

        # !!! Check no_copy config
        if not (hasattr(self.config, 'no_copy') and self.config.no_copy):
            # !!! Calculate p_copy based on Few-Shot NLG
            p_copy_input = torch.cat([last_cross_attention_output, sequence_output, embedding_output], dim=-1)
            p_copy = self.p_copy_linear(p_copy_input)
            p_copy = torch.sigmoid(p_copy)
            p_gen = 1 - p_copy
            # !!! Calculate prediction scores with p_gen
            prediction_scores_ = p_gen * softmax(prediction_scores, dim=-1)
            # !!! Take average of attention scores from 12 heads, dim=1 is the head dimention
            last_cross_attention_ = p_copy * torch.mean(last_cross_attention, dim=1)
            # !!! Expand input_ids to (1, 128, 512)
            encoder_input_ids_ = encoder_input_ids.unsqueeze(1).expand(last_cross_attention_.size(0), last_cross_attention_.size(1), -1)
            # !!! Calculate findal prediction scores with input ids
            prediction_scores_.scatter_add(2, encoder_input_ids_, last_cross_attention_)
            prediction_scores_ = prediction_scores_ + 1e-8
        else:
            prediction_scores_ = softmax(prediction_scores, dim=-1)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores_[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            loss_fct = NLLLoss()
            lm_loss = loss_fct(torch.log(shifted_prediction_scores.view(-1, self.config.vocab_size)), labels.view(-1))

            # !!! Check no_copy config
            if not (hasattr(self.config, 'no_copy') and self.config.no_copy):
                if not (hasattr(self.config, 'gen_loss_rate') and self.config.gen_loss_rate == 0):
                    # !!! Save copy score for each example
                    copy_gate_scores = torch.zeros([input_ids.size(0), 1]).to(labels.device)
                    for batch, dec_labels in enumerate(labels):
                        # !!! Find same input_ids between decoder input and encoder input
                        # enc_dec_same_tokens = (dec_labels[:,None]==encoder_input_ids[batch][None,:])*attention_mask[batch][1:,None]
                        # enc_dec_same_tokens = torch.nonzero(enc_dec_same_tokens)
                        # !!! Get unique positions from the decoder inputs, 0 is the decoder input position
                        # dec_token_matched_positions = torch.unique(enc_dec_same_tokens[:, 0])
                        dec_token_matched_positions = torch.nonzero(copy_mask[batch]==1).squeeze(1)
                        same_token_copy_score = torch.sum(p_gen[batch][dec_token_matched_positions, :])
                        # !!! Save current copy gate score
                        copy_gate_scores[batch, 0] = same_token_copy_score
                    lm_loss = lm_loss + (self.config.gen_loss_rate if hasattr(self.config, 'gen_loss_rate') else 1.0) * torch.mean(copy_gate_scores)

        if not return_dict:
            output = (prediction_scores_,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores_,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
