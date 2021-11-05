from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.encoder_decoder import EncoderDecoderModel
from transformers.models.encoder_decoder.configuration_encoder_decoder import \
    EncoderDecoderConfig

from ..twt_generation_utils import TWTGenerationMixin
from .modeling_twt_bert_generation import (TWTBertGenerationDecoder,
                                           TWTBertGenerationEncoder)


class TWTEncoderDecoderModel(TWTGenerationMixin, EncoderDecoderModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        PreTrainedModel.__init__(self, config)

        if encoder is None:
            encoder = TWTBertGenerationEncoder(config.encoder)

        if decoder is None:
            decoder = TWTBertGenerationDecoder(config.decoder)

        self.encoder = encoder
        self.decoder = decoder
        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()

    # Add cross attention mask (use ** kwargs)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import EncoderDecoderModel, BertTokenizer
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints

            >>> # forward
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            >>> # training
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
            >>> loss, logits = outputs.loss, outputs.logits

            >>> # save and load from pretrained
            >>> model.save_pretrained("bert2bert")
            >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

            >>> # generation
            >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # !!! Pop cross attention mask
        cross_attention_mask = kwargs_encoder.pop('cross_attention_mask')

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]

        # !!! Check cross attention mask
        if hasattr(self.config.decoder, 'no_cross_attention_mask') and self.config.decoder.no_cross_attention_mask:
            cross_attention_mask = attention_mask
        else:
            if cross_attention_mask is None:
                cross_attention_mask = attention_mask

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_input_ids=encoder_outputs['input_ids_output'],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=cross_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
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
        input_dict = super(TWTEncoderDecoderModel, self).prepare_inputs_for_generation(input_ids, past, attention_mask, use_cache, encoder_outputs)
        input_dict["cross_attention_mask"] = kwargs["decoder_cross_attention_mask"]
        return input_dict

# TODO: uncomment the following could help remove code in class TWTGenerationMixin
# def adjust_logits_during_generation(self, logits: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
#     # the logits here are probability, while the input for beam search is logit in 
#     # generation_utils, the logit will further be the input of log_softmax, which 
#     # equals to log(p), note that log_softmax(log(p)) = log(p), so we adjust the 
#     # probablity to log(p)
#     return torch.log(logits)
