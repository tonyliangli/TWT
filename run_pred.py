#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
import os
import sys
import six
import json
import copy
import collections
from tqdm import tqdm
import argparse
import logging
import numpy as np
import torch

from transformers import (
    EncoderDecoderModel,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    BertTokenizerFast,
    BertTokenizer,
    T5TokenizerFast,
    T5Tokenizer,
    BartTokenizerFast,
    BartTokenizer
)

from language.totto.totto_to_twt_utils import ADDITIONAL_SPECIAL_TOKENS
from language.totto.table_to_text_html_utils import *
from utils.data_utils import load_json_data, gather_jsonl_data, inc_load_cache
from eval_metrics import eval_with_all_metrics, clac_bleu
from twt_preprocessing import get_metas_list
from model import (
    TWTEncoderDecoderModel,
    TWTT5ForConditionalGeneration,
    TWTBartForConditionalGeneration
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# !!! Do not use T5Tokenizer for prefix matching
MODEL_CLASSES = {
    "bert2bert": (EncoderDecoderModel, BertTokenizerFast),
    "twt_bert2bert": (TWTEncoderDecoderModel, BertTokenizerFast),
    "t5": (T5ForConditionalGeneration, T5TokenizerFast),
    "twt_t5": (TWTT5ForConditionalGeneration, T5TokenizerFast),
    "bart": (BartForConditionalGeneration, BartTokenizerFast),
    "twt_bart": (TWTBartForConditionalGeneration, BartTokenizerFast)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def generate_single(model, tokenizer, args, prompt_text, generation_inputs=None):
    # Prepare decoder input text
    if generation_inputs and ('decoder_input_ids' in generation_inputs):
        encoded_prompt = generation_inputs['decoder_input_ids'].to(args.device)
    else:
        prefix = args.prefix if args.prefix else ""
        encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    encoder_input_ids = generation_inputs['input_ids'].to(args.device)
    attention_mask = generation_inputs['attention_mask'].to(args.device)
    decoder_input_ids = input_ids

    gen_kwargs = {}
    if 'gen_kwargs' in generation_inputs:
        gen_kwargs = copy.deepcopy(generation_inputs['gen_kwargs'])
        for kwarg_key, gen_kwarg in gen_kwargs.items():
            gen_kwargs[kwarg_key] = gen_kwarg.to(args.device)

    if not args.greedy_search:
        gen_kwargs.update(
            {
                'top_k': args.k,
                'top_p': args.p,
                'do_sample': True,
                'temperature': args.temperature,
                'repetition_penalty': args.repetition_penalty,
                'num_return_sequences': args.num_return_sequences,
            }
        )

    # !!! Add prefix masking
    if hasattr(args, "masking_func") and args.masking_func is not None:
        gen_kwargs['decoder_masking_func'] = args.masking_func

    # Check clean up tokenizations (do no clean up for prefix masking)
    clean_up_tokenization_spaces = True
    if hasattr(args, 'clean_up_tokenization_spaces') and args.clean_up_tokenization_spaces is not None:
        clean_up_tokenization_spaces = args.clean_up_tokenization_spaces

    # Do prediction
    output_sequences = model.generate(
        input_ids=encoder_input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        output_attentions=True,
        **gen_kwargs
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=clean_up_tokenization_spaces)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=clean_up_tokenization_spaces)):]
        )

        # Strip left white space if prompt_text is empty
        if not prompt_text:
            total_sequence = total_sequence.lstrip()

        generated_sequences.append(total_sequence)

    return generated_sequences


def predict_dataset(model, tokenizer, args):
    dataset_records = gather_jsonl_data(args.dataset_path)
    model_inputs = []
    if hasattr(args, 'model_inputs_path') and args.model_inputs_path:
        model_inputs = inc_load_cache(args.model_inputs_path)

    # None-full models need linearized inputs
    linearized_model_inputs = []
    if args.model_type not in ["twt_bert2bert", "twt_t5", "twt_bart"]:
        if hasattr(args, 'linearized_model_inputs_path') and args.linearized_model_inputs_path:
            linearized_model_inputs = inc_load_cache(args.linearized_model_inputs_path)

    results = []
    if model_inputs:
        for model_input in tqdm(model_inputs):
            dataset_record = dataset_records[model_input['record_id']]
            table_json_data = load_json_data(f"{args.tables_dir}/{dataset_record['table_id']}.json")
            predictions = []
            # Get prefix from dataset record
            prefix = dataset_record['prefixes'][model_input['prefix_id']]
            pred_start_idx = dataset_record['start_indices'][model_input['prefix_id']]
            # !!! Special for full model
            if args.model_type in ["twt_bert2bert", "twt_t5", "twt_bart"]:
                generation_inputs = build_generation_input(model_input, tokenizer, args.no_row_col_embeddings)
            else:
                # Build linearized generation inputs directly from cache
                generation_inputs = linearized_model_inputs[model_input['record_id']]
                # full_table_metadata_str = linearize_full_table(table=table_json_data['data'], metas=get_metas_list(table_json_data['meta'])
                # generation_inputs = tokenizer([full_table_metadata_str], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            generated_sequences = generate_single(model, tokenizer, args, prefix, generation_inputs)
            predictions = [f"{prefix}{generated_sequence[len(prefix):]}" for generated_sequence in generated_sequences]
            result = {
                'target': dataset_record['output_sentence'],
                'predictions': predictions,
                'record_id': model_input['record_id'],
                'prefix_id': model_input['prefix_id'],
                'prefix_str': prefix,
            }
            # metrics = calc_eval_metrics(result)
            _, metrics_stats = eval_with_all_metrics(
                predictions,
                prefix,
                pred_start_idx,
                dataset_record['output_sentence'],
                table_json_data['data'],
                get_metas_list(table_json_data['meta']),
                dataset_record['matched_facts'],
                args.use_model_based_metrics
            )
            result['metrics'] = metrics_stats
            results.append(json.dumps(result))
    return results


def build_generation_input(model_input, tokenizer, no_row_col_embed=False):
    input_ids = torch.tensor(model_input['input_input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(model_input['input_attention_mask']).unsqueeze(0)
    # Remove BOS and EOS tokens
    # if model_input['prefix_input_ids'][0] == tokenizer.bos_token_id:
    #     model_input['prefix_input_ids'] = model_input['prefix_input_ids'][1:]
    if model_input['prefix_input_ids'][-1] == tokenizer.eos_token_id:
        model_input['prefix_input_ids'] = model_input['prefix_input_ids'][:-1]
    decoder_input_ids = torch.tensor(model_input['prefix_input_ids']).unsqueeze(0)

    if not no_row_col_embed:
        row_ids = torch.tensor(model_input['input_row_ids']).unsqueeze(0)
        col_ids = torch.tensor(model_input['input_col_ids']).unsqueeze(0)
    else:
        row_ids = torch.zeros(len(model_input['input_row_ids']), dtype=torch.int64).unsqueeze(0)
        col_ids = torch.zeros(len(model_input['input_col_ids']), dtype=torch.int64).unsqueeze(0)

    gen_kwargs = {
        'type_ids': torch.tensor(model_input['input_type_ids']).unsqueeze(0),
        'row_ids': row_ids,
        'col_ids': col_ids,
        'decoder_cross_attention_mask': torch.tensor(model_input['cross_attention_mask']).unsqueeze(0)[:,0,:]
    }
    generation_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'gen_kwargs': gen_kwargs
    }
    return generation_inputs


def calc_eval_metrics(result):
    metrics = collections.OrderedDict({
        "bleu_max": None,
        "bleu_avg": None
    })
    metrics['bleu_max'], metrics['bleu_avg'] = clac_bleu(result['prefix_str'], result['predictions'], result['target'])
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        type=str,
        # required=True,
        help="Path to pre-trained tokenzier or shortcut name",
    )

    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default="</s>", help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument(
        "--greedy_search",
        action="store_true",
        default=False,
        help="Whether to use greedy search",
    )
    parser.add_argument("--no_row_col_embeddings",
                        action="store_true",
                        default=False,
                        help="Whether to use row/col embeddings.")

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--clean_up_tokenization_spaces",
        action='store_true',
        default=True,
        help='Whether to clean up tokenization_spaces'
    )
    parser.add_argument(
        "--use_model_based_metrics",
        action='store_true',
        default=False,
        help='Whether to use model based metrics like BLEURT or Bert Score'
    )
    args = parser.parse_args()
    return args


def prepare_generation(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")
    model = model_class.from_pretrained(args.model_name_or_path)

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name_or_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})

    if model.__class__.__name__ in ["EncoderDecoderModel", "TWTEncoderDecoderModel"]:
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    elif model.__class__.__name__ in ["TWTT5ForConditionalGeneration"]:
        tokenizer.bos_token = tokenizer.pad_token
    else:
        pass
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=512)
    logger.info(args)

    return model, tokenizer, args


def main():
    args = parse_args()
    model, tokenizer, args = prepare_generation(args)
    predict_dataset(model, tokenizer, args)


if __name__ == "__main__":
    main()
