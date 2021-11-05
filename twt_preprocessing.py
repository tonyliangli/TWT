import os
import copy
import json
import itertools
import argparse
import collections

import numpy as np
from transformers import (
    BertTokenizerFast,
    BertTokenizer,
    T5TokenizerFast,
    T5Tokenizer,
    BartTokenizer,
    BartTokenizerFast,
    logging
)

from utils.data_utils import (
    inc_dump_cache,
    load_json_data,
    load_jsonl_data,
    detok_bpe,
    detok_wordpiece,
    detok_sentencepiece
)
from language.tabfact.preprocess_data import align_inputs
from language.totto.totto_to_twt_utils import parse_table, parse_linearized_table, linearize_full_table, ADDITIONAL_SPECIAL_TOKENS
from twt_build import build_random_generation_prefix

# Suppress warnings
logger = logging.get_logger("transformers.tokenization_utils_base")
logger.setLevel(logging.CRITICAL)


MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
CACHE_DATA_FREQ = 1000


def build_table(json_data):
    return json_data["table"]["data"]


def build_metas(json_data):
    return json_data['table']['meta']


def get_metas_list(metas_dict):
    if metas_dict:
        return list(filter(None, metas_dict.values()))
    return []


def get_type_id(meta_type, tokenizer):
    pos_offset = get_pos_offset(tokenizer)
    meta_type_map = {
        'special': 0 + pos_offset,
        'caption': 1 + pos_offset,
        'page_title': 1 + pos_offset,
        'section_title': 2 + pos_offset,
        'cell': 3 + pos_offset,
    }
    if meta_type in meta_type_map:
        return meta_type_map[meta_type]
    return 0


def tokenize_meta(meta_data, tokenizer):
    meta_tokens, metas_basic_tokens, metas_basic_token_indicies, meta_token_type_ids = [], [], [], []
    meta_type_id = 0
    for meta_type, meta_content in meta_data.items():
        if meta_content:
            meta_type_id = get_type_id(meta_type, tokenizer)
            meta_content_tokens = list(
                itertools.chain(*(tokenizer.tokenize(t, is_split_into_words=False) for t in meta_content.split(" ")))
            )
            # Detok tokens
            basic_tokens, basic_token_indicies = detok_tokenized(meta_content_tokens, tokenizer)
            # Save basic tokens and corresponding indicies
            metas_basic_tokens.append(basic_tokens)
            metas_basic_token_indicies.append(basic_token_indicies)
            # Concat all meta tokens
            meta_tokens += meta_content_tokens
            meta_token_type_ids += [meta_type_id] * len(meta_content_tokens)
    assert len(meta_tokens) == len(meta_token_type_ids), "The length of meta tokens and token type ids should be identical."
    return meta_tokens, metas_basic_tokens, metas_basic_token_indicies, meta_token_type_ids


def tokenize_linearized_meta(meta_data, tokenizer):
    meta_tokens, metas_basic_tokens, metas_basic_token_indicies, meta_token_type_ids = [], [], [], []
    meta_type_id = 0
    for meta_type, meta_content in meta_data.items():
        if meta_content:
            meta_type_id = get_type_id(meta_type, tokenizer)
            meta_content = "<meta> " + meta_content + " </meta>"
            meta_content_tokens = list(
                itertools.chain(*(tokenizer.tokenize(t, is_split_into_words=False) for t in meta_content.split(" ")))
            )
            # Detok tokens
            basic_tokens, basic_token_indicies = detok_tokenized(meta_content_tokens, tokenizer)
            # Save basic tokens and corresponding indicies
            metas_basic_tokens.append(basic_tokens)
            metas_basic_token_indicies.append(basic_token_indicies)
            # Concat all meta tokens
            meta_tokens += meta_content_tokens
            meta_token_type_ids += [meta_type_id] * len(meta_content_tokens)
    assert len(meta_tokens) == len(meta_token_type_ids), "The length of meta tokens and token type ids should be identical."
    return meta_tokens, metas_basic_tokens, metas_basic_token_indicies, meta_token_type_ids


def build_completion_prefix(json_data):
    if 'current_prefix_id' in json_data and 'start_indices' in json_data:
        if len(json_data['prefixes']) > json_data['current_prefix_id'] and len(json_data['start_indices']) > json_data['current_prefix_id']:
            return json_data['current_prefix_id'], json_data['prefixes'][json_data['current_prefix_id']], json_data['start_indices'][json_data['current_prefix_id']]
    return None, None, None


def build_output_sentence(json_data):
    return json_data['output_sentence']


def tokenize_sentence(sentence, tokenizer, padding="max_length"):
    output_tokenized = tokenizer(sentence.split(" "), padding=padding, truncation=True, max_length=128, is_split_into_words=True)
    output_tokens = tokenizer.convert_ids_to_tokens(output_tokenized['input_ids'])
    output_input_ids = output_tokenized['input_ids']
    output_attention_mask = output_tokenized['attention_mask']
    # Shift output input ids from labels for T5
    if tokenizer.__class__.__name__ in ["T5TokenizerFast", "T5Tokenizer"]:
        if output_input_ids:
            if output_input_ids[0] != tokenizer.bos_token_id:
                output_labels = output_input_ids
                output_tokens = shift_right_target(output_tokens, tokenizer.bos_token)
                output_input_ids = shift_right_target(output_input_ids, tokenizer.bos_token_id)
                output_attention_mask = shift_right_target(output_attention_mask, 1)
            else:
                raise Exception('By default, the T5Tokenizer will not add the BOS token to the start of sequence')
    elif tokenizer.__class__.__name__ in ["BartTokenizerFast", "BartTokenizer"]:
        if output_input_ids:
            if output_input_ids[0] == tokenizer.bos_token_id:
                output_labels = output_input_ids.copy()[1:]
                # The last pad token shall be ignored
                output_labels.append(tokenizer.pad_token_id)
            else:
                raise Exception('By default, the BartTokenizer should add the BOS token to the start of sequence')
    else:
        output_labels = output_input_ids.copy()

    basic_tokens, basic_token_indicies = detok_tokenized(output_tokens, tokenizer)

    return output_tokens, basic_tokens, basic_token_indicies, output_input_ids, output_attention_mask, output_labels


def tokenize_table_cell(parsed_cells, cell_values, tokenizer):
    for parsed_cell, cell_value in zip(parsed_cells, cell_values):
        parsed_cell_tokens = tokenizer.tokenize(parsed_cell)
        cell_value_tokens = tokenizer.tokenize(cell_value)
        basic_tokens = detok_tokenized(cell_value_tokens, tokenizer)
        yield parsed_cell_tokens, basic_tokens
    return iter([]), iter([])


def align_output_with_table(table, metas_basic_tokens, output_basic_tokens, eos_token):
    def find_valid_tokens(tokens):
        if eos_token in tokens:
            valid_tokens = tokens[0:tokens.index(eos_token)+1]
        else:
            valid_tokens = tokens
        return valid_tokens

    meta_tokenized_sents = [" ".join(find_valid_tokens(meta_basic_tokens)) for meta_basic_tokens in metas_basic_tokens]
    output_tokenized_sent = " ".join(find_valid_tokens(output_basic_tokens))
    _, match_res = align_inputs((table, meta_tokenized_sents, output_tokenized_sent))
    try:
        # Follow the tabfact format to support multiple output sentences (statements)
        matched_facts = match_res[4][0]
    except IndexError:
        matched_facts = []
    facts_coord_to_indicies = parse_aligned_facts(matched_facts)

    return matched_facts, facts_coord_to_indicies


def parse_aligned_facts(matched_facts):
    # The table coordinate is the key
    coord_to_indicies = collections.OrderedDict()
    if matched_facts:
        for i, (fact_start_idx, matched_fact) in enumerate(matched_facts.items()):
            fact_str, table_coords = matched_fact
            fact_end_idx = fact_start_idx + len(fact_str.split(" "))
            for table_coord in table_coords:
                proc_table_coord = table_coord
                # Because all metadata attend to the output sentence, there is no need to mark the meta id
                if table_coord[0] == -1:
                    proc_table_coord = (-1, -1)
                if proc_table_coord not in coord_to_indicies:
                    coord_to_indicies[proc_table_coord] = []
                coord_to_indicies[proc_table_coord].append((fact_str, (fact_start_idx, fact_end_idx)))
    return coord_to_indicies


def shift_right_target(shift_target, shift_content):
    shifted_target = []
    if shift_target:
        shifted_target = [shift_content] + shift_target[:-1]
    return shifted_target


def detok_tokenized(tokens, tokenizer):
    basic_tokens, basic_token_indicies = [], []
    # Return basic tokens (merged from word pieces) for string matching purposes
    if tokenizer.__class__.__name__ in ["BertTokenizerFast", "BertTokenizer"]:
        basic_tokens, basic_token_indicies = detok_wordpiece(tokens)
    elif tokenizer.__class__.__name__ in ["T5TokenizerFast", "T5Tokenizer"]:
        basic_tokens, basic_token_indicies = detok_sentencepiece(tokens, tokenizer.bos_token, tokenizer.eos_token)
    elif tokenizer.__class__.__name__ in ["BartTokenizerFast", "BartTokenizer"]:
        basic_tokens, basic_token_indicies = detok_bpe(tokens, tokenizer.eos_token, tokenizer.additional_special_tokens)
    else:
        pass
    return basic_tokens, basic_token_indicies


def get_pos_offset(tokenizer):
    if tokenizer.__class__.__name__ in ["BartTokenizerFast", "BartTokenizer"]:
        return 2
    return 0


def build_model_inputs(json_data, tokenizer, record_id=0, build_cross_attention_mask_with_prefix=False):
    # Build and tokenize output sentence
    output_sentence = build_output_sentence(json_data)
    output_tokens, output_basic_tokens, output_basic_token_indicies, output_input_ids, output_attention_mask, output_labels = tokenize_sentence(output_sentence, tokenizer)

    # Build completion prefix (for generation only)
    prefix_id, completion_prefix, completion_start_index = build_completion_prefix(json_data)
    prefix_tokens, prefix_basic_tokens, prefix_basic_token_indicies, prefix_input_ids, prefix_attention_mask = None, None, None, None, None
    if completion_prefix:
        prefix_tokens, prefix_basic_tokens, prefix_basic_token_indicies, prefix_input_ids, prefix_attention_mask, prefix_labels = tokenize_sentence(completion_prefix, tokenizer, "do_not_pad")

    # Build and tokenize input meta data
    meta_data = build_metas(json_data)
    meta_tokenize_func = tokenize_linearized_meta if tokenizer.__class__.__name__ in ["BartTokenizerFast", "BartTokenizer"] else tokenize_meta
    meta_tokens, metas_basic_tokens, metas_basic_token_indicies, meta_token_type_ids = meta_tokenize_func(meta_data, tokenizer)
    # Set meta start index. For BERT, [CLS] is the start token.
    meta_start_index = 0
    if meta_tokens:
        meta_start_index = 1 if tokenizer.cls_token else 0

    # Set type ids for special tokens and cell tokens
    special_type_id, cell_type_id = get_type_id('special', tokenizer), get_type_id('cell', tokenizer)

    # Build and tokenize input table
    table = build_table(json_data)
    # Align the output sentence with the table
    matched_facts, facts_coord_to_indicies = align_output_with_table(table, metas_basic_tokens, output_basic_tokens, tokenizer.eos_token)
    # Parse table
    table_parse_func = parse_linearized_table if tokenizer.__class__.__name__ in ["BartTokenizerFast", "BartTokenizer"] else parse_table
    parsed_cells, cell_values, row_indicies, col_indicies, adujusted_facts_coord_to_indicies = table_parse_func(table, facts_coord_to_indicies)

    # The position of BART starts from 2
    pos_offset = get_pos_offset(tokenizer)

    # Set cell start index. For BERT, [SEP] token is after meta tokens.
    cell_start_index = meta_start_index + len(meta_tokens) + 1 if tokenizer.sep_token else meta_start_index + len(meta_tokens)

    cell_tokens, cell_type_ids, row_ids, col_ids,  = [], [], [], []
    attend_row_ids, attend_col_ids = set(), set()
    cross_attention_mask = np.zeros((MAX_OUTPUT_LENGTH, MAX_INPUT_LENGTH), dtype=np.int64)
    output_copy_mask = np.zeros(MAX_OUTPUT_LENGTH, dtype=np.int64)
    for i, (parsed_cell_tokens, cell_value_basic_tokens) in enumerate(tokenize_table_cell(parsed_cells, cell_values, tokenizer)):
        # Skip this token if it cannot fit to length completely
        if cell_start_index + len(parsed_cell_tokens) > MAX_INPUT_LENGTH:
            break
        cell_tokens.extend(parsed_cell_tokens)
        cell_type_ids += [cell_type_id] * len(parsed_cell_tokens)
        row_ids += [row_indicies[i] + pos_offset] * len(parsed_cell_tokens)
        col_ids += [col_indicies[i] + pos_offset] * len(parsed_cell_tokens)

        # Match with full output sentence first
        if (row_indicies[i], col_indicies[i]) in adujusted_facts_coord_to_indicies:
            for (fact_str, output_matched_index) in adujusted_facts_coord_to_indicies[(row_indicies[i], col_indicies[i])]:
                # Get min and max index from the original word piece or sentence piece tokens
                output_min_matched_index = output_basic_token_indicies[output_matched_index[0]][0]
                output_max_matched_index = output_basic_token_indicies[output_matched_index[1] - 1][-1]
                # Set cross attention mask for current output token and cell (not necessary since we use all rows and columns of this cell for all output tokens)
                # cross_attention_mask[output_min_matched_index:output_max_matched_index + 1, cell_start_index:cell_start_index+len(parsed_cell_tokens)] = 1
                # Set output copy mask (for training only)
                output_copy_mask[output_min_matched_index:output_max_matched_index + 1] = 1
                # Match cell value with prefix (if is set and exist)
                if build_cross_attention_mask_with_prefix:
                    if prefix_basic_tokens:
                        # The matched fact is in the prefix portion
                        if output_matched_index[1] <= completion_start_index:
                            # Save attend row postion id and col position id
                            attend_row_ids.add(row_indicies[i] + pos_offset)
                            attend_col_ids.add(col_indicies[i] + pos_offset)
                else:
                    # Save attend row postion id and col position id directly
                    attend_row_ids.add(row_indicies[i] + pos_offset)
                    attend_col_ids.add(col_indicies[i] + pos_offset)

        # Shift cell start position
        cell_start_index += len(parsed_cell_tokens)

    # Process facts that are aligned to metadata (the metadata index is (-1, -1))
    if (-1, -1) in adujusted_facts_coord_to_indicies:
        for (fact_str, output_matched_index) in adujusted_facts_coord_to_indicies[(-1, -1)]:
            # Get min and max index from the original word piece or sentence piece tokens
            output_min_matched_index = output_basic_token_indicies[output_matched_index[0]][0]
            output_max_matched_index = output_basic_token_indicies[output_matched_index[1] - 1][-1]
            # Set cross attention mask for current output token and cell (not necessary since we use all rows and columns of this cell for all output tokens)
            # cross_attention_mask[output_min_matched_index:output_max_matched_index + 1, cell_start_index:cell_start_index+len(parsed_cell_tokens)] = 1
            # Set output copy mask (for training only)
            output_copy_mask[output_min_matched_index:output_max_matched_index + 1] = 1

    # Build final input
    if tokenizer.cls_token and tokenizer.sep_token:
        input_tokens = [tokenizer.cls_token] + meta_tokens + [tokenizer.sep_token] + cell_tokens if meta_tokens else [tokenizer.cls_token] + cell_tokens
        input_type_ids = [special_type_id] + meta_token_type_ids + [special_type_id] + cell_type_ids if meta_token_type_ids else [special_type_id] + cell_type_ids
        input_row_ids = [0 + pos_offset] + [0 + pos_offset] * len(meta_tokens) + [0 + pos_offset] + row_ids if meta_tokens else [0 + pos_offset] + row_ids
        input_col_ids = [0 + pos_offset] + [0 + pos_offset] * len(meta_tokens) + [0 + pos_offset] + col_ids if meta_tokens else [0 + pos_offset] + col_ids
    else:
        input_tokens = meta_tokens + cell_tokens if meta_tokens else cell_tokens
        input_type_ids = meta_token_type_ids + cell_type_ids if meta_token_type_ids else cell_type_ids
        input_row_ids = [0 + pos_offset] * len(meta_tokens) + row_ids if meta_tokens else row_ids
        input_col_ids = [0 + pos_offset] * len(meta_tokens) + col_ids if meta_tokens else col_ids

    input_attention_mask = [1] * len(input_tokens)

    # Save input index with valid content
    input_min_valid_index, input_max_valid_index = 0, len(input_tokens) - 1
    # Pad input to max length
    if len(input_tokens) < MAX_INPUT_LENGTH:
        pad_length = MAX_INPUT_LENGTH - len(input_tokens)
        input_tokens.extend([tokenizer.pad_token] * pad_length)
        input_type_ids.extend([tokenizer.pad_token_id] * pad_length)
        input_row_ids.extend([tokenizer.pad_token_id] * pad_length)
        input_col_ids.extend([tokenizer.pad_token_id] * pad_length)
        input_attention_mask.extend([0] * pad_length)
    # Process input tokens to input input ids
    input_input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    # Output tokens attend to each meta data token
    output_min_valid_index = 0
    output_max_valid_index = output_tokens.index(tokenizer.eos_token) if tokenizer.eos_token in output_tokens else len(output_tokens) - 1
    if meta_tokens:
        cross_attention_mask[output_min_valid_index:output_max_valid_index + 1, meta_start_index:meta_start_index + len(meta_tokens)] = 1

    # Output tokens attend to each matched row and column
    if attend_row_ids or attend_col_ids:
        if attend_row_ids:
            cross_attention_mask[output_min_valid_index:output_max_valid_index + 1, np.where(np.in1d(np.array(input_row_ids), list(attend_row_ids)))] = 1
        if attend_col_ids:
            cross_attention_mask[output_min_valid_index:output_max_valid_index + 1, np.where(np.in1d(np.array(input_col_ids), list(attend_col_ids)))] = 1
    else:
        # No matched cells, attend all by default
        cross_attention_mask[:output_max_valid_index + 1, :input_max_valid_index + 1] = 1
    # Make sure padding positions are masked
    cross_attention_mask[output_max_valid_index + 1:, input_max_valid_index + 1:] = 0

    assert len(input_tokens) == len(input_input_ids) == len(input_type_ids) == len(input_attention_mask) == len(input_row_ids) == len(input_col_ids), "Input length error"
    assert len(output_tokens) == len(output_input_ids) == len(output_attention_mask) == len(output_copy_mask) == len(output_labels), "Output length error"

    return {
        'record_id': record_id,
        'input_tokens': input_tokens,
        'input_input_ids': input_input_ids,
        'input_type_ids': input_type_ids,
        'input_attention_mask': input_attention_mask,
        'input_row_ids': input_row_ids,
        'input_col_ids': input_col_ids,
        'output_tokens': output_tokens,
        'output_input_ids': output_input_ids,
        'output_attention_mask': output_attention_mask,
        'output_copy_mask': output_copy_mask,
        'output_labels': output_labels,
        'cross_attention_mask': cross_attention_mask,
        'prefix_id': prefix_id,
        'prefix_tokens': prefix_tokens,
        'prefix_basic_tokens': prefix_basic_tokens,
        'prefix_basic_token_indicies': prefix_basic_token_indicies,
        'prefix_input_ids': prefix_input_ids,
        'prefix_attenttion_mask': prefix_attention_mask
    }


def build_model_inputs_with_prefix(json_data, tokenizer, record_id=0):
    model_inputs = []
    for prefix_id, (prefix, start_idx) in enumerate(zip(json_data['prefixes'], json_data['start_indices'])):
        # Set current prefix id
        json_data['current_prefix_id'] = prefix_id
        # Build model inputs
        model_input_data = build_model_inputs(json_data, tokenizer, record_id, True)
        if model_input_data['prefix_tokens']:
            # Build casual with prefix mask
            seq_ids = np.arange(MAX_OUTPUT_LENGTH, dtype=np.int64)
            seq_ids_expanded_x = np.expand_dims(seq_ids, 0).repeat(MAX_OUTPUT_LENGTH, axis=0)
            seq_ids_expandes_y = np.expand_dims(seq_ids, 1).repeat(MAX_OUTPUT_LENGTH, axis=1)
            output_causal_mask = (seq_ids_expanded_x <= seq_ids_expandes_y).astype(np.int64)
            output_causal_mask[0:len(model_input_data['prefix_tokens']), 0:len(model_input_data['prefix_tokens'])] = 1
            # Combine with attention mask
            output_attention_mask = np.array(model_input_data["output_attention_mask"], dtype=np.int64)
            output_attention_mask_x = np.expand_dims(output_attention_mask, 0).repeat(MAX_OUTPUT_LENGTH, axis=0)
            output_attention_mask_y = np.expand_dims(output_attention_mask, 1).repeat(MAX_OUTPUT_LENGTH, axis=1)
            output_attention_mask_2d = output_attention_mask_x * output_attention_mask_y
            output_causal_attention_mask = output_attention_mask_2d * output_causal_mask
            model_input_data['output_attention_mask'] = output_causal_attention_mask
            model_inputs.append(model_input_data)
    return model_inputs


def proc_cache_input_data(input_data_path, table_data_dir, tokenizer, cache_file_path, use_prefix=False):
    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)

    model_inputs = []
    for record_id, input_json_data in enumerate(load_jsonl_data(input_data_path, True, False)):
        # Load table json data
        table_json_data = load_json_data(f"{table_data_dir}/{input_json_data['table_id']}.json")
        input_json_data['table'] = table_json_data
        if not use_prefix:
            # Build causal model inputs
            model_input_data = build_model_inputs(input_json_data, tokenizer, record_id, False)
            model_inputs.append(model_input_data)
        else:
            # Build prefix model inputs
            model_input_data = build_model_inputs_with_prefix(input_json_data, tokenizer, record_id)
            model_inputs.extend(model_input_data)
        if len(model_inputs) % CACHE_DATA_FREQ == 0:
            inc_dump_cache(cache_file_path, model_inputs)
            model_inputs = []
    if model_inputs:
        inc_dump_cache(cache_file_path, model_inputs)
        model_inputs = []
    print("Process and saved successfully")


def proc_cache_linearized_input_data(input_data_path, tokenizer, table_data_dir, output_file_path, cache_file_path):
    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)

    processed_json_examples = []
    model_inputs = []
    for record_id, input_json_data in enumerate(load_jsonl_data(input_data_path, True, False)):
        # Load table json data
        table_json_data = load_json_data(f"{table_data_dir}/{input_json_data['table_id']}.json")
        input_json_data['table'] = table_json_data
        table, metas = build_table(input_json_data), build_metas(input_json_data)
        output_sentence = build_output_sentence(input_json_data)

        # Table strings without page and section title.
        full_table_metadata_str = (
            linearize_full_table(
                table=table,
                metas=get_metas_list(metas)
            )
        )

        processed_json_example = {
            'record_id': record_id,
            'full_table_metadata_str': full_table_metadata_str,
            'final_sentence': output_sentence
        }

        if not os.path.exists(output_file_path):
            processed_json_examples.append(json.dumps(processed_json_example))
        model_inputs.append(tokenizer([full_table_metadata_str], padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt"))

        if len(model_inputs) % CACHE_DATA_FREQ == 0:
            inc_dump_cache(cache_file_path, model_inputs)
            model_inputs = []

    if model_inputs:
        inc_dump_cache(cache_file_path, model_inputs)
        model_inputs = []

    if not os.path.exists(output_file_path):
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(os.linesep.join(processed_json_examples))


def proc_cache_random_input_data(input_data_path, table_data_dir, tokenizer, output_file_path, cache_file_path):
    # Remove existing cache file
    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)

    processed_json_examples = []
    model_inputs = []
    for record_id, input_json_data in enumerate(load_jsonl_data(input_data_path, True, False)):
        # Load table json data
        output_sentence = build_output_sentence(input_json_data)
        output_tokens = output_sentence.split(" ")

        # Build random prefixes
        generation_prefixes, trigger_indicies = build_random_generation_prefix(output_tokens)
        # Prefix is not always random, depending on the length of the output
        if generation_prefixes and trigger_indicies:
            input_json_data['prefixes'] = generation_prefixes
            input_json_data['start_indices'] = trigger_indicies

        # Make a copy for building model inputs
        input_json_data_copy = copy.deepcopy(input_json_data)
        # Load table json data
        table_json_data = load_json_data(f"{table_data_dir}/{input_json_data_copy['table_id']}.json")
        input_json_data_copy['table'] = table_json_data

        # Build prefix model inputs
        model_input_data = build_model_inputs_with_prefix(input_json_data_copy, tokenizer, record_id)
        model_inputs.extend(model_input_data)
        # Build dataset
        processed_json_examples.append(json.dumps(input_json_data))

        if len(model_inputs) % CACHE_DATA_FREQ == 0:
            inc_dump_cache(cache_file_path, model_inputs)
            model_inputs = []

    if model_inputs:
        inc_dump_cache(cache_file_path, model_inputs)
        model_inputs = []

    if not os.path.exists(output_file_path):
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(os.linesep.join(processed_json_examples))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        # required=True,
        help="Model type selected: bert2bert, t5 or bart",
    )
    parser.add_argument(
        "--model_size",
        default=None,
        type=str,
        # required=True,
        help="Model size selected: base or large",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        type=str,
        # required=True,
        help="Pattern selected: causal, prefix, random, or linearized",
    )
    parser.add_argument(
        "--is_random",
        action="store_true",
        default=False,
        help="If the current input data file is already randomly built"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    rand_pattern = "_random" if args.is_random else ""

    current_dir = os.path.abspath(os.path.dirname(__file__))

    tables_dir = os.path.join(current_dir, "data/dataset/twt/tables")

    totto_train_path = os.path.join(current_dir, "data/dataset/twt", f"totto{rand_pattern}_train.jsonl")
    totto_dev_path = os.path.join(current_dir, "data/dataset/twt", f"totto{rand_pattern}_dev.jsonl")
    totto_test_path = os.path.join(current_dir, "data/dataset/twt", f"totto{rand_pattern}_test.jsonl")

    tabfact_train_path = os.path.join(current_dir, "data/dataset/twt", f"tabfact{rand_pattern}_train.jsonl")
    tabfact_dev_path = os.path.join(current_dir, "data/dataset/twt", f"tabfact{rand_pattern}_dev.jsonl")
    tabfact_test_path = os.path.join(current_dir, "data/dataset/twt", f"tabfact{rand_pattern}_test.jsonl")

    if args.model_type == "bert2bert":
        tokenizer = BertTokenizerFast.from_pretrained(f"bert-{args.model_size}-uncased")
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    elif args.model_type == "t5":
        tokenizer = T5TokenizerFast.from_pretrained(f"t5-{args.model_size}")
        tokenizer.bos_token = tokenizer.pad_token
    elif args.model_type == "bart":
        tokenizer = BartTokenizerFast.from_pretrained(f"facebook/bart-{args.model_size}", add_prefix_space=True)
        tokenizer.cls_token = None
        tokenizer.sep_token = None
    else:
        print("Please assign --model_type with bert2bert or t5")
        exit()

    # Add special tokens
    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})

    if args.pattern not in ["causal", "prefix", "linearized", "random"]:
        print("Please assign --pattern with causal or prefix")
        exit()

    if args.pattern == "linearized":
        print("Building totto train model inputs")
        proc_cache_linearized_input_data(
            totto_train_path,
            tokenizer,
            tables_dir,
            f"./data/dataset/twt/totto{rand_pattern}_linearized_train.jsonl",
            f"./data/cache/twt/totto{rand_pattern}_linearized_{args.model_type}_{args.model_size}_train_model_inputs.pkl.gz"
        )
        print("Building totto validation model inputs")
        proc_cache_linearized_input_data(
            totto_dev_path,
            tokenizer,
            tables_dir,
            f"./data/dataset/twt/totto{rand_pattern}_linearized_dev.jsonl",
            f"./data/cache/twt/totto{rand_pattern}_linearized_{args.model_type}_{args.model_size}_dev_model_inputs.pkl.gz"
        )
        print("Building totto test model inputs")
        proc_cache_linearized_input_data(
            totto_test_path,
            tokenizer,
            tables_dir,
            f"./data/dataset/twt/totto{rand_pattern}_linearized_test.jsonl",
            f"./data/cache/twt/totto{rand_pattern}_linearized_{args.model_type}_{args.model_size}_test_model_inputs.pkl.gz"
        )
        print("Building tabfact train model inputs")
        proc_cache_linearized_input_data(
            tabfact_train_path,
            tokenizer,
            tables_dir,
            f"./data/dataset/twt/tabfact{rand_pattern}_linearized_train.jsonl",
            f"./data/cache/twt/tabfact{rand_pattern}_linearized_{args.model_type}_{args.model_size}_train_model_inputs.pkl.gz"
        )
        print("Building tabfact validation model inputs")
        proc_cache_linearized_input_data(
            tabfact_dev_path,
            tokenizer,
            tables_dir,
            f"./data/dataset/twt/tabfact{rand_pattern}_linearized_dev.jsonl",
            f"./data/cache/twt/tabfact{rand_pattern}_linearized_{args.model_type}_{args.model_size}_dev_model_inputs.pkl.gz"
        )
        print("Building tabfact test model inputs")
        proc_cache_linearized_input_data(
            tabfact_test_path,
            tokenizer,
            tables_dir,
            f"./data/dataset/twt/tabfact{rand_pattern}_linearized_test.jsonl",
            f"./data/cache/twt/tabfact{rand_pattern}_linearized_{args.model_type}_{args.model_size}_test_model_inputs.pkl.gz"
        )
    # This will also build the random dataset
    elif args.pattern == "random":
        print("Building totto random train set and model inputs")
        proc_cache_random_input_data(
            totto_train_path,
            tables_dir,
            tokenizer,
            "./data/dataset/twt/totto_random_train.jsonl",
            f"./data/cache/twt/totto_random_{args.model_type}_{args.model_size}_train_model_inputs.pkl.gz"
        )
        print("Building totto random validation set and model inputs")
        proc_cache_random_input_data(
            totto_dev_path,
            tables_dir,
            tokenizer,
            "./data/dataset/twt/totto_random_dev.jsonl",
            f"./data/cache/twt/totto_random_{args.model_type}_{args.model_size}_dev_model_inputs.pkl.gz"
        )
        print("Building totto random test set and model inputs")
        proc_cache_random_input_data(
            totto_test_path,
            tables_dir,
            tokenizer,
            "./data/dataset/twt/totto_random_test.jsonl",
            f"./data/cache/twt/totto_random_{args.model_type}_{args.model_size}_test_model_inputs.pkl.gz"
        )
        print("Building tabfact random train set and model inputs")
        proc_cache_random_input_data(
            tabfact_train_path,
            tables_dir,
            tokenizer,
            "./data/dataset/twt/tabfact_random_train.jsonl",
            f"./data/cache/twt/tabfact_random_{args.model_type}_{args.model_size}_train_model_inputs.pkl.gz"
        )
        print("Building tabfact random validation set and model inputs")
        proc_cache_random_input_data(
            tabfact_dev_path,
            tables_dir,
            tokenizer,
            "./data/dataset/twt/tabfact_random_dev.jsonl",
            f"./data/cache/twt/tabfact_random_{args.model_type}_{args.model_size}_dev_model_inputs.pkl.gz"
        )
        print("Building tabfact random test set and model inputs")
        proc_cache_random_input_data(
            tabfact_test_path,
            tables_dir,
            tokenizer,
            "./data/dataset/twt/tabfact_random_test.jsonl",
            f"./data/cache/twt/tabfact_random_{args.model_type}_{args.model_size}_test_model_inputs.pkl.gz"
        )
    else:
        use_prefix = False
        if args.pattern == "prefix":
            use_prefix = True

        print("Building totto train model inputs")
        proc_cache_input_data(
            totto_train_path,
            tables_dir,
            tokenizer,
            f"./data/cache/twt/totto{rand_pattern}_{args.pattern}_{args.model_type}_{args.model_size}_train_model_inputs.pkl.gz",
            use_prefix
        )
        print("Building totto validation model inputs")
        proc_cache_input_data(
            totto_dev_path,
            tables_dir,
            tokenizer,
            f"./data/cache/twt/totto{rand_pattern}_{args.pattern}_{args.model_type}_{args.model_size}_dev_model_inputs.pkl.gz",
            use_prefix
        )
        print("Building totto test model inputs")
        proc_cache_input_data(
            totto_test_path,
            tables_dir,
            tokenizer,
            f"./data/cache/twt/totto{rand_pattern}_{args.pattern}_{args.model_type}_{args.model_size}_test_model_inputs.pkl.gz",
            use_prefix
        )
        print("Building tabfact train model inputs")
        proc_cache_input_data(
            tabfact_train_path,
            tables_dir,
            tokenizer,
            f"./data/cache/twt/tabfact{rand_pattern}_{args.pattern}_{args.model_type}_{args.model_size}_train_model_inputs.pkl.gz",
            use_prefix
        )
        print("Building tabfact validation model inputs")
        proc_cache_input_data(
            tabfact_dev_path,
            tables_dir,
            tokenizer,
            f"./data/cache/twt/tabfact{rand_pattern}_{args.pattern}_{args.model_type}_{args.model_size}_dev_model_inputs.pkl.gz",
            use_prefix
        )
        print("Building tabfact test model inputs")
        proc_cache_input_data(
            tabfact_test_path,
            tables_dir,
            tokenizer,
            f"./data/cache/twt/tabfact{rand_pattern}_{args.pattern}_{args.model_type}_{args.model_size}_test_model_inputs.pkl.gz",
            use_prefix
        )


if __name__ == "__main__":
    main()
