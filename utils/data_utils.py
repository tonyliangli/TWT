import os
import six
import gzip
import json
import string
import difflib
import collections
import pickle
import _pickle as cPickle
from typing import List
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase


def inc_dump_cache(file_path, obj):
    """
    Dump an object with cPickle incrementally
    :param file_path: input file path
    :param obj: Any object for pickle to dump
    """
    # Check if directory of the file path exists, create if not
    with gzip.open(file_path, 'ab') as f_pkl_gz:
        pickler = cPickle.Pickler(f_pkl_gz, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(obj)


def inc_load_cache(path, test_mode=False):
    """
    Load pickled file incrementally
    :param path: directory or file path
    :return: loaded object from pickle file
    """
    def load_pickle(fp):
        """
        Load data from pickle file path
        :param fp: input file path
        """
        # Get file name with type
        base_name = os.path.basename(fp)
        # Separate file name and type
        file_name, file_type = os.path.splitext(base_name)
        # Check file type
        if file_type == ".gz" and ".pkl" in file_name:
            # Load data from pickle
            with gzip.open(fp, 'rb') as f_pkl_gz:
                try:
                    while True:
                        # Load pickle data
                        obj = cPickle.load(f_pkl_gz)
                        # Check if data type matches
                        assert isinstance(obj, list), "pickle data should be list"
                        # Save global data
                        data.extend(obj)
                        if test_mode:
                            break
                except EOFError:
                    pass

    # To load from pickle file
    data = []
    if os.path.isdir(path):
        # Load all files
        for file_name in os.listdir(path):
            # Get file path
            file_path = os.path.join(path, file_name)
            # Load pickle data
            load_pickle(file_path)
    elif os.path.isfile(path):
        # Load single file
        load_pickle(path)
    else:
        pass

    return data


def load_json_data(f_path):
    # Load table json data
    json_data = None
    with open(f_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def load_jsonl_data(f_path, verbose=True, lower=False):
    with open(f_path, 'r') as f:
        count = 0
        for line in f:
            if line:
                if verbose:
                    if count % 100 == 0:
                        print("Num examples processed: %d" % count)
                data_line = six.ensure_text(line, "utf-8")
                if lower:
                    data_line = data_line.lower()
                json_data = json.loads(data_line)
                count += 1
                yield json_data
        if verbose:
            print("Num examples processed: %d" % count)
    return iter([])
 

def gather_jsonl_data(f_path):
    json_data_gather = []
    for json_data in load_jsonl_data(f_path, False, False):
        json_data_gather.append(json_data)
    return json_data_gather


def fuzzy_match_tokens(output_tokens, input_tokens, eos_token):
    """Match output tokens (from the target sentence) with input tokens (from a table cell)"""
    valid_input_tokens = []
    for input_token in input_tokens:
        if input_token not in string.punctuation and input_token not in ["(", ")"]:
            valid_input_tokens.append(input_token)
    
    valid_output_tokens = output_tokens[0:output_tokens.index(eos_token)+1] if eos_token in output_tokens else output_tokens

    if len(valid_input_tokens) == 0:
        return (-1, -1)

    matched_count = 0
    min_matched_pos, max_matched_pos = len(output_tokens), -1

    for valid_input_token in valid_input_tokens:
        if valid_input_token in valid_output_tokens:
            matched_pos = valid_output_tokens.index(valid_input_token)
            if matched_pos < min_matched_pos:
                min_matched_pos = matched_pos
            if matched_pos > max_matched_pos:
                max_matched_pos = matched_pos
            matched_count += 1

    # Overlap more than 40% of the input token
    if float(matched_count) / len(valid_input_tokens) > 0.4:
        # Matched length should be less than 2 times of the length of valid input tokens
        if max_matched_pos - min_matched_pos <= 2 * len(valid_input_tokens):
            return (min_matched_pos, max_matched_pos)

    return (-1, -1)


def detok_bpe(tokens, eos_token, special_tokens=[]):
    """Reverse build complete tokens from bpe tokens"""
    basic_tokens, token_indicies = [], []
    if tokens and len(tokens) > 0:
        buffer_str = ""
        buffer_indicies = []
        for index, token in enumerate(tokens):
            # End with EOS token
            if token == eos_token:
                break
            if token.startswith('Ġ') or token in special_tokens:
                if buffer_indicies:
                    basic_tokens.append(buffer_str)
                    token_indicies.append(buffer_indicies)
                    buffer_str = ""
                    buffer_indicies = []
                basic_tokens.append(token.lstrip("Ġ"))
                token_indicies.append([index])
            else:
                if index > 0:
                    if tokens[index-1].startswith("Ġ"):
                        buffer_str = basic_tokens.pop(-1)
                        buffer_str += token
                        buffer_indicies.extend(token_indicies.pop(-1))
                        buffer_indicies.append(index)
                    else:
                        buffer_str += token
                        buffer_indicies.append(index)
                else:
                    basic_tokens.append(token)
                    token_indicies.append([index])
        # Clear buffer
        if buffer_indicies:
            basic_tokens.append(buffer_str)
            token_indicies.append(buffer_indicies)
            buffer_str = ""
            buffer_indicies = []
        # Add EOS token
        if token == eos_token:
            basic_tokens.append(token)
            token_indicies.append([index])
    return basic_tokens, token_indicies


def detok_wordpiece(tokens):
    """Reverse build complete tokens from wordpiece tokens"""
    basic_tokens, token_indicies = [], []
    if tokens and len(tokens) > 0:
        buffer_str = ""
        buffer_indicies = []
        for index, token in enumerate(tokens):
            if not token.startswith('##'):
                if buffer_indicies:
                    basic_tokens.append(buffer_str)
                    token_indicies.append(buffer_indicies)
                    buffer_str = ""
                    buffer_indicies = []
                basic_tokens.append(token)
                token_indicies.append([index])
            else:
                if index > 0:
                    if not tokens[index-1].startswith("##"):
                        buffer_str = basic_tokens.pop(-1)
                        buffer_str += token.lstrip("##")
                        buffer_indicies.extend(token_indicies.pop(-1))
                        buffer_indicies.append(index)
                    else:
                        buffer_str += token.lstrip("##")
                        buffer_indicies.append(index)
                else:
                    buffer_str = token.lstrip("##")
                    buffer_indicies.append(index)
    return basic_tokens, token_indicies


def detok_sentencepiece(tokens, bos_token, eos_token):
    basic_tokens, token_indicies = [], []
    if tokens and len(tokens) > 0:
        buffer_str = ""
        buffer_indicies = []
        for index, token in enumerate(tokens):
            # Special tokens and punctuations
            if token in [bos_token, eos_token] or token in string.punctuation:
                if buffer_indicies:
                    basic_tokens.append(buffer_str)
                    token_indicies.append(buffer_indicies)
                    buffer_str = ""
                    buffer_indicies = []
                # Add these tokens to list
                basic_tokens.append(token)
                token_indicies.append([index])
                if token == bos_token or token in string.punctuation:
                    continue
                if token == eos_token:
                    break

            # Process other tokens
            if not token.startswith('▁'):
                buffer_str += token
                buffer_indicies.append(index)
            else:
                if buffer_indicies:
                    basic_tokens.append(buffer_str)
                    token_indicies.append(buffer_indicies)
                    # Clear buffer
                    buffer_str = ""
                    buffer_indicies = []
                if token != "▁":
                    buffer_str = token.lstrip('▁')
                    buffer_indicies = [index]

        if buffer_indicies:
            basic_tokens.append(buffer_str)
            token_indicies.append(buffer_indicies)

    return basic_tokens, token_indicies


def is_whole_word(haystack, start_offset, end_offset):
    prev_char = ' '
    if start_offset > 0:
        prev_char = haystack[start_offset - 1]
    next_char = ' '
    if end_offset < len(haystack) - 1:
        next_char = haystack[end_offset + 1]
    # Add to matched words if matched type is a word
    if prev_char == ' ' and next_char == ' ':
        return True
    return False


def is_valid_ahocorasick_match(haystack: str, matched: str, end_offset: int, whole_word_only: True):
    # Calculate start offset from label text and end offset
    start_offset = end_offset - len(matched) + 1
    # Cut label from haystack with start and end offset, check if both labels are the same
    if haystack[start_offset:start_offset + len(matched)] == matched:
        # Check if current label is a word by checking spaces before and after current label
        if whole_word_only:
            if is_whole_word(haystack, start_offset, end_offset):
                return True
        else:
            return True
    return False


def build_ahocorasick(match_items: List[str], tokenize=False):
    A = ahocorasick.Automaton(ahocorasick.STORE_ANY)
    for idx, matched in enumerate(match_items):
        if not A.exists(match_item.lower().strip()):
            A.add_word(match_item, ([idx], matched))
        else:
            exist_value = A.get(matched)
            exist_value[0].append(idx)
    return A


def match_ahocorasick(A, haystack, whole_word_only=True):
    matched_items = []
    if A:
        for end_offset, (indicies, matched) in automation.iter_long(haystack):
            # Check if current match is a valid match
            if is_valid_ahocorasick_match(haystack, matched, end_offset, whole_word_only):
                matched_items.append((indicies, matched, end_offset))
    return matched_items


def match_block(input_str: str, output_str: str):
    s = difflib.SequenceMatcher(None, input_str, output_str)
    matched_blocks = s.get_matching_blocks()
    return list(matched_blocks)


def tokenize(text):
    tokens, token_offsets = [], []
    if text:
        tokenizer = TreebankWordTokenizer()
        tok_spans = tokenizer.span_tokenize(text)
        if tok_spans:
            token_offsets = list(tok_spans)
            tokens = [text[tok_offset[0]:tok_offset[1]] for tok_offset in token_offsets]
    return tokens, token_offsets


def norm_and_tokenize(text):
    tokens, token_offsets = [], []    
    if text:
        normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
        norm_text = normalizer.normalize_str(text)
        tokens, token_offsets = tokenize(norm_text)
    return norm_text, tokens, token_offsets


def detokenize(tokenized_text):
    if tokenized_text:
        tokens = tokenized_text.split("")
        detokenizer = TreebankWordDetokenizer()
        detokenized_text = detokenizer.detokenize(tokens)
        return detokenized_text
    return tokenized_text


def get_tokenized_offsets(tokens):
    token_offsets = []
    if tokens:
        start_idx = 0
        for token in tokens:
            token_offsets.append((start_idx, start_idx + len(token)))
            # +1 for space
            start_idx = start_idx + len(token) + 1
    return token_offsets


def get_idx_from_offset(token_offsets, start_offset, end_offset):
    start_idx, end_idx = -1, -1
    if token_offsets:
        for idx, token_offset in enumerate(token_offsets):
            if start_offset >= token_offset[0] and start_offset <= token_offset[1]:
                start_idx = idx
            if end_offset >= token_offset[0] and end_offset <= token_offset[1]:
                end_idx = idx + 1
                break
    return start_idx, end_idx
