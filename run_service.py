import os
import sys
import copy
import json
import importlib
import collections

import torch
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerBase
from flask import Flask, request, Response

from utils.data_utils import load_json_data, load_jsonl_data, norm_and_tokenize
from language.totto.totto_to_twt_utils import linearize_full_table, ADDITIONAL_SPECIAL_TOKENS
from language.tabfact.convert_to_totto import convert_df
from twt_preprocessing import build_model_inputs_with_prefix, tokenize_sentence, tokenize_meta, align_output_with_table, get_metas_list
from run_pred import parse_args, adjust_length_to_model, set_seed, build_generation_input, generate_single
from model.prefix_matching import prefix_matching
# from totto.baselines.completion.batch_completion import parse_args, prepare_completion, serve


class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    MODELS = collections.OrderedDict({
        't5_twt/base': {
            'module_name': "model",
            'class_name': "TWTT5ForConditionalGeneration",
            'model_path': "/bdmstorage/teamdrive/tasks/final/exps/tabfact_prefix_t5_base_twt_0.4_gen_loss/checkpoints/checkpoint-10000",
            # 'model_path': "/bdmstorage/teamdrive/tasks/final_base/exps/totto_prefix_t5_base_twt_0.4_gen_loss/checkpoints/checkpoint-154000",
            'tokenizer_name': "t5-base",
        },
        'bert2bert_twt/base': {
            'module_name': "model.bert2bert.modeling_enc_dec_twt",
            'class_name': "TWTEncoderDecoderModel",
            'model_path': "/bdmstorage/teamdrive/tasks/final/exps/tabfact_prefix_bert2bert_base_twt_0.4_gen_loss/checkpoints/checkpoint-40000",
            'tokenizer_name': "bert-base-uncased",
        },
        't5/base': {
            'module_name': "transformers",
            'class_name': "T5ForConditionalGeneration",
            'model_path': "/bdmstorage/teamdrive/tasks/final/exps/tabfact_causal_t5_base/checkpoints/checkpoint-24000",
            'tokenizer_name': "t5-base",
        },
        'bert2bert/base': {
            'module_name': "transformers",
            'class_name': "EncoderDecoderModel",
            'model_path': "/bdmstorage/teamdrive/tasks/final/exps/tabfact_causal_bert2bert_base/checkpoints/checkpoint-16000",
            'tokenizer_name': "bert-base-uncased",
        }
    })
    TOKENIZERS = {
        't5-base': {
            "module_name": "model.transformers",
            "class_name": "T5Tokenizer",
        },
        'bert-base-uncased': {
            "module_name": "model.transformers",
            "class_name": "BertTokenizer",
        }
    }


class ModelManager(object):
    def __init__(self):
        super().__init__()
        self.models = collections.OrderedDict()
        self.tokenizers = collections.OrderedDict()

    def load_models(self, config_models, config_tokenizers):
        for model_name, config_model in config_models.items():
            # Load model
            model_module = importlib.import_module(config_model['module_name'])
            model_class = getattr(model_module, config_model['class_name'])
            model = model_class.from_pretrained(config_model['model_path'])
            model.to(app.device)
            self.models[model_name] = model
            # Load tokenizer
            tokenizer_name = config_model['tokenizer_name']
            if tokenizer_name not in self.tokenizers:
                tokenizer_module = importlib.import_module(config_tokenizers[tokenizer_name]['module_name'])
                tokenizer_class = getattr(tokenizer_module, config_tokenizers[tokenizer_name]['class_name'])
                tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
                tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
                if "bert" in tokenizer_name:
                    tokenizer.bos_token = tokenizer.cls_token
                    tokenizer.eos_token = tokenizer.sep_token
                elif "t5" in tokenizer_name:
                    tokenizer.bos_token = tokenizer.pad_token
                else:
                    pass
                self.tokenizers[tokenizer_name] = tokenizer

    def get_model(self, model_name):
        """Get a model object by qualified name."""
        if model_name in self.models:
            return self.models[model_name]
        return None

    def get_tokenizer(self, tokenizer_name):
        if tokenizer_name in self.tokenizers:
            return self.tokenizers[tokenizer_name]
        return None


def format_table(table_data):
    headers, data = [], []
    for idx, row in enumerate(table_data):
        if idx == 0:
            headers = [{'type': 'text', 'title': col['value'].strip(), 'width': 100} for col in row]
        else:
            data.append([col['value'].strip() for col in row])
    return headers, data


def is_valid_sequence(prefix, generated_sequences):
    for generated_sequence in generated_sequences:
        if generated_sequence[:len(prefix)] != prefix:
            return False
    return True


def format_output(model_name, prefix, generated_sequences, use_custom=False):
    custom_names = {
        't5_twt/base': "ours",
        't5/base': "baseline"
    }
    display_name = model_name
    if use_custom:
        display_name = custom_names[model_name]
    prefix = PreTrainedTokenizerBase.clean_up_tokenization(prefix)
    sentences = [{'value': f"{display_name.upper()}: {generated_sequence}", 'id': f"{generated_sequence}"} for generated_sequence in generated_sequences]
    return sentences


def load_dataset():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(current_dir, "data/dataset/final/tabfact_test.jsonl")
    tables_dir = os.path.join(current_dir, "data/dataset/final/tables")
    dataset_records = []
    table_ids = set()
    record_id = 0
    for dataset_record in load_jsonl_data(dataset_path, False, False):
        table_id = dataset_record['table_id']
        if table_id not in table_ids:
            table_json_data = load_json_data(f"{tables_dir}/{table_id}.json")
            dataset_records.append({
                'record_id': record_id,
                'table': table_json_data,
                'output_sentence': dataset_record['output_sentence'],
                'prefixes': dataset_record['prefixes'],
                'start_indices': dataset_record['start_indices']
            })
            table_ids.add(table_id)
            record_id += 1
    return dataset_records


def prepare_args(model_name, temperature, top_p):
    args = copy.deepcopy(parse_args())
    args.device = app.device
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    set_seed(args)
    if "bert2bert" in model_name:
        args.stop_token = "[SEP]"
        args.greedy_search = False
    elif "t5" in model_name:
        args.stop_token = "</s>"
        args.greedy_search = True
    else:
        pass
    args.temperature = temperature
    args.p = top_p
    args.num_return_sequences = 1
    args.length = adjust_length_to_model(20, max_sequence_length=512)
    args.clean_up_tokenization_spaces = False
    return args


def prepare_table(table_headers, table_data):
    model_input_table = []
    if table_headers and table_data:
        df = pd.DataFrame(data=table_data, columns=table_headers, dtype=str)
        df = df.applymap(lambda x: np.nan if (isinstance(x, str) and not x.strip()) else x)
        # Last not all-nan row
        first_valid_row_idx, last_valid_row_idx = df.first_valid_index(), df.last_valid_index()
        df_T = df.T
        first_valid_col, last_valid_col = df_T.first_valid_index(), df_T.last_valid_index()
        df_active = df.loc[first_valid_row_idx:last_valid_row_idx, first_valid_col:last_valid_col]
        model_input_table = convert_df(df_active)
    return model_input_table


def proc_generated_sequence(prefix, generated_sequences, table_data, meta_data, tokenizer):
    prefix_tokens = prefix.split(" ")
    generated_sequences_processed = []
    for generated_sequence in generated_sequences:
        generated_basic_tokens = generated_sequence.split(" ")
        _, metas_basic_tokens, _, _, _ = tokenize_meta(meta_data, tokenizer)
        # Align the output sentence with the table
        matched_facts, _ = align_output_with_table(table_data, metas_basic_tokens, generated_basic_tokens, tokenizer.eos_token)
        generation_end_idx = len(generated_basic_tokens)
        generation_start_char = generated_sequence[len(prefix)] if len(generated_sequence) > len(prefix) else None
        # The generated sentence should be aligned to the table
        aligned_to_table = False
        if matched_facts and generation_start_char is not None:
            for (fact_start_idx, matched_fact) in matched_facts.items():
                fact_end_idx = fact_start_idx + len(matched_fact[0].split(" "))
                if (generation_start_char != " " and fact_end_idx >= len(prefix_tokens)) or (generation_start_char == " " and fact_end_idx > len(prefix_tokens)):
                    if not aligned_to_table:
                        aligned_to_table = any(coord[0] != -1 for coord in matched_fact[1])
                    generation_end_idx = fact_end_idx
                    break
        if aligned_to_table:
            generated_sequences_processed.append(PreTrainedTokenizerBase.clean_up_tokenization(" ".join(generated_basic_tokens[:generation_end_idx])[len(prefix):]))
    return generated_sequences_processed


def prepare_masking_func(prefix, **kwargs):
    # Apply prefix mask
    prefix_tokens = prefix.split(" ")
    prefix_last_token = prefix_tokens[-1]
    masking_func = app.masking_factory([prefix_last_token], **kwargs)
    return masking_func


def prepare_model_input_with_prefix_mask(prefix, model_input, tokenizer):
    model_input_with_prefix = copy.deepcopy(model_input)
    prefix_tokens = prefix.split(" ")
    model_prefixes = [" ".join(prefix_tokens[:-1])]
    # Build completion prefix
    if len(prefix_tokens) > 1:
        _, _, _, prefix_input_ids, _, _ = tokenize_sentence(model_prefixes[0], tokenizer, "do_not_pad")
        # Only modify prefix input ids
        model_input_with_prefix['prefix_input_ids'] = prefix_input_ids
    else:
        # Only modify prefix input ids
        model_input_with_prefix['prefix_input_ids'] = [tokenizer.bos_token_id]
    return model_prefixes, model_input_with_prefix


# Init Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# Init device
app.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Init dataset
print("Load dataset")
app.dataset_records = load_dataset()
# Init models and tokenizers
print("Load models and tokenizers")
app.model_manager = ModelManager()
app.model_manager.load_models(Config.MODELS, Config.TOKENIZERS)
# Init masking factory
print("Load masking factory")
app.masking_factory = prefix_matching.PrefixMaskingFuncFactory(app.model_manager.get_tokenizer('t5-base'), 1, 'vocab')


@app.route('/api/get_example', methods=['GET'])
def get_example():
    try:
        record_id = int(request.args.get('record_id', 0))
        if record_id >= len(app.dataset_records):
            record_id = 0
    except (ValueError, TypeError):
        record_id = 0
    dataset_record = app.dataset_records[record_id]
    table_headers, table_data = format_table(dataset_record['table']['data'])
    dataset_record['table_headers'] = table_headers
    dataset_record['table_data'] = table_data
    dataset_record['meta_data'] = os.linesep.join(get_metas_list(dataset_record['table']['meta'])) if get_metas_list(dataset_record['table']['meta']) else ""
    return Response(json.dumps(dataset_record), status=200, mimetype='application/json')


@app.route('/api/do_complete', methods=['POST'])
def do_complete():
    try:
        model_size = str(request.get_json().get('model_size', "t5_twt/base"))
    except (ValueError, TypeError):
        model_size = "t5_twt/base"
    try:
        temperature = float(request.get_json().get('temperature', 1.0))
    except (ValueError, TypeError):
        temperature = 1.0
    try:
        top_p = float(request.get_json().get('top_p', 0.9))
    except (ValueError, TypeError):
        top_p = 0.9
    try:
        # User input context
        context = str(request.get_json().get('context', ""))
    except (ValueError, TypeError):
        context = ""

    # Table data
    meta_data = request.get_json().get('meta_data', "")
    table_headers = request.get_json().get('table_headers', "")
    table_data = request.get_json().get('table_data', {})

    # Prepare model and tokenizer
    model_names = []
    if model_size == "all":
        model_names = list(Config.MODELS.keys())
    elif model_size == "custom":
        model_names = ["t5_twt/base", "t5/base"]
    else:
        model_names.append(model_size)

    json_data = {}
    # Build custom input
    if context:
        model_input_table = prepare_table(table_headers.split(","), table_data)
        if model_input_table:
            _, context_tokens, _ = norm_and_tokenize(context)
            context_tokenized = " ".join(context_tokens)
            metas_tokenized = [" ".join(norm_and_tokenize(meta)[1]) for meta in meta_data.strip().split(os.linesep)[:2]] if meta_data.strip() else []
            json_data['table'] = {'meta': metas_tokenized, 'data': model_input_table}
            json_data['output_sentence'] = context_tokenized
            json_data['prefixes'] = [context_tokenized]
            json_data['start_indices'] = [len(context_tokens)]

    output_sequences = []
    for model_name in model_names:
        model = app.model_manager.get_model(model_name)
        tokenizer = app.model_manager.get_tokenizer(Config.MODELS[model_name]['tokenizer_name'])
        masking_func = None

        if json_data:
            model_inputs, generation_inputs = None, None
            model_prefixes = json_data['prefixes']
            if "twt" in model_name:
                model_inputs = build_model_inputs_with_prefix(json_data, tokenizer)
                if model_inputs:
                    model_input = model_inputs[0]
                    if "t5" in model_name:
                        if not context.endswith(" "):
                            # Apply prefix mask
                            masking_func = prepare_masking_func(json_data['prefixes'][0])
                            # Prepare prefix model inputs
                            model_prefixes, model_input = prepare_model_input_with_prefix_mask(json_data['prefixes'][0], model_input, tokenizer)
                    generation_inputs = build_generation_input(model_input, tokenizer, False)
            else:
                full_table_metadata_str = linearize_full_table(table=json_data['table']['data'], metas=get_metas_list(json_data['table']['meta']))
                generation_inputs = tokenizer([full_table_metadata_str], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

            if generation_inputs:
                model_prefix = model_prefixes[0]
                # We use the original json data to acquire the original prefix
                prefix = json_data['prefixes'][0]
                args = prepare_args(model_name, temperature, top_p)
                args.masking_func = masking_func
                generated_sequences = generate_single(model, tokenizer, args, model_prefix, generation_inputs)

                # Add fall back (4 times) for prefix masking
                if args.masking_func is not None:
                    if not is_valid_sequence(prefix, generated_sequences):
                        # Remove space marker from the prefix mask
                        args.masking_func = prepare_masking_func(json_data['prefixes'][0], remove_space_marker=True)
                        generated_sequences = generate_single(model, tokenizer, args, model_prefix, generation_inputs)
                        if not is_valid_sequence(prefix, generated_sequences):
                            # Force the prefix as the next token
                            args.masking_func = prepare_masking_func(json_data['prefixes'][0], force_prefix=True)
                            generated_sequences = generate_single(model, tokenizer, args, model_prefix, generation_inputs)
                            if not is_valid_sequence(prefix, generated_sequences):
                                # Do not use the prefix mask
                                args.masking_func = None
                                generation_inputs = build_generation_input(model_inputs[0], tokenizer, False)
                                generated_sequences = generate_single(model, tokenizer, args, prefix, generation_inputs)
                                if not is_valid_sequence(prefix, generated_sequences):
                                    # Return empty result
                                    generated_sequences = []

                generated_sequences_proc = proc_generated_sequence(prefix, generated_sequences, json_data['table']['data'], get_metas_list(json_data['table']['meta']), tokenizer)
                model_output_sequences = format_output(model_name, f"{prefix} " if context.endswith(" ") else prefix, generated_sequences_proc, model_size == "custom")
                if model_output_sequences:
                    output_sequences.extend(model_output_sequences)

    return Response(json.dumps({'sentences': output_sequences}), status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.getenv('PORT', 5001)
    app.run('0.0.0.0', port=port)
