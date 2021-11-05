import os
import json
import string
import random
import argparse
import collections

from utils.data_utils import load_jsonl_data, norm_and_tokenize, get_idx_from_offset, get_tokenized_offsets
from utils.recognizer_utils import build_sub_table, rec_input, rec_output, find_missing_info
from language.tabfact.preprocess_data import align_inputs
from language.tabfact.convert_to_totto import convert_table


class BuildConfig:
    PREFIX_FACT_AMOUNT = 1
    MAX_ROWS = 20
    MAX_COLS = 10
    RANDOM_COUNT = 2


def get_max_table_id(tables_dir: str):
    max_table_id = 0
    if os.path.isdir(tables_dir):
        table_ids = sorted([int(file_name.rstrip(".json")) for file_name in os.listdir(tables_dir)], reverse=True)
        if table_ids:
            max_table_id = table_ids[0]
    return max_table_id


def load_tabfact_data(data_path, split_ids_path):
    split_ids = set()
    with open(split_ids_path, 'r', encoding="utf-8") as split_ids_file:
        split_ids = set(json.load(split_ids_file))
    with open(data_path, 'r', encoding="utf-8") as data_file:
        count = 0
        json_data = json.load(data_file)
        if json_data:
            for data_id, (table_id, data_items) in enumerate(json_data.items()):
                if table_id in split_ids:
                    if count % 100 == 0:
                        print("Num examples processed: %d" % count)
                    count += 1
                    yield data_id, table_id, data_items
    return iter([]), iter([]), iter([])


def is_valid_table(table_data):
    if table_data:
        if len(table_data) <= BuildConfig.MAX_ROWS:
            if len(table_data[0]) <= BuildConfig.MAX_COLS:
                return True
    return False


def build_table(table_data, table_metas):
    dump_data = {
        "meta": table_metas,
        "data": table_data
    }
    return json.dumps(dump_data, indent=4)


def build_meta(caption="", page_title="", section_title=""):
    meta_data = {
        'caption': caption,
        'page_title': page_title,
        'section_title': section_title,
    }
    return meta_data


def save_data_examples(output_path, data_examples):
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write(os.linesep.join(data_examples))


def save_data_tables(output_dir, data_tables):
    for (table_id, data_table) in data_tables:
        with open(f"{output_dir}/{table_id}.json", 'w', encoding="utf-8") as f:
            f.write(data_table)


def construct_data_format(table_id, output_sentence, prefixes, start_indices, source, source_id):
    data_format = {
        'table_id': table_id,
        'output_sentence': output_sentence,
        'prefixes': prefixes,
        'start_indices': start_indices,
        'source': source,
        'source_id': source_id,
    }
    return data_format


def build_from_tabfact(data_path, split_ids_path, tables_path, current_output_table_id):
    data_examples, random_examples, data_tables = [], [], []
    for data_id, table_id, data_items in load_tabfact_data(data_path, split_ids_path):
        # Load tottto style table
        totto_table = convert_table(f"{tables_path}/{table_id}")
        table_built = False
        if is_valid_table(totto_table):
            output_tokenized_sents, output_labels, table_caption = data_items
            for output_tokenized_sent, output_label in zip(output_tokenized_sents, output_labels):
                # Only build entail statements
                if output_label == 1:
                    output_tokens = output_tokenized_sent.split(" ")
                    output_tokenized_offsets = get_tokenized_offsets(output_tokens)
                    # Match facts between table, metas and output sentence
                    # try:
                    _, match_res = align_inputs((totto_table, [table_caption], output_tokenized_sent))
                    try:
                        # Follow the tabfact format to support multiple output sentences (statements)
                        matched_facts = match_res[4][0]
                    except IndexError:
                        matched_facts = []

                    generation_prefixes, trigger_indicies, random_prefixes, random_trigger_indicies = [], [], [], []
                    # Match missing values between input and output
                    input_values, input_times = rec_input(totto_table, [table_caption], False)
                    output_values, output_times = rec_output(output_tokenized_sent)
                    missing_values, _ = find_missing_info(input_values, input_times, output_values, output_times)
                    # Matched facts are necessary for fact coverage calculation
                    if matched_facts:
                        generation_prefixes, trigger_indicies = build_generation_prefix(output_tokens, output_tokenized_offsets, matched_facts, missing_values)
                        random_prefixes, random_trigger_indicies = build_random_generation_prefix(output_tokens)

                    if generation_prefixes or random_prefixes:
                        if not table_built:
                            # New table id
                            current_output_table_id += 1
                            # Build table
                            data_table = build_table(totto_table, build_meta(table_caption))
                            data_tables.append((current_output_table_id, data_table))
                            # Set table saved
                            table_built = True
                        if generation_prefixes:
                            # Save data
                            data_example = construct_data_format(current_output_table_id, output_tokenized_sent, generation_prefixes, trigger_indicies, "tabfact", data_id)
                            # Build data for self use
                            data_example['matched_facts'] = matched_facts
                            data_examples.append(json.dumps(data_example))
                        if random_prefixes:
                            # Save data
                            random_example = construct_data_format(current_output_table_id, output_tokenized_sent, random_prefixes, random_trigger_indicies, "tabfact", data_id)
                            # Build data for self use
                            random_example['matched_facts'] = matched_facts
                            random_examples.append(json.dumps(random_example))

    return data_examples, random_examples, data_tables


def build_from_totto(data_path, current_output_table_id):
    data_examples, random_examples, data_tables = [], [], []

    for data_id, json_data in enumerate(load_jsonl_data(data_path, True, True)):
        table = json_data['table']
        # Normalize and tokenize metadata
        meta_tokenized_sents = collections.OrderedDict()
        if json_data['table_page_title']:
            _, page_title_tokens, _ = norm_and_tokenize(json_data['table_page_title'])
            meta_tokenized_sents['page_title'] = " ".join(page_title_tokens)
        if json_data['table_section_title']:
            _, section_title_tokens, _ = norm_and_tokenize(json_data['table_section_title'])
            meta_tokenized_sents['section_title'] = " ".join(section_title_tokens)
        # Normalize and tokenize target sentence
        output_sentence = json_data['sentence_annotations'][0]['final_sentence']
        _, output_tokens, _ = norm_and_tokenize(output_sentence)
        output_tokenized_sent = " ".join(output_tokens)
        output_tokenized_offsets = get_tokenized_offsets(output_tokens)
        # Match facts between table, metas and output sentence
        # try:
        _, match_res = align_inputs((table, list(meta_tokenized_sents.values()), output_tokenized_sent))
        try:
            # Follow the tabfact format to support multiple output sentences (statements)
            matched_facts = match_res[4][0]
        except IndexError:
            matched_facts = []

        generation_prefixes, trigger_indicies, random_prefixes, random_trigger_indicies = [], [], [], []
        # Match missing values between input and output
        subtable, _ = build_sub_table(json_data)
        input_values, input_times = rec_input(subtable, list(meta_tokenized_sents.values()), True)
        output_values, output_times = rec_output(output_tokenized_sent)
        missing_values, _ = find_missing_info(input_values, input_times, output_values, output_times)
        # Matched facts are necessary for fact coverage calculation
        if matched_facts:
            generation_prefixes, trigger_indicies = build_generation_prefix(output_tokens, output_tokenized_offsets, matched_facts, missing_values)
            random_prefixes, random_trigger_indicies = build_random_generation_prefix(output_tokens)

        if generation_prefixes or random_prefixes:
            # New table id
            current_output_table_id += 1
            # Build table
            data_table = build_table(
                table,
                build_meta(
                    "",
                    meta_tokenized_sents['page_title'] if 'page_title' in meta_tokenized_sents else "",
                    meta_tokenized_sents['section_title'] if 'section_title' in meta_tokenized_sents else "",
                )
            )
            data_tables.append((current_output_table_id, data_table))
            if generation_prefixes:
                # Save data
                data_example = construct_data_format(current_output_table_id, output_tokenized_sent, generation_prefixes, trigger_indicies, "totto", data_id)
                # Build data for self use
                data_example['matched_facts'] = matched_facts
                data_examples.append(json.dumps(data_example))
            if random_prefixes:
                # Save data
                random_example = construct_data_format(current_output_table_id, output_tokenized_sent, random_prefixes, random_trigger_indicies, "totto", data_id)
                # Build data for self use
                random_example['matched_facts'] = matched_facts
                random_examples.append(json.dumps(random_example))

    return data_examples, random_examples, data_tables


def build_random_generation_prefix(output_tokens):
    generation_prefixes, trigger_indicies = [], []
    if len(output_tokens) > 3:
        trigger_indicies = sorted(random.sample(list(range(1, len(output_tokens) - 1)), BuildConfig.RANDOM_COUNT))
        generation_prefixes = [" ".join(output_tokens[:trigger_idx]) for trigger_idx in trigger_indicies]
    return generation_prefixes, trigger_indicies


def build_generation_prefix(output_tokens, output_token_offsets, matched_facts, matched_values):
    generation_prefixes, trigger_indicies, preserved_indicies = [], [], []
    if matched_facts:
        # The prefix should contain at least 1 fact from the table
        if len(matched_facts) > BuildConfig.PREFIX_FACT_AMOUNT:
            for i, (fact_start_idx, matched_fact) in enumerate(matched_facts.items()):
                if i > BuildConfig.PREFIX_FACT_AMOUNT - 1:
                    fact_str, _ = matched_fact
                    fact_length = len(fact_str.split(" "))
                    generation_prefixes.append(" ".join(output_tokens[:fact_start_idx]))
                    trigger_indicies.append(fact_start_idx)
                else:
                    preserved_indicies.append(fact_start_idx)

    # There must be preserved facts in the prefix
    if matched_values and preserved_indicies:
        for matched_value in matched_values:
            value_start_idx, value_end_idx = get_idx_from_offset(output_token_offsets, matched_value['Start'], matched_value['End'])
            # The offset should be matched to the index
            if value_start_idx != -1 and value_end_idx != -1:
                # Ordinal number
                if matched_value['TypeName'] == "ordinal":
                    # The value should not be the last word and the next word should not be punctuation
                    if value_end_idx < len(output_tokens) and output_tokens[value_end_idx] not in string.punctuation:
                        if value_end_idx not in trigger_indicies:
                            # Logic should not be prior to the fact
                            if value_end_idx > preserved_indicies[-1]:
                                generation_prefixes.append(" ".join(output_tokens[:value_end_idx]))
                                trigger_indicies.append(value_end_idx)

                else:
                    if value_start_idx not in trigger_indicies:
                        # Logic should not be prior to the fact
                        if value_start_idx > preserved_indicies[-1]:
                            generation_prefixes.append(" ".join(output_tokens[:value_start_idx]))
                            trigger_indicies.append(value_start_idx)

    return generation_prefixes, trigger_indicies


def set_seed(seed):
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10, help="Random seed for building random prefixes")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Set seed for building random prefixes
    set_seed(args.seed)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    output_tables_dir = os.path.join(current_dir, "data/dataset/twt/tables")

    # Build from ToTTo train set
    print("Build from TOTTO train set")
    max_table_id = get_max_table_id(output_tables_dir)
    totto_train_path = os.path.join(current_dir, "data/dataset/totto/totto_train_data.jsonl")
    totto_train_examples, totto_train_random_examples, totto_train_tables = build_from_totto(totto_train_path, max_table_id)
    save_data_tables(output_tables_dir, totto_train_tables)

    # Build from ToTTo dev set
    print("Build from TOTTO dev set")
    max_table_id = get_max_table_id(output_tables_dir)
    totto_dev_path = os.path.join(current_dir, "data/dataset/totto/totto_dev_data.jsonl")
    totto_dev_output_path = os.path.join(current_dir, "data/dataset/twt/totto_dev.jsonl")
    totto_dev_random_output_path = os.path.join(current_dir, "data/dataset/twt/totto_random_dev.jsonl")
    totto_dev_examples, totto_dev_random_examples, totto_dev_tables = build_from_totto(totto_dev_path, max_table_id)
    save_data_examples(totto_dev_output_path, totto_dev_examples)
    save_data_examples(totto_dev_random_output_path, totto_dev_random_examples)
    save_data_tables(output_tables_dir, totto_dev_tables)

    # Move part of the ToTTo train set to the ToTTo test set (same length as the dev examples)
    print("Build TOTTO test set from train set")
    totto_train_output_path = os.path.join(current_dir, "data/dataset/twt/totto_train.jsonl")
    totto_train_random_output_path = os.path.join(current_dir, "data/dataset/twt/totto_random_train.jsonl")
    save_data_examples(totto_train_output_path, totto_train_examples[:-len(totto_dev_examples)])
    save_data_examples(totto_train_random_output_path, totto_train_random_examples[:-len(totto_dev_examples)])
    totto_test_output_path = os.path.join(current_dir, "data/dataset/twt/totto_test.jsonl")
    totto_test_random_output_path = os.path.join(current_dir, "data/dataset/twt/totto_random_test.jsonl")
    save_data_examples(totto_test_output_path, totto_train_examples[-len(totto_dev_examples):])
    save_data_examples(totto_test_random_output_path, totto_train_random_examples[-len(totto_dev_examples):])

    # For TabFact
    tabfact_r1_data_path = os.path.join(current_dir, "data/dataset/tabfact/collected_data/r1_training_all.json")
    tabfact_r2_data_path = os.path.join(current_dir, "data/dataset/tabfact/collected_data/r2_training_all.json")
    tabfact_tables_path = os.path.join(current_dir, "data/dataset/tabfact/all_csv")

    # Build from TabFact train set
    print("Build from TabFact r1 train set")
    max_table_id = get_max_table_id(output_tables_dir)
    tabfact_train_ids_path = os.path.join(current_dir, "data/dataset/tabfact/train_id.json")
    tabfact_train_output_path = os.path.join(current_dir, "data/dataset/twt/tabfact_train.jsonl")
    tabfact_train_random_output_path = os.path.join(current_dir, "data/dataset/twt/tabfact_random_train.jsonl")
    tabfact_r1_train_examples, tabfact_r1_train_random_examples, tabfact_r1_train_tables = build_from_tabfact(tabfact_r1_data_path, tabfact_train_ids_path, tabfact_tables_path, max_table_id)
    save_data_tables(output_tables_dir, tabfact_r1_train_tables)
    print("Build from TabFact r2 train set")
    max_table_id = get_max_table_id(output_tables_dir)
    tabfact_r2_train_examples, tabfact_r2_train_random_examples, tabfact_r2_train_tables = build_from_tabfact(tabfact_r2_data_path, tabfact_train_ids_path, tabfact_tables_path, max_table_id)
    save_data_tables(output_tables_dir, tabfact_r2_train_tables)
    save_data_examples(tabfact_train_output_path, tabfact_r1_train_examples + tabfact_r2_train_examples)
    save_data_examples(tabfact_train_random_output_path, tabfact_r1_train_random_examples + tabfact_r2_train_random_examples)

    # Build from TabFact dev set
    print("Build from TabFact r1 dev set")
    max_table_id = get_max_table_id(output_tables_dir)
    tabfact_dev_ids_path = os.path.join(current_dir, "data/dataset/tabfact/val_id.json")
    tabfact_dev_output_path = os.path.join(current_dir, "data/dataset/twt/tabfact_dev.jsonl")
    tabfact_dev_random_output_path = os.path.join(current_dir, "data/dataset/twt/tabfact_random_dev.jsonl")
    tabfact_r1_dev_examples, tabfact_r1_dev_random_examples, tabfact_r1_dev_tables = build_from_tabfact(tabfact_r1_data_path, tabfact_dev_ids_path, tabfact_tables_path, max_table_id)
    save_data_tables(output_tables_dir, tabfact_r1_dev_tables)
    print("Build from TabFact r2 dev set")
    max_table_id = get_max_table_id(output_tables_dir)
    tabfact_r2_dev_examples, tabfact_r2_dev_random_examples, tabfact_r2_dev_tables = build_from_tabfact(tabfact_r2_data_path, tabfact_dev_ids_path, tabfact_tables_path, max_table_id)
    save_data_tables(output_tables_dir, tabfact_r2_dev_tables)
    save_data_examples(tabfact_dev_output_path, tabfact_r1_dev_examples + tabfact_r2_dev_examples)
    save_data_examples(tabfact_dev_random_output_path, tabfact_r1_dev_random_examples + tabfact_r2_dev_random_examples)

    # Build from TabFact test set
    print("Build from TabFact r1 test set")
    max_table_id = get_max_table_id(output_tables_dir)
    tabfact_test_ids_path = os.path.join(current_dir, "data/dataset/tabfact/test_id.json")
    tabfact_test_output_path = os.path.join(current_dir, "data/dataset/twt/tabfact_test.jsonl")
    tabfact_test_random_output_path = os.path.join(current_dir, "data/dataset/twt/tabfact_random_test.jsonl")
    tabfact_r1_test_examples, tabfact_r1_test_random_examples, tabfact_r1_test_tables = build_from_tabfact(tabfact_r1_data_path, tabfact_test_ids_path, tabfact_tables_path, max_table_id)
    save_data_tables(output_tables_dir, tabfact_r1_test_tables)
    print("Build from TabFact r2 test set")
    max_table_id = get_max_table_id(output_tables_dir)
    tabfact_r2_test_examples, tabfact_r2_test_random_examples, tabfact_r2_test_tables = build_from_tabfact(tabfact_r2_data_path, tabfact_test_ids_path, tabfact_tables_path, max_table_id)
    save_data_tables(output_tables_dir, tabfact_r2_test_tables)
    save_data_examples(tabfact_test_output_path, tabfact_r1_test_examples + tabfact_r2_test_examples)
    save_data_examples(tabfact_test_random_output_path, tabfact_r1_test_random_examples + tabfact_r2_test_random_examples)


if __name__ == "__main__":
    main()
