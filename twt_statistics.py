import os
from utils.data_utils import load_json_data, load_jsonl_data
from twt_build import BuildConfig


def get_stats_template():
    template = {
        'avg_prefix_len': 0,
        'avg_target_len': 0,
        'avg_rows': 0,
        'avg_cols': 0,
        'avg_cells': 0,
        'med_rows': 0,
        'med_cols': 0,
        'med_cells': 0,
        'fact_prefixes': 0,
        'logic_prefixes': 0
    }
    return template.copy()


def median(data):
    sorted_data = sorted(data)
    half = len(sorted_data) // 2
    return (sorted_data[half] + sorted_data[~half])/2


def load_dataset(dataset_dir, source='totto', is_random=True):
    splits = ['train', 'dev', 'test']
    if os.path.exists(dataset_dir):
        for split in splits:
            dataset_file_path = os.path.join(dataset_dir, f"{source}{'_random' if is_random and split != 'test' else ''}_{split}.jsonl")
            if os.path.exists(dataset_file_path):
                print(f"Count dataset from {os.path.basename(dataset_file_path)}")
                for json_data in load_jsonl_data(dataset_file_path, False, False):
                    yield split, json_data
            else:
                print(f"{dataset_file_path} missing.")
    return iter([])


def count_table(table_data):
    row_num, col_num, cell_num = 0, 0, 0
    if table_data and len(table_data) > 0:
        # Count row num
        max_row_span = 1
        for last_row_col in table_data[-1]:
            if last_row_col['row_span'] > max_row_span:
                max_row_span = last_row_col['row_span']
        row_num = len(table_data) + max_row_span - 1
        # Count col num
        for col in table_data[0]:
            col_num += col['column_span']
        cell_num = row_num * col_num
    return row_num, col_num, cell_num


def count_dataset(dataset_dir, tables_dir, source='totto', is_random=True):
    count_items = {
        'prefix_len': [],
        'target_len': [],
        'row_nums': [],
        'col_nums': [],
        'cell_nums': [],
        'fact_prefix_nums': [],
        'logic_prefix_nums': [],
    }
    dataset_items = {}
    for split, json_data in load_dataset(dataset_dir, source, is_random):
        table_json_data = load_json_data(f"{tables_dir}/{json_data['table_id']}.json")
        # Count table
        row_num, col_num, cell_num = count_table(table_json_data['data'])
        count_items['row_nums'].append(row_num)
        count_items['col_nums'].append(col_num)
        count_items['cell_nums'].append(cell_num)
        # Count target
        count_items['target_len'].append(len(json_data['output_sentence'].split(" ")))
        # Count prefix
        for prefix in json_data['prefixes']:
            count_items['prefix_len'].append(len(prefix.split(" ")))
            count_items['target_len'].append(len(prefix.split(" ")))
        # Count dataset
        if split not in dataset_items:
            dataset_items[split] = {'pair_num': 0, 'prefix_num': 0}
        dataset_items[split]['pair_num'] += 1
        dataset_items[split]['prefix_num'] += len(json_data['prefixes'])
        # Count logic prefixes:
        fact_indicies = [int(fact_idx) for fact_idx in list(json_data['matched_facts'].keys())[BuildConfig.PREFIX_FACT_AMOUNT:]]
        logic_prefix_num = len(set(json_data['start_indices']).difference(set(fact_indicies)))
        count_items['logic_prefix_nums'].append(logic_prefix_num)
        # Count fact prefixes
        fact_prefix_num = len(json_data['start_indices']) - logic_prefix_num
        count_items['fact_prefix_nums'].append(fact_prefix_num)
    return count_items, dataset_items


def summarize_count(count_items, dataset_items):
    template = get_stats_template()
    for split, dataset_item in dataset_items.items():
        template[f'{split}_pairs'] = dataset_item['pair_num']
        template[f'{split}_prefixes'] = dataset_item['prefix_num']
    template['avg_prefix_len'] = sum(count_items['prefix_len'])/len(count_items['prefix_len'])
    template['avg_target_len'] = sum(count_items['target_len'])/len(count_items['prefix_len'])
    template['avg_rows'] = sum(count_items['row_nums'])/len(count_items['row_nums'])
    template['avg_cols'] = sum(count_items['col_nums'])/len(count_items['col_nums'])
    template['avg_cells'] = sum(count_items['cell_nums'])/len(count_items['cell_nums'])
    template['med_rows'] = median(count_items['row_nums'])
    template['med_cols'] = median(count_items['col_nums'])
    template['med_cells'] = median(count_items['cell_nums'])
    template['fact_prefixes'] = sum(count_items['fact_prefix_nums'])
    template['logic_prefixes'] = sum(count_items['logic_prefix_nums'])
    return template


def main():
    dataset_dir = "./data/dataset/twt"
    tables_dir = "./data/dataset/twt/tables"
    totto_count_items, totto_dataset_items = count_dataset(dataset_dir, tables_dir, 'totto', True)
    print("totto dataset statistics")
    totto_statistics = summarize_count(totto_count_items, totto_dataset_items)
    print(totto_statistics)
    tabfact_count_items, tabfact_dataset_items = count_dataset(dataset_dir, tables_dir, 'tabfact', True)
    print("tabfact dataset statistics")
    tabfact_statistics = summarize_count(tabfact_count_items, tabfact_dataset_items)
    print(tabfact_statistics)


if __name__ == "__main__":
    main()
