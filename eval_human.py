import os
import json
import imgkit
import random
from language.totto.table_to_text_html_utils import get_table_html
from utils.data_utils import gather_jsonl_data, load_jsonl_data, load_json_data
from twt_statistics import BuildConfig


def table_html_to_img():
    pass


def sample_data(test_data_amount, choose_amout, output_file_path):
    chosen_sample_ids = random.sample(list(range(test_data_amount - 1)), choose_amout)
    with open(output_file_path, 'w') as f:
        f.write(os.linesep.join([str(chosen_sample_id) for chosen_sample_id in chosen_sample_ids]))
    return chosen_sample_ids


def build_labeling_data(pred_file_path, dataset_file_path, tables_dir, table_imgs_dir, table_css_file_path, sample_ids, label_studio_file_path, prefix_types_file_path):
    options = {
        'xvfb': '',
        'quiet': ''
    }
    label_studio_records = []
    prefix_type_strs = []
    os.makedirs(table_imgs_dir, exist_ok=True)
    dataset_records = gather_jsonl_data(dataset_file_path)
    for i, prediction in enumerate(load_jsonl_data(pred_file_path, False)):
        if i in sample_ids:
            record_id = prediction['record_id']
            pred_sents = prediction['predictions']
            prefix = prediction['prefix_str']
            prefix_id = prediction['prefix_id']
            target_sent = prediction['target']
            dataset_record = dataset_records[record_id]
            if not os.path.exists(f"{table_imgs_dir}/f{dataset_record['table_id']}.jpg"):
                table = load_json_data(f"{tables_dir}/{dataset_record['table_id']}.json")
                table_html = get_table_html(table['data'], [])
                imgkit.from_string(table_html, f"{table_imgs_dir}/{dataset_record['table_id']}.jpg", options=options, css=table_css_file_path)
            label_studio_record = {
                'table': f"/data/local-files/?d=dataset1/{dataset_record['table_id']}.jpg",
                'target': target_sent,
                'prefix': prefix,
                'prediction': pred_sents[0]
            }
            label_studio_records.append({'data': label_studio_record})

            # Count logic prefixes:
            fact_indicies = [int(fact_idx) for fact_idx in list(dataset_record['matched_facts'].keys())[BuildConfig.PREFIX_FACT_AMOUNT:]]
            if dataset_record['start_indices'][prefix_id] in fact_indicies:
                prefix_type_str = 'factual'
            else:
                prefix_type_str = 'logical'
            prefix_type_strs.append(prefix_type_str)

    with open(label_studio_file_path, 'w') as f:
        json.dump(label_studio_records, f)

    if not os.path.exists(prefix_types_file_path):
        with open(prefix_types_file_path, 'w') as f:
            f.write(os.linesep.join(prefix_type_strs))


def build_tabfact():
    data_total_amount, data_sample_amount = 13955, 100

    task_name = "tabfact_random_causal_t5_base"
    pred_file_path = f"./output/human_eval/tabfact/{task_name}_test_set_predictions.jsonl"
    label_studio_file_path = f"./output/human_eval/tabfact/{task_name}_label_studio.json"
    prefix_types_file_path = "./output/human_eval/tabfact/sample_prefix_types.txt"
    sample_ids_file_path = "./output/human_eval/tabfact/sample_ids.txt"
    table_imgs_dir = "./output/human_eval/tabfact/table_imgs"
    dataset_file_path = "./data/dataset/twt/tabfact_test.jsonl"

    table_css_file_path = "./output/human_eval/table.css"
    tables_dir = "./data/dataset/twt/tables"

    if not os.path.exists(sample_ids_file_path):
        sample_ids = sample_data(data_total_amount, data_sample_amount, sample_ids_file_path)
    else:
        with open(sample_ids_file_path, 'r') as f:
            sample_ids = [int(line.strip()) for line in f.readlines() if line.strip()]
    build_labeling_data(pred_file_path, dataset_file_path, tables_dir, table_imgs_dir, table_css_file_path, sample_ids, label_studio_file_path, prefix_types_file_path)


def build_totto():
    data_total_amount, data_sample_amount = 27042, 100

    task_name = "totto_random_causal_t5_base"
    pred_file_path = f"./output/human_eval/totto/{task_name}_test_set_predictions.jsonl"
    label_studio_file_path = f"./output/human_eval/totto/{task_name}_label_studio.json"
    prefix_types_file_path = "./output/human_eval/totto/sample_prefix_types.txt"
    sample_ids_file_path = "./output/human_eval/totto/sample_ids.txt"
    table_imgs_dir = "./output/human_eval/totto/table_imgs"
    dataset_file_path = "./data/dataset/twt/totto_test.jsonl"

    table_css_file_path = "./output/human_eval/table.css"
    tables_dir = "./data/dataset/twt/tables"

    if not os.path.exists(sample_ids_file_path):
        sample_ids = sample_data(data_total_amount, data_sample_amount, sample_ids_file_path)
    else:
        with open(sample_ids_file_path, 'r') as f:
            sample_ids = [int(line.strip()) for line in f.readlines() if line.strip()]
    build_labeling_data(pred_file_path, dataset_file_path, tables_dir, table_imgs_dir, table_css_file_path, sample_ids, label_studio_file_path, prefix_types_file_path)


def summarize_results(res_dir):
    tabfact_sample_prefix_types_file_path = "./output/human_eval/tabfact/sample_prefix_types.txt"
    tabfact_sample_prefix_types = []
    with open(tabfact_sample_prefix_types_file_path, 'r') as f:
        tabfact_sample_prefix_types = [line.strip() for line in f.readlines() if line.strip()]

    totto_sample_prefix_types_file_path = "./output/human_eval/totto/sample_prefix_types.txt"
    totto_sample_prefix_types = []
    with open(totto_sample_prefix_types_file_path, 'r') as f:
        totto_sample_prefix_types = [line.strip() for line in f.readlines() if line.strip()]

    for res_file_name in os.listdir(res_dir):
        eval_results = load_json_data(os.path.join(res_dir, res_file_name))
        eval_results_sorted = sorted(eval_results, key=lambda x: x['id'])

        eval_scores, factual_scores, logical_scores = [], [], []

        if "tabfact" in res_file_name:
            sample_prefix_types = tabfact_sample_prefix_types
        elif "totto" in res_file_name:
            sample_prefix_types = totto_sample_prefix_types
        else:
            sample_prefix_types = []

        for eval_result, sample_prefix_type in zip(eval_results_sorted, sample_prefix_types):
            try:
                eval_score = int(eval_result['annotations'][0]['result'][0]['value']['choices'][0])
            except IndexError:
                eval_score = 1
            eval_scores.append(eval_score)
            if sample_prefix_type == "factual":
                factual_scores.append(eval_score)
            elif sample_prefix_type == "logical":
                logical_scores.append(eval_score)
            else:
                pass
        print(f"Evaluation Result for {res_file_name}")
        print(f"Average Score: {sum(eval_scores)/len(eval_scores)}")
        print(f"Average Factual Score: {sum(factual_scores)/len(factual_scores)}")
        print(f"Average Logical Score: {sum(logical_scores)/len(logical_scores)}")


if __name__ == "__main__":
    # build_tabfact()
    # build_totto()
    summarize_results("./output/human_eval/results")
