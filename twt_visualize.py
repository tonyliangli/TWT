import os

from language.totto.table_to_text_html_utils import get_html_header, get_table_html
from utils.data_utils import load_json_data, load_jsonl_data, gather_jsonl_data
from twt_preprocessing import get_metas_list


def build_record_html(record_id, table, metas, target_sent, prefix_htmls):
    table_html = get_table_html(table, [])
    meta_htmls = "<br/>".join([f"<b>Meta {i+1}</b>: {meta} " for i, meta in enumerate(metas)])
    target_html = f"<h3>Target</h3>{target_sent}"
    pred_html = f"<h3>Predictions</h3>{''.join(prefix_htmls)}"
    record_html = f"<div style='border: 2px solid black'><b>Record ID</b>: {record_id} <br/> {meta_htmls} <br/> {table_html} <br/> {target_html} <br/> {pred_html}</div>"
    return record_html


def build_prefix_pred_html(prefix, pred_sents, metrics):
    metrics_str = ", ".join([f"{metric_type}: {metric_value}" for metric_type, metric_value in metrics['avg'].items()])
    metrics_html = f"<h4>metrics: ({metrics_str})</h4>"
    prefix_html = f"<h4>prefix: [{prefix}]</h4>"
    prefix_pred_html = "<br/>".join([f"{pred_sent}" for pred_sent in pred_sents])
    prefix_html = f"{metrics_html}{prefix_html}{prefix_pred_html}"

    return prefix_html


def visualize_data(prediction_path, dataset_path, tables_dir, output_path):
    if os.path.exists(prediction_path):
        dataset_records = gather_jsonl_data(dataset_path)
        record_id = -1
        table, metas, target_sent = None, None, None
        prefix_htmls, record_htmls = [], []
        for prediction in load_jsonl_data(prediction_path, False, False):
            if record_id != -1:
                if prediction['record_id'] != record_id:
                    # Process last record
                    if prefix_htmls:
                        record_htmls.append(build_record_html(record_id, table, metas, target_sent, prefix_htmls))
                        prefix_htmls = []

                    # A new dataset record
                    record_id = prediction['record_id']
                    prefix_id = prediction['prefix_id']
                    pred_sents = prediction['predictions']
                    metrics = prediction['metrics']

                    dataset_record = dataset_records[record_id]
                    table_data = load_json_data(f"{tables_dir}/{dataset_record['table_id']}.json")
                    table, metas = table_data['data'], get_metas_list(table_data['meta'])
                    target_sent = dataset_record['output_sentence']
                    prefix = dataset_record['prefixes'][prefix_id]

                    prefix_htmls.append(build_prefix_pred_html(prefix, pred_sents, metrics))
                else:
                    record_id = prediction['record_id']
                    dataset_record = dataset_records[record_id]
                    prefix_id = prediction['prefix_id']
                    prefix = dataset_record['prefixes'][prefix_id]
                    pred_sents = prediction['predictions']
                    metrics = prediction['metrics']

                    prefix_htmls.append(build_prefix_pred_html(prefix, pred_sents, metrics))
            else:
                # A new dataset record
                record_id = prediction['record_id']
                prefix_id = prediction['prefix_id']
                pred_sents = prediction['predictions']
                metrics = prediction['metrics']

                dataset_record = dataset_records[record_id]
                table_data = load_json_data(f"{tables_dir}/{dataset_record['table_id']}.json")
                table, metas = table_data['data'], get_metas_list(table_data['meta'])
                target_sent = dataset_record['output_sentence']
                prefix = dataset_record['prefixes'][prefix_id]

                prefix_htmls.append(build_prefix_pred_html(prefix, pred_sents, metrics))
        # Process last record
        if prefix_htmls:
            record_htmls.append(build_record_html(record_id, table, metas, target_sent, prefix_htmls))
            prefix_htmls = []

        with open(output_path, 'w') as f:
            html_header = get_html_header(f"Results for task: {os.path.basename(output_path)}")
            f.write(html_header)
            f.write("<br>".join(record_htmls))
            f.write("</body></html>")
