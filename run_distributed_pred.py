import os
import copy
import time
import argparse
import shutil
import traceback
import torch

from utils.data_utils import load_json_data
from twt_split_inputs import split_model_inputs
from run_distributed_eval import do_eval
from twt_visualize import visualize_data


DATASET_NAMES = ["totto", "tabfact"]
PATTERNS = ["causal", "prefix"]
MODEL_TYPES = ["bert2bert", "t5", "bart"]
MODEL_SIZES = ["base", "large"]
MODEL_CUSTOMS = ["twt"]


def parse_task_name(task_name):
    dataset_name, pattern, model_type, model_size, model_custom = None, None, None, None, None
    if task_name and "_" in task_name:
        task_name_split = task_name.split("_")
        if len(task_name_split) >= 4:
            if task_name_split[0] in DATASET_NAMES:
                dataset_name = task_name_split[0]
            if task_name_split[1] == "random" and task_name_split[2] in PATTERNS:
                pattern = f"{task_name_split[1]}_{task_name_split[2]}"
                if task_name_split[3] in MODEL_TYPES:
                    model_type = task_name_split[3]
                if task_name_split[4] in MODEL_SIZES:
                    model_size = task_name_split[4]
                if len(task_name_split) > 5:
                    if task_name_split[5] in MODEL_CUSTOMS:
                        model_custom = task_name_split[5]
                else:
                    model_custom = ""
            else:
                if task_name_split[1] in PATTERNS:
                    pattern = task_name_split[1]
                if task_name_split[2] in MODEL_TYPES:
                    model_type = task_name_split[2]
                if task_name_split[3] in MODEL_SIZES:
                    model_size = task_name_split[3]
                if len(task_name_split) > 4:
                    if task_name_split[4] in MODEL_CUSTOMS:
                        model_custom = task_name_split[4]
                else:
                    model_custom = ""
    return dataset_name, pattern, model_type, model_size, model_custom


def prepare_task_args(task_name, task_name_parsed, task_dir, task_best_checkpoint, args):
    task_args = copy.deepcopy(args)
    dataset_name, pattern, model_type, model_size, model_custom = task_name_parsed
    if args.use_random_testset:
        print(f"Predict with {dataset_name} random {task_args.dataset_split} set")
        task_args.dataset_path = f"{task_args.temp_dir}/input/{dataset_name}_random_{task_args.dataset_split}.jsonl"
        task_args.model_inputs_path = f"{task_args.temp_dir}/input/{dataset_name}_random_prefix_{model_type}_{model_size}_{task_args.dataset_split}_model_inputs.pkl.gz"
        if model_custom != "twt":
            task_args.linearized_model_inputs_path = f"{task_args.temp_dir}/input/{dataset_name}_random_linearized_{model_type}_{model_size}_{task_args.dataset_split}_model_inputs.pkl.gz"
    else:
        print(f"Predict with {dataset_name} twt {task_args.dataset_split} set")
        task_args.dataset_path = f"{task_args.temp_dir}/input/{dataset_name}_{task_args.dataset_split}.jsonl"
        task_args.model_inputs_path = f"{task_args.temp_dir}/input/{dataset_name}_prefix_{model_type}_{model_size}_{task_args.dataset_split}_model_inputs.pkl.gz"
        if model_custom != "twt":
            task_args.linearized_model_inputs_path = f"{task_args.temp_dir}/input/{dataset_name}_linearized_{model_type}_{model_size}_{task_args.dataset_split}_model_inputs.pkl.gz"
    task_args.tables_dir = f"{task_args.temp_dir}/input/tables"

    task_args.devided_model_inputs_dir = f"{task_args.temp_dir}/devided_input"
    task_args.devided_predictions_dir = f"{task_args.temp_dir}/devided_output/{task_name}"
    task_args.reduced_prediction_path = os.path.join(task_dir, f"{'random_' if args.use_random_testset else ''}{task_args.dataset_split}_set_predictions", f"{task_name}.jsonl")
    task_args.model_type = "_".join([model_item for model_item in (model_custom, model_type) if model_item])
    task_args.model_name_or_path = os.path.join(task_dir, "checkpoints", task_best_checkpoint)
    if model_type == "bert2bert":
        task_args.tokenizer_name_or_path = f"bert-{model_size}-uncased"
        task_args.stop_token = "[SEP]"
        task_args.greedy_search = False
    elif model_type == "t5":
        task_args.tokenizer_name_or_path = f"t5-{model_size}"
        task_args.stop_token = "</s>"
        task_args.greedy_search = True
    elif model_type == "bart":
        task_args.tokenizer_name_or_path = f"facebook/bart-{model_size}"
        task_args.stop_token = "</s>"
        task_args.greedy_search = False
    else:
        pass
    if "without_struct_embed" in task_name:
        task_args.no_row_col_embeddings = True
    task_args.task_dir = task_dir
    return task_args


def prepare_task_files(task_args):
    # Create summary directories
    if not os.path.exists(task_args.summary_dir):
        os.makedirs(task_args.summary_dir, exist_ok=True)
    if not os.path.exists(os.path.join(task_args.summary_dir, "predictions")):
        os.makedirs(os.path.join(task_args.summary_dir, "predictions"), exist_ok=True)
    if not os.path.exists(os.path.join(task_args.summary_dir, "metrics")):
        os.makedirs(os.path.join(task_args.summary_dir, "metrics"), exist_ok=True)
    if not os.path.exists(os.path.join(task_args.summary_dir, "visualized")):
        os.makedirs(os.path.join(task_args.summary_dir, "visualized"), exist_ok=True)
    # Create predictions directories
    if not os.path.exists(task_args.devided_predictions_dir):
        os.makedirs(task_args.devided_predictions_dir, exist_ok=True)
    # Prepare split model inputs
    if os.path.exists(task_args.model_inputs_path):
        first_split_path = os.path.join(task_args.devided_model_inputs_dir,
                                        "0",
                                        os.path.basename(task_args.model_inputs_path))
        if not os.path.exists(first_split_path):
            split_model_inputs(task_args)
    else:
        print(f"Model inputs not found: {task_args.model_inputs_path}")


def prepare_prev_predictions(args):
    task_best_checkpoints = {}
    pred_metrics_dir = os.path.join(args.summary_dir, "metrics")
    if os.path.exists(pred_metrics_dir):
        for metric_file_name in os.listdir(pred_metrics_dir):
            if f"_{args.dataset_split}_set_metrics.json" in metric_file_name:
                metric_file_path = os.path.join(pred_metrics_dir, metric_file_name)
                metric_json_data = load_json_data(metric_file_path)
                if 'checkpoint' in metric_json_data and len(metric_json_data['checkpoint']) > 0:
                    task_best_checkpoint = metric_json_data['checkpoint'][0]
                    suffix_start_idx = metric_file_name.index(f"_{args.dataset_split}_set_metrics.json")
                    task_name = metric_file_name[:suffix_start_idx]
                    if task_name not in task_best_checkpoints:
                        task_best_checkpoints[task_name] = []
                    # Save checkpoint history
                    task_best_checkpoints[task_name].append(task_best_checkpoint)
    return task_best_checkpoints


def prepare_task_best_checkpoint(task_dir):
    best_checkpoint = None
    if os.path.exists(task_dir):
        best_checkpoint_record_file_path = os.path.join(task_dir, "predictions", "best_metrics.json")
        if os.path.exists(best_checkpoint_record_file_path):
            best_checkpoint_record = load_json_data(best_checkpoint_record_file_path)
            if 'checkpoint' in best_checkpoint_record:
                best_checkpoint = best_checkpoint_record['checkpoint']
        else:
            print(f"Best checkpoint record file not found: {best_checkpoint_record_file_path}")
    else:
        print(f"Task directory not found: f{task_dir}")
    return best_checkpoint


def is_valid_task_name(task_name_parsed):
    return all(task_name_split is not None for task_name_split in task_name_parsed)


def predict_with_task_best_checkpoint(task_args):
    task_name = os.path.basename(task_args.task_dir)
    task_best_checkpoint = os.path.basename(task_args.model_name_or_path)
    print(f"Start predicting for {task_name} with {task_best_checkpoint}")
    do_eval(task_args)
    # Follow the path rules from distributed evaluation
    reduced_prediction_dir = os.path.join(os.path.dirname(task_args.reduced_prediction_path), task_best_checkpoint)
    predictions_file_path = os.path.join(reduced_prediction_dir, os.path.basename(task_args.reduced_prediction_path))
    metrics_file_path = os.path.join(reduced_prediction_dir, f"{os.path.basename(task_args.reduced_prediction_path)}_metrics.json")
    if os.path.exists(predictions_file_path) and os.path.exists(metrics_file_path):
        return predictions_file_path, metrics_file_path
    else:
        print(f"Predictions or metrics file not found for {task_name} with {task_best_checkpoint}")
    return None, None


def summarize_predictions(predictions_file_path, metrics_file_path, task_args):
    if predictions_file_path and metrics_file_path:
        task_name = os.path.basename(task_args.task_dir)
        summary_predictions_file_path = os.path.join(task_args.summary_dir, "predictions", f"{task_name}_{task_args.dataset_split}_set_predictions.jsonl")
        summary_metrics_file_path = os.path.join(task_args.summary_dir, "metrics", f"{task_name}_{task_args.dataset_split}_set_metrics.json")
        if os.path.exists(summary_predictions_file_path) and os.path.exists(summary_metrics_file_path):
            task_best_checkpoint = os.path.basename(task_args.model_name_or_path)
            print(f"New prediction results generated for {task_name} with {task_best_checkpoint}")
        # Copy prediction results and metrics to summary directory
        shutil.copy(predictions_file_path, summary_predictions_file_path)
        shutil.copy(metrics_file_path, summary_metrics_file_path)
        if os.path.exists(summary_predictions_file_path) and os.path.exists(summary_metrics_file_path):
            return summary_predictions_file_path, summary_metrics_file_path
    return None, None


def visualize_predictions(predictions_file_path, task_args):
    visualize_file_path = os.path.join(task_args.summary_dir, "visualized", f"{os.path.basename(task_args.task_dir)}_{task_args.dataset_split}_set_visualized.html")
    visualize_data(predictions_file_path, task_args.dataset_path, task_args.tables_dir, visualize_file_path)
    if os.path.exists(visualize_file_path):
        return visualize_file_path
    return None


def do_prediction(args):
    task_best_checkpoints = prepare_prev_predictions(args)
    while True:
        print("Scanning best checkpoints")
        task_names = os.listdir(args.input_dir)
        for task_name in task_names:
            try:
                task_dir = os.path.join(args.input_dir, task_name)
                # Parse task name
                task_name_parsed = parse_task_name(task_name)
                if is_valid_task_name(task_name_parsed):
                    task_best_checkpoint = prepare_task_best_checkpoint(task_dir)
                    if task_best_checkpoint is not None:
                        # Prepare args for current task
                        task_args = prepare_task_args(task_name, task_name_parsed, task_dir, task_best_checkpoint, args)
                        if task_name not in task_best_checkpoints:
                            task_best_checkpoints[task_name] = []
                            # Prepare prediction files
                            prepare_task_files(task_args)
                        # Predict only if the best checkpoint for the current task has not been used for prediction before
                        if task_best_checkpoint not in task_best_checkpoints[task_name]:
                            # Start prediction
                            predictions_file_path, metrics_file_path = predict_with_task_best_checkpoint(task_args)
                            # Summarize predictions
                            summary_predictions_file_path, _ = summarize_predictions(predictions_file_path, metrics_file_path, task_args)
                            # Save checkpoint history
                            task_best_checkpoints[task_name].append(task_best_checkpoint)
                            # Visualize results
                            visualize_predictions(summary_predictions_file_path, task_args)
            except Exception:
                traceback.print_exc()
        time.sleep(10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help='A model directory or a standard task output directory')
    parser.add_argument("--temp_dir",
                        type=str,
                        required=True,
                        help='Directory to save temp files for prediction')
    parser.add_argument("--summary_dir",
                        type=str,
                        required=True,
                        help='Directory to save summarized results')
    parser.add_argument("--dataset_split",
                        type=str,
                        default="test",
                        help='The dataset split to predict')
    parser.add_argument("--split_num",
                        type=int,
                        required=True,
                        help='How many splits will be generated')
    # parser.add_argument("--predict_all_tasks",
    #                     action='store_true',
    #                     default=False,
    #                     help='Whether to predict all tasks, the models_dir must be a standard task output directory')
    # parser.add_argument("--scan_new_best_checkpoints",
    #                     action='store_true',
    #                     default=False,
    #                     help='If set to true, the process will keep running to scan new checkpoints')

    # parser.add_argument(
    #     "--greedy_search",
    #     action="store_true",
    #     default=False,
    #     help="Whether to use greedy search",
    # )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help="The number of samples to generate."
    )
    parser.add_argument(
        "--use_random_testset",
        action="store_true",
        default=False,
        help="Whether to use the random test set",
    )

    args = parser.parse_args()
    return args


def set_default_args(args):
    args.daemon_mode = False
    args.length = 20
    args.stop_token = "</s>"
    args.temperature = 1.0
    args.repetition_penalty = 1.0
    args.k = 0
    args.p = 0.9
    args.greedy_search = False
    args.no_row_col_embeddings = False
    args.prefix = ""
    args.seed = 42
    args.no_cuda = False
    args.fp16 = False
    args.metric_for_best_model = "avg#bleu_score"
    args.greater_is_better = True
    args.clean_up_tokenization_spaces = True
    args.use_model_based_metrics = True

    return args


def main():
    args = parse_args()
    args = set_default_args(args)
    print(args)
    do_prediction(args)


if __name__ == "__main__":
    # Set torch multi processing
    torch.multiprocessing.set_start_method('spawn')
    main()
