import os
import copy
import json
import time
import argparse
import collections
import multiprocessing as mp
import torch

from eval_metrics import METRIC_TEMPLATE
from run_pred import MODEL_CLASSES, prepare_generation, predict_dataset
from utils.data_utils import gather_jsonl_data

DAEMON_TIMEOUT_SECONDS = 86400
CHECKPOINT_SCAN_SLEEP_SECONDS = 10


def predict_chunk(args):
    def predict():
        # Predict
        model, tokenizer, gen_args = prepare_generation(args)
        results = predict_dataset(model, tokenizer, gen_args)
        devided_prediction_path = os.path.join(args.devided_predictions_dir,
                                               os.path.basename(args.model_name_or_path),
                                               split_id,
                                               os.path.basename(args.reduced_prediction_path))
        os.makedirs(os.path.dirname(devided_prediction_path), exist_ok=True)
        with open(devided_prediction_path, 'w', encoding="utf-8") as f:
            f.write(os.linesep.join(results))

    split_id = str(args.cuda_device_id)
    try:
        # Set current gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_id)
        torch.cuda.set_device(int(args.cuda_device_id))
        print(f"Predict split {split_id} with CUDA: {os.environ['CUDA_VISIBLE_DEVICES']}.")
        # Do prediction
        predict()
    except RuntimeError as e:
        args.no_cuda = True
        print(e)
        print(f"GPU memory overflow. Use CPU to predict split {split_id} instead.")
        # Do prediction again
        predict()


def distributed_eval(args):
    if os.path.exists(args.devided_model_inputs_dir) and os.path.isdir(args.devided_model_inputs_dir):
        processes = []
        for i in range(0, args.split_num):
            args_copy = copy.deepcopy(args)
            args_copy.cuda_device_id = str(i)
            args_copy.model_inputs_path = os.path.join(args_copy.devided_model_inputs_dir,
                                                       str(i),
                                                       os.path.basename(args_copy.model_inputs_path))
            # predict_chunk(args_copy)
            process = mp.Process(target=predict_chunk, args=(args_copy,))
            process.start()
            processes.append(process)
        # Wait for sub processes to finish
        for process in processes:
            process.join()

        # Merge predictions
        all_results = []
        for i in range(0, args.split_num):
            devided_prediction_path = os.path.join(args.devided_predictions_dir,
                                                   os.path.basename(args.model_name_or_path),
                                                   str(i),
                                                   os.path.basename(args.reduced_prediction_path))
            all_results += gather_jsonl_data(devided_prediction_path)
            # with open(devided_prediction_path, 'r') as f:
            #     for line in f:
            #         if line:
            #             all_results.append(line.strip())

        # Write reduced result
        reduced_prediction_dir = os.path.join(os.path.dirname(args.reduced_prediction_path), os.path.basename(args.model_name_or_path))
        os.makedirs(reduced_prediction_dir, exist_ok=True)
        with open(os.path.join(reduced_prediction_dir, os.path.basename(args.reduced_prediction_path)), 'w', encoding="utf-8") as f:
            f.write(os.linesep.join([json.dumps(all_result) for all_result in all_results]))
        return all_results
    else:
        print("Distributed model inputs don't exist.")


def get_new_checkpoint_steps(used_checkpoint_steps, checkpoints_dir):
    checkpoint_steps = []
    for checkpoint_dir in os.listdir(checkpoints_dir):
        if "checkpoint-" in checkpoint_dir:
            checkpoint_step = checkpoint_dir.lstrip("checkpoint-")
            checkpoint_steps.append(checkpoint_step)
    if used_checkpoint_steps:
        remaining_checkpoint_steps = list(set(checkpoint_steps).difference(used_checkpoint_steps))
        return list(sorted(remaining_checkpoint_steps))
    else:
        return list(sorted(checkpoint_steps))


def parse_best_metric(best_metric):
    if "#" in best_metric:
        parsed_calc, parsed_metric = best_metric.split("#")
        if parsed_calc in ["avg", "max"] and parsed_metric in METRIC_TEMPLATE:
            return parsed_calc, parsed_metric
    return "avg", list(METRIC_TEMPLATE.keys())[0]


def summarize_metrics(results):
    metric_values, metrics_summaries = collections.OrderedDict(), collections.OrderedDict()
    if results:
        if 'metrics' in results[0]:
            # Get all metric types
            calc_types = list(results[0]['metrics'].keys())
            metric_types = list(results[0]['metrics'][calc_types[0]].keys())
            for result in results:
                for calc_type in calc_types:
                    if calc_type not in metric_values:
                        metric_values[calc_type] = collections.OrderedDict()
                    for metric_type in metric_types:
                        if metric_type not in metric_values[calc_type]:
                            metric_values[calc_type][metric_type] = []
                        metric_values[calc_type][metric_type].append(result['metrics'][calc_type][metric_type])

            for calc_type, metric_type_values in metric_values.items():
                if calc_type not in metrics_summaries:
                    metrics_summaries[calc_type] = collections.OrderedDict()
                for metric_type, values in metric_type_values.items():
                    metrics_summaries[calc_type][metric_type] = sum(values) / len(values)
    return metrics_summaries


def do_eval(args):
    if args.daemon_mode:
        if os.path.exists(args.checkpoints_dir) and os.path.isdir(args.checkpoints_dir):
            used_checkpoint_steps, checkpoints_metrics = [], []
            start_time = time.time()
            while True:
                # In case the new generated checkpoint is incomplete
                try:
                    print("Scanning checkpoints")
                    checkpoint_steps = get_new_checkpoint_steps(used_checkpoint_steps, args.checkpoints_dir)
                    if checkpoint_steps:
                        for checkpoint_step in checkpoint_steps:
                            print(f"New checkpoint: checkpoint-{checkpoint_step} found")
                            # Wait for a while in case the checkpoint is not fully written
                            time.sleep(10)
                            args.model_name_or_path = os.path.join(args.checkpoints_dir, f"checkpoint-{checkpoint_step}")
                            results = distributed_eval(args)
                            used_checkpoint_steps.append(checkpoint_step)
                            # Summarize metrics
                            metrics_summaries = summarize_metrics(results)
                            metrics_summaries['checkpoint'] = f"checkpoint-{checkpoint_step}"
                            checkpoints_metrics.append(metrics_summaries)
                            # Write metrics
                            write_metrics_path = os.path.join(os.path.dirname(args.reduced_prediction_path), "metrics_state.json")
                            with open(write_metrics_path, 'w', encoding="utf-8") as f:
                                json.dump(checkpoints_metrics, f, indent=4)
                            print(f"checkpoint-{checkpoint_step} evaluation completed")
                            # Update start time
                            start_time = time.time()
                        # Write best metrics
                        if checkpoints_metrics:
                            best_calc_type, best_metric_type = parse_best_metric(args.metric_for_best_model)
                            best_metrics = list(sorted(checkpoints_metrics, key=lambda item: item[best_calc_type][best_metric_type], reverse=args.greater_is_better))
                            best_metrics_path = os.path.join(os.path.dirname(args.reduced_prediction_path), "best_metrics.json")
                            with open(best_metrics_path, 'w', encoding="utf-8") as f:
                                json.dump(best_metrics[0], f, indent=4)

                    time.sleep(CHECKPOINT_SCAN_SLEEP_SECONDS)
                    if time.time() - start_time >= DAEMON_TIMEOUT_SECONDS:
                        break
                except Exception:
                    pass
    else:
        if os.path.exists(args.model_name_or_path):
            # Get checkpoint name
            checkpoint_name = os.path.basename(args.model_name_or_path),
            results = distributed_eval(args)
            # Summarize metrics
            metrics_summaries = summarize_metrics(results)
            metrics_summaries['checkpoint'] = checkpoint_name
            # Write metrics
            reduced_prediction_dir = os.path.join(os.path.dirname(args.reduced_prediction_path), os.path.basename(args.model_name_or_path))
            write_metrics_path = os.path.join(reduced_prediction_dir, f"{os.path.basename(args.reduced_prediction_path)}_metrics.json")
            with open(write_metrics_path, 'w', encoding="utf-8") as f:
                json.dump(metrics_summaries, f, indent=4)
            print(f"{checkpoint_name} evaluation completed")
        else:
            print("Model chekpoints directory not found.")


def parse_args():
    parser = argparse.ArgumentParser()

    # For distribution
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help='File path of the dataset')
    parser.add_argument("--tables_dir",
                        type=str,
                        required=True,
                        help='Directory of the table files')
    parser.add_argument("--model_inputs_path",
                        type=str,
                        required=True,
                        help='File path of the cached model inputs')
    parser.add_argument("--linearized_model_inputs_path",
                        type=str,
                        help='File path of the cached linearized model inputs')
    parser.add_argument("--devided_model_inputs_dir",
                        type=str,
                        required=True,
                        help='Directory of the deviced chached model inputs')
    parser.add_argument("--devided_predictions_dir",
                        type=str,
                        required=True,
                        help='Directory of the devided predictions')
    parser.add_argument("--reduced_prediction_path",
                        required=True,
                        type=str,
                        help='Directory of the reduced prediction')
    parser.add_argument("--split_num",
                        type=int,
                        required=True,
                        help='How many splits will be generated')
    parser.add_argument("--daemon_mode",
                        action='store_true',
                        default=False,
                        help='If set to true, the process will keep running to scan new checkpoints')
    parser.add_argument("--checkpoints_dir",
                        # required=True,
                        type=str,
                        help='Directory of the reduced prediction')

    # For prediction
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

    # For evaluation
    parser.add_argument("--metric_for_best_model",
                        type=str,
                        default="avg#bleu_score",
                        help='The metric for choosing the best model checkpoint')
    parser.add_argument("--greater_is_better",
                        action='store_true',
                        default=True,
                        help='Whether the value of the metric for best model is the greater the better')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    do_eval(args)


if __name__ == "__main__":
    # Set torch multi processing
    torch.multiprocessing.set_start_method('spawn')
    main()
