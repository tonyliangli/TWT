import os
import sys
import math
import argparse
from utils.data_utils import inc_dump_cache, inc_load_cache, load_jsonl_data


CACHE_DATA_FREQ = 1000


def align_model_input_with_dataset(dataset_path, model_inputs_path, output_dir):
    # Load model input cache
    model_inputs = inc_load_cache(model_inputs_path)
    aligned_inputs = []
    current_id = 0
    for record_id, json_data in enumerate(load_jsonl_data(dataset_path, True, False)):
        for prefix_id, (prefix_str, prefix_start_idx) in enumerate(zip(json_data['prefixes'], json_data['start_indices'])):
            model_input = model_inputs[current_id]
            model_input['record_id'] = record_id
            model_input['prefix_id'] = prefix_id
            # model_input['prefix_str'] = prefix_str
            # model_input['prefix_start_idx'] = prefix_start_idx
            # model_input['output_sentence'] = json_data['output_sentence']
            aligned_inputs.append(model_input)
            current_id += 1
            # Dump data
            if len(aligned_inputs) % CACHE_DATA_FREQ == 0:
                inc_dump_cache(os.path.join(output_dir, os.path.basename(model_inputs_path)), aligned_inputs)
                aligned_inputs = []
    if aligned_inputs:
        inc_dump_cache(os.path.join(output_dir, os.path.basename(model_inputs_path)), aligned_inputs)
        aligned_inputs = []

    assert current_id == len(model_inputs)


def chunk_data(data, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def split_model_inputs(args):
    print("Load cached model inputs from data")
    model_inputs = inc_load_cache(args.model_inputs_path)
    chunk_size = math.ceil(len(model_inputs) / args.split_num)
    total_count = 0
    for i, chunk in enumerate(chunk_data(model_inputs, chunk_size)):
        current_path = os.path.join(args.devided_model_inputs_dir,
                                    str(i),
                                    os.path.basename(args.model_inputs_path))
        # Remove existing file
        if os.path.exists(current_path):
            os.remove(current_path)
        os.makedirs(os.path.dirname(current_path), exist_ok=True)
        # Dump split cache file
        inc_dump_cache(current_path, chunk)
        print(f"Create new chunk file {current_path} with {len(chunk)} records")
        total_count += len(chunk)
    assert total_count == len(model_inputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_num",
                        type=int,
                        required=True,
                        help='how many splits will be generated')
    parser.add_argument("--model_inputs_path",
                        type=str,
                        required=True,
                        help='File path of the cached model inputs')
    parser.add_argument("--devided_model_inputs_dir",
                        type=str,
                        required=True,
                        help='Directory of the deviced chached model inputs')

    args = parser.parse_args()
    return args


def main():
    # dataset_path = "./data/dataset/final/tabfact_dev.jsonl"
    # model_inputs_path = "./data/cache/final/tabfact_t5_prefix_dev_model_inputs.pkl.gz"
    # output_dir = "./data/cache/final_aligned"
    # align_model_input_with_dataset(dataset_path, model_inputs_path, output_dir)
    args = parse_args()
    split_model_inputs(args)


if __name__ == "__main__":
    main()
