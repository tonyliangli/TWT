# TWT: Table with Written Text for Controlled Data-to-Text Generation

This is the source code for our paper [TWT: Table with Written Text for Controlled Data-to-Text Generation](https://underline.io/lecture/38303-twt-table-with-written-text-for-controlled-data-to-text-generation) (Findings of EMNLP 2021).

In this paper, we propose to generate text conditioned on the structured data (table) and a preﬁx (the written text) by leveraging the pretrained models.
We present a new dataset, Table with Written Text (TWT), by repurposing two existing datasets: ToTTo and TabFact. TWT contains both factual and logical statements that are faithful to the structured data.
Compared with other datasets, TWT is of practical use, the preﬁx controls the topic of the generated text, and the output model could provide intelligent assistance for writing with structured data.

## Preliminaries

### Enviroment Setup

The baseline codes use Python 3.7 and Pytorch 1.6.0.
Install Python dependency: `pip install -r requirements.txt`.

### Download the TWT dataset

The TWT dataset can be downloaded from [here](https://drive.google.com/file/d/1U6JVytDIMwlxGiz--UqdCWt1LQAQ5aAR/view?usp=sharing). Place it in the `data/dataset` folder.
Run the following commands:

```shell
unzip twt.zip
cd twt
unzip tables.zip
```

### Dataset Preprocessing

```shell
python twt_preprocessing.py --model_type=t5 --model_size=base --pattern=prefix --is_random
```

Options for `--model_type`: t5, bert2bert. Please refer to the `twt_preprocessing.py` to find more instructions for different arguments.

The pickled training, validation and test files will be generated in `data/cache`.

### Train with Validation

All training scripts are placed in the `tasks/train/tabfact` and `tasks/train/totto` folder.

For example, to train our model initialized with T5 on the ToTTo source, run:

```shell
bash tasks/train/random/totto/twt_t5/base/totto_random_prefix_t5_base_twt_0.4_gen_loss.sh
```

To train the T5 baseline model on the Totto source, run:

```shell
bash tasks/train/random/totto/t5/base/totto_random_causal_t5_base.sh
```

Note that you must run the scrip under the project root. By default the script loads the pickled files from `data/cache` and outputs the checkpoints along with evaluation results to the directory defined in the script.

### Run Prediction

You may find three prediction scrips in the `tasks/pred` folder.

Run `run_distributed_pred.sh` to start automatic prediction. The script will scan for the best checkpoint under the `input_dir` defined in the script.

```shell
bash tasks/pred/run_distributed_pred.sh
```

Run `run_pred_base.sh` to predict on the TWT test set with a baseline mode.

```shell
bash tasks/pred/run_pred_bash.sh
```

Run `run_pred_twt.sh` to predict on the TWT test set with our model.

```shell
bash tasks/pred/run_pred_twt.sh
```

You may change the default input and output direction inside the script.


## Questions

If you have any question, please go ahead and [open an issue](https://github.com/tonyliangli/TWT/issues).


## Citation

```
@inproceedings{li-etal-2021-twt,
    title = "TWT: Table with Written Text for Controlled Data-to-Text Generation",
    author = "Li, Tongliang and
      Fang, Lei and
      Jian-Guang, Lou and
      Li, Zhoujun",
    year = "2021"
}
```
