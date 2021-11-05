import os
import argparse
from typing import Optional
from dataclasses import dataclass, field

import torch
import datasets
import sacrebleu
from transformers import (
    BertTokenizerFast,
    BertTokenizer,
    BertGenerationEncoder,
    BertGenerationDecoder,
    EncoderDecoderModel,
    TrainingArguments
)

from model import Seq2SeqTrainer
from utils.data_utils import inc_load_cache
from language.totto.totto_to_twt_utils import ADDITIONAL_SPECIAL_TOKENS


ENCODE_MAX_LENGTH = 512
DECODE_MAX_LENGTH = 128


def collate_fn(batch):
    return torch.utils.data.dataloader.default_collate(batch)


class CachedDataset():
    def __init__(self, model_inputs, linearized_model_inputs):
        self.model_inputs = model_inputs
        self.linearized_model_inputs = linearized_model_inputs

    def __getitem__(self, index):
        model_inputs = self.model_inputs[index]
        # Load corresponding linearized inputs
        record_id = model_inputs['record_id']
        liearized_model_inputs = self.linearized_model_inputs[record_id]
        input_item = {}
        # Modify input_ids and attention_mask with linearized inputs
        input_item['input_ids'] = liearized_model_inputs['input_ids'].squeeze(0)
        input_item['attention_mask'] = liearized_model_inputs['attention_mask'].squeeze(0)
        input_item['decoder_input_ids'] = torch.tensor(model_inputs['output_input_ids'])
        input_item['decoder_attention_mask'] = torch.tensor(model_inputs['output_attention_mask'])
        input_item["labels"] = model_inputs['output_labels']
        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        input_item["labels"] = [-100 if token == 0 else token for token in input_item["labels"]]
        input_item["labels"] = torch.tensor(input_item["labels"])
        return input_item

    def __len__(self):
        return len(self.model_inputs)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": "Which lr scheduler to use."}
    )


class BertSeq2Seq(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()

        # Create output dir if it doesn't exist
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)

    def _init_tokenizer(self):
        tokenizer = BertTokenizerFast.from_pretrained(f"bert-{self.args.model_size}-uncased")
        tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})

        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token

        return tokenizer

    def _init_model(self):
        # use BERT's cls token as BOS token and sep token as EOS token
        encoder = BertGenerationEncoder.from_pretrained(f"bert-{self.args.model_size}-uncased", bos_token_id=101, eos_token_id=102)
        # resize token embeddings
        encoder.resize_token_embeddings(len(self.tokenizer))
        # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
        decoder = BertGenerationDecoder.from_pretrained(f"bert-{self.args.model_size}-uncased", add_cross_attention=True, is_decoder=True, output_attentions=True, output_hidden_states=True, bos_token_id=101, eos_token_id=102)
        # resize token embeddings
        decoder.resize_token_embeddings(len(self.tokenizer))

        # load model
        bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # set special tokens
        bert2bert.config.decoder_start_token_id = self.tokenizer.bos_token_id
        bert2bert.config.eos_token_id = self.tokenizer.eos_token_id
        bert2bert.config.pad_token_id = self.tokenizer.pad_token_id

        # sensible parameters for beam search
        bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
        bert2bert.config.max_length = 50
        # bert2bert.config.min_length = 10
        # bert2bert.config.no_repeat_ngram_size = 3
        bert2bert.config.early_stopping = True
        # bert2bert.config.length_penalty = 2.0
        bert2bert.config.num_beams = 4

        return bert2bert

    def compute_metrics(self, pred):
        # load rouge for validation
        rouge = datasets.load_metric("rouge")

        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        sacrebleu_output = sacrebleu.corpus_bleu(pred_str, [label_str])
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        with open(self.args.output_dir + "/eval_label_" + str(round(sacrebleu_output.score, 4)) + ".txt", 'w') as f:
            f.write(os.linesep.join(label_str))
        with open(self.args.output_dir + "/eval_predict_" + str(round(sacrebleu_output.score, 4)) + ".txt", 'w') as f:
            f.write(os.linesep.join(pred_str))

        return {
            "sacrebleu_score": round(sacrebleu_output.score, 4),
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    def train(self):
        checkpoints_dir = self.args.output_dir + "/checkpoints/"
        # Create checkpoint dir if it doesn't exist
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)

        # set training arguments - these params are not really tuned, feel free to change
        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoints_dir,

            learning_rate=self.args.learning_rate,
            per_device_train_batch_size=self.args.train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            per_device_eval_batch_size=self.args.eval_batch_size,
            # eval_accumulation_steps=4,
            predict_with_generate=True,
            evaluation_strategy=self.args.evaluation_strategy,
            do_train=self.args.do_train,
            do_eval=self.args.do_eval,
            logging_steps=self.args.logging_steps,  # set to 1000 for full training
            save_steps=self.args.save_steps,  # set to 500 for full training
            eval_steps=self.args.eval_steps,  # set to 8000 for full training
            warmup_steps=self.args.warmup_steps,  # set to 2000 for full training
            max_steps=self.args.max_steps,  # delete for full training
            num_train_epochs=self.args.num_train_epochs,

            overwrite_output_dir=True,
            # save_total_limit=3,
            fp16=False,

            # load_best_model_at_end=True,
            # metric_for_best_model="sacrebleu_score",
            # greater_is_better=True
        )

        # load data
        train_model_inputs = inc_load_cache(self.args.train_model_inputs_file)
        train_linearized_model_input = inc_load_cache(self.args.train_linearized_model_inputs_file)
        print(f"Train data size: {str(len(train_model_inputs))}")
        train_dataset = CachedDataset(train_model_inputs, train_linearized_model_input)

        val_model_inputs = inc_load_cache(self.args.val_model_inputs_file)
        val_linearized_model_input = inc_load_cache(self.args.val_linearized_model_inputs_file)
        print(f"Dev data size: {str(len(val_model_inputs))}")
        val_dataset = CachedDataset(val_model_inputs, val_linearized_model_input)

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size",
                        type=str,
                        default="base",
                        help="Model size: base or large")
    parser.add_argument("--train_model_inputs_file",
                        type=str,
                        default="./data/cache/twt/totto_random_prefix_bert2bert_base_train_model_inputs.pkl.gz",
                        help="Path of the train file")
    parser.add_argument("--train_linearized_model_inputs_file",
                        type=str,
                        default="./data/cache/twt/totto_random_linearized_bert2bert_base_train_model_inputs.pkl.gz",
                        help="Path of the train file")
    parser.add_argument("--val_model_inputs_file",
                        type=str,
                        default="./data/cache/twt/totto_random_prefix_bert2bert_base_dev_model_inputs.pkl.gz",
                        help='Path of the validation file')
    parser.add_argument("--val_linearized_model_inputs_file",
                        type=str,
                        default="./data/cache/twt/totto_random_linearized_bert2bert_base_dev_model_inputs.pkl.gz",
                        help='Path of the validation file')
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output",
                        help="Output dirctory")

    parser.add_argument("--do_train",
                        action="store_true",
                        default=True,
                        help="If do training")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="Learning rate")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=8,
                        help="Train batch size")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=30,
                        help="Training epochs")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=2000,
                        help="Warmup steps")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=1000,
                        help="Logging steps")
    parser.add_argument("--save_steps",
                        type=int,
                        default=2000,
                        help="Model save steps")
    parser.add_argument("--max_steps",
                        type=int,
                        default=40000,
                        help="Max steps to train")

    parser.add_argument("--do_eval",
                        action="store_true",
                        default=False,
                        help="If do evaluation")
    parser.add_argument("--evaluation_strategy",
                        type=str,
                        default="no",
                        help="Evaluation strategy: steps or no")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=2,
                        help="Evaluation batch size")
    parser.add_argument("--eval_steps",
                        type=int,
                        default=2000,
                        help="Model save steps")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    bert_seq2seq = BertSeq2Seq(args)
    bert_seq2seq.train()


if __name__ == '__main__':
    main()
