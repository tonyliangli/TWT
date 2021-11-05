import re
import sacrebleu
import collections

from language.totto.totto_to_twt_utils import get_parent_tables
from language.totto.totto_parent_eval import _normalize_text, overlap_probability, parent
from language.tabfact.preprocess_data import align_inputs, get_lemmatize
from utils.data_utils import tokenize, load_json_data, load_jsonl_data
from twt_preprocessing import get_metas_list


BLEURT_SCORER, BERT_SCORER = None, None
METRIC_TEMPLATE = collections.OrderedDict({
        'bleu_score': 0,
        'bleurt_score': 0,
        'bert_score_precision': 0,
        'bert_score_recall': 0,
        'bert_score_f_score': 0,
        'parent_precision': 0,
        'parent_recall': 0,
        'parent_f_score': 0,
        'fact_coverage': 0,
        'em_1_ratio': 0,
        'em_2_ratio': 0,
        'em_3_ratio': 0,
        'match_rate': 0,
        'char_saved': 0
})


def load_bleurt_scorer():
    from bleurt import score as bleurt_scorer
    return bleurt_scorer.BleurtScorer()


def load_bert_scorer():
    from bert_score import BERTScorer
    return BERTScorer(lang="en", rescale_with_baseline=True)


def chunk_prediction(prediction, eos_token=None):
    # Remove EOS token if it exists
    if eos_token and eos_token in prediction:
        end_pos = prediction.index(eos_token)
    else:
        end_pos = len(prediction)
    return prediction[:end_pos]


def get_pred_after_prefix(prefix, prediction):
    try:
        pred_tokens, pred_token_offsets = tokenize(prediction)

        pred_after_prefix_tokens = []
        for i, (pred_token, pred_token_offset) in enumerate(zip(pred_tokens, pred_token_offsets)):
            if pred_token_offset[0] >= len(prefix):
                pred_after_prefix_tokens = pred_tokens[i:]
                break

        if pred_after_prefix_tokens:
            pred_after_prefix = " ".join(pred_after_prefix_tokens)
        else:
            pred_after_prefix = prediction[len(prefix):].strip()
    except Exception:
        pred_after_prefix = prediction[len(prefix):].strip()

    return pred_after_prefix


def get_target_after_prefix(prefix, target):
    target_after_prefix = target[len(prefix):].strip()
    return target_after_prefix


def parse_facts_in_target(facts_in_target, metas, pred_start_idx=0):
    all_facts = set()
    coords_to_facts = {}
    if facts_in_target:
        for i, (fact_start_idx, fact_in_target) in enumerate(facts_in_target.items()):
            # Only consider facts after the prefix
            if int(fact_start_idx) >= pred_start_idx:
                fact_str, table_coords = fact_in_target
                # Save to all facts
                all_facts.add(fact_str)
                for table_coord in table_coords:
                    # Process facts aligned to meta
                    proc_table_coord = tuple(table_coord)
                    if proc_table_coord[0] == -1:
                        if len(metas) > proc_table_coord[1]:
                            aligned_meta = metas[proc_table_coord[1]]
                            if fact_str in aligned_meta:
                                aligned_meta_start_idx = aligned_meta.index(fact_str)
                                # Note that the index here is a string offset instead of the word index
                                proc_table_coord = (-1, proc_table_coord[1], aligned_meta_start_idx)
                            else:
                                lemma_meta_tokens, _ = get_lemmatize(aligned_meta, False)
                                lemma_fact_tokens, _ = get_lemmatize(fact_str, False)
                                lemma_meta_tokenized = " ".join(lemma_meta_tokens)
                                lemma_fact_tokenized = " ".join(lemma_fact_tokens)
                                if lemma_fact_tokenized in lemma_meta_tokenized:
                                    aligned_meta_start_idx = lemma_meta_tokenized.index(lemma_fact_tokenized)
                                    proc_table_coord = (-1, proc_table_coord[1], aligned_meta_start_idx)
                                else:
                                    proc_table_coord = (-1, -1)
                        else:
                            proc_table_coord = (-1, -1)
                    if proc_table_coord not in coords_to_facts:
                        coords_to_facts[proc_table_coord] = []
                    coords_to_facts[proc_table_coord].append(fact_str)
    return all_facts, coords_to_facts


def clac_bleu(prediction, target):
    # load sacrebleu for validation
    bleu = sacrebleu.corpus_bleu([prediction], [[target]], lowercase=True, smooth_method='add-k')
    return bleu.score


def calc_parent(prediction, target, table, metas, facts_in_target):
    def _table_reader(table_line):
        entries = table_line.lower().split("\t")
        parent_table = [[
            _normalize_text(member).split() for member in entry.split("|||")
        ] for entry in entries]
        return parent_table

    def _text_reader(text_line):
        line = _normalize_text(text_line)
        return [line.split()]

    # Convert table format to parent
    table_prec, table_rec = get_parent_tables(table, metas, facts_in_target)
    precision_tables, recall_tables = [_table_reader(table_prec.lower())], [_table_reader(table_rec.lower())]

    generations = _text_reader(prediction)
    # Use multi reference style
    references = _text_reader(target)
    references = [[x] for x in references]

    precision, recall, f_score, all_f = parent(
        generations,
        references,
        precision_tables,
        recall_tables,
        entailment_fn=overlap_probability
    )

    return precision, recall, f_score


def calc_fact_coverage(prediction, table, metas, all_facts, coords_to_facts):
    # Init fact coverage stats
    fact_coverage = 0
    if all_facts:
        # Align prediction with inputs
        _, match_res = align_inputs((table, metas, prediction))
        facts_in_pred = []
        try:
            # Follow the tabfact format to support multiple output sentences (statements)
            facts_in_pred = match_res[4][0]
        except IndexError:
            facts_in_pred = []

        if facts_in_pred:
            matched_facts = set()
            for i, (fact_start_idx, fact_in_pred) in enumerate(facts_in_pred.items()):
                fact_str, table_coords = fact_in_pred
                for table_coord in table_coords:
                    # Process facts aligned to meta
                    proc_table_coord = tuple(table_coord)
                    if proc_table_coord[0] == -1:
                        if len(metas) > proc_table_coord[1]:
                            aligned_meta = metas[proc_table_coord[1]]
                            if fact_str in aligned_meta:
                                aligned_meta_start_idx = aligned_meta.index(fact_str)
                                # Note that the index here is a string offset instead of the word index
                                proc_table_coord = (-1, proc_table_coord[1], aligned_meta_start_idx)
                            else:
                                lemma_meta_tokens, _ = get_lemmatize(aligned_meta, False)
                                lemma_fact_tokens, _ = get_lemmatize(fact_str, False)
                                lemma_meta_tokenized = " ".join(lemma_meta_tokens)
                                lemma_fact_tokenized = " ".join(lemma_fact_tokens)
                                if lemma_fact_tokenized in lemma_meta_tokenized:
                                    aligned_meta_start_idx = lemma_meta_tokenized.index(lemma_fact_tokenized)
                                    proc_table_coord = (-1, proc_table_coord[1], aligned_meta_start_idx)
                                else:
                                    proc_table_coord = (-1, -1)
                        else:
                            proc_table_coord = (-1, -1)
                if proc_table_coord in coords_to_facts:
                    matched_facts.add(fact_str)

            if matched_facts:
                fact_coverage = len(matched_facts) / len(all_facts)

    return fact_coverage


def clac_EM_N(prediction, target, n=1):
    em_n_ratio = 0
    pred_after_prex_tokens, target_after_prefix_tokens = prediction.split(" "), target.split(" ")
    if pred_after_prex_tokens[:n] == target_after_prefix_tokens[:n]:
        em_n_ratio = 1
    return em_n_ratio


def calc_match_rate(prediction, target):
    match_rate = 0
    # Remove all white spaces
    pred_after_prex_no_whitespace = re.sub(r"\s+", "", prediction)
    target_after_prefix_no_white_space = re.sub(r"\s+", "", target)
    if pred_after_prex_no_whitespace == target_after_prefix_no_white_space:
        match_rate = 1
    return match_rate


def calc_char_saved(prediction, target):
    char_saved_count = 0
    pred_tokens, target_tokens = prediction.split(" "), target.split(" ")
    same_tokens = list(set(pred_tokens).intersection(set(target_tokens)))
    if same_tokens:
        for pred_token in pred_tokens:
            if pred_token in same_tokens:
                char_saved_count += len(pred_token)
    # for i, pred_token in enumerate(pred_tokens):
    #     if i < len(target_tokens):
    #         if pred_token == target_tokens[i]:
    #             char_saved_count += len(pred_token)
    #         else:
    #             # Only find continous same tokens from the beginning
    #             break
    #     else:
    #         break
    return char_saved_count


def clac_bleurt(bleurt_scorer, prediction, target):
    scores = bleurt_scorer.score(references=[target], candidates=[prediction])
    return scores[0]


def clac_bert_score(bert_scorer, prediction, target):
    P, R, F1 = bert_scorer.score([prediction], [target])
    return float(P[0]), float(R[0]), float(F1[0])


def eval_with_all_metrics(predictions, prefix, pred_start_idx, target, table, metas, facts_in_target, use_model_based_metrics=False):

    pred_metrics = []

    try:
        target_after_prefix = get_target_after_prefix(prefix, target)
    except Exception:
        target_after_prefix = target

    # Get aligned facts from the target sentence after the prefix (starting from pred_start_idx)
    try:
        target_fact_strs, coords_to_target_facts = parse_facts_in_target(facts_in_target, metas, pred_start_idx)
    except Exception:
        target_fact_strs, coords_to_target_facts = set(), {}

    for prediction in predictions:
        pred_metric = METRIC_TEMPLATE.copy()

        # Get prediction after prefix
        try:
            pred_after_prefix = get_pred_after_prefix(prefix, prediction)
        except Exception:
            pred_after_prefix = prediction

        # Evaluate with different metrics
        try:
            pred_metric['bleu_score'] = clac_bleu(pred_after_prefix, target_after_prefix)
        except Exception:
            pass

        if use_model_based_metrics:
            global BLEURT_SCORER, BERT_SCORER
            if not BLEURT_SCORER:
                BLEURT_SCORER = load_bleurt_scorer()
            if not BERT_SCORER:
                BERT_SCORER = load_bert_scorer()
            try:
                pred_metric['bleurt_score'] = clac_bleurt(BLEURT_SCORER, pred_after_prefix, target_after_prefix)
            except Exception:
                pass
            try:
                pred_metric['bert_score_precision'], pred_metric['bert_score_recall'], pred_metric['bert_score_f_score'] = clac_bert_score(BERT_SCORER, pred_after_prefix, target_after_prefix)
            except Exception:
                pass

        try:
            pred_metric['parent_precision'], pred_metric['parent_recall'], pred_metric['parent_f_score'] = calc_parent(pred_after_prefix, target_after_prefix, table, metas, facts_in_target)
        except Exception:
            pass

        try:
            pred_metric['fact_coverage'] = calc_fact_coverage(pred_after_prefix, table, metas, target_fact_strs, coords_to_target_facts)
        except Exception:
            pass

        try:
            pred_metric['em_1_ratio'] = clac_EM_N(pred_after_prefix, target_after_prefix, 1)
        except Exception:
            pass

        try:
            pred_metric['em_2_ratio'] = clac_EM_N(pred_after_prefix, target_after_prefix, 2)
        except Exception:
            pass

        try:
            pred_metric['em_3_ratio'] = clac_EM_N(pred_after_prefix, target_after_prefix, 3)
        except Exception:
            pass

        try:
            pred_metric['match_rate'] = calc_match_rate(pred_after_prefix, target_after_prefix)
        except Exception:
            pass

        try:
            pred_metric['char_saved'] = calc_char_saved(pred_after_prefix, target_after_prefix)
        except Exception:
            pass

        pred_metrics.append(pred_metric)

    metrics_stats = {
        'max': METRIC_TEMPLATE.copy(),
        'avg': METRIC_TEMPLATE.copy()
    }

    for metric_type in METRIC_TEMPLATE.keys():
        predic_metric_values = [pred_metric[metric_type] for pred_metric in pred_metrics]
        metrics_stats['max'][metric_type] = max(predic_metric_values)
        metrics_stats['avg'][metric_type] = sum(predic_metric_values) / len(predic_metric_values)

    return pred_metrics, metrics_stats


def test_metrics():
    test_data = {}
    for data in load_jsonl_data("./data/dataset/twt/totto_dev.jsonl", False, False):
        test_data = data
        break
    table_data = load_json_data(f"./data/dataset/twt/tables/{test_data['table_id']}.json")
    prefix, start_idx = test_data['prefixes'][0], test_data['start_indices'][0]
    target = test_data['output_sentence']
    predictions = [
        "daniel henry chamberlain was the 76th governor of south carolina from 1874 .",
        "daniel henry chamberlain was the 76th governor of south carolina from 1874 to 1878."
    ]
    pred_metrics, metrics_stats = eval_with_all_metrics(predictions, prefix, start_idx, target, table_data['data'], get_metas_list(table_data['meta']), test_data['matched_facts'])
    print(pred_metrics)
    print(metrics_stats)


if __name__ == "__main__":
    test_metrics()
