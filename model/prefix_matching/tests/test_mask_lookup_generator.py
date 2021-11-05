import pdb
import time
import pytest

from transformers import GPT2Tokenizer

from ..prefix_matching.mask_lookup_generator import get_matching_range

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    return tokenizer

def test_get_matching_range():
    vocab = ['ted', 'teen', 'teenth', 'tein', 'tek', 'tel', 'tele', 'tell', 'telling', 'tem',
            'temp', 'template', 'ten', 'tenance', 'teness', 'ter', 'tera', 'terday', 'tered',
            'tering', 'terior', 'term', 'termin', 'termination', 'terms', 'tern', 'ternal']

    left, right = get_matching_range(vocab, 'tel')
    # Left should be inclusive, and right exclusive
    assert (left, right) == (5, 9)

    # Case where nothing matches
    left, right = get_matching_range(vocab, 'nomatch')
    assert (left, right) == (0, 0)

    # Case where everything match
    left, right = get_matching_range(vocab, 't')
    assert (left, right) == (0, len(vocab))



