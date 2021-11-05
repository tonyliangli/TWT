import pytest
import pdb

from transformers import GPT2Tokenizer

from .. import prefix_matching 

PM_TYPE='vocab'

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    return tokenizer

def test_simple_prefix():
    # 'cana' has 2 possible tokenization:
    # either ' can' (460), followed by any token starting with a (e.g. 'ada' (4763))
    # or ' canal' (29365) directly. 
    prefix_batch = ['cana']
    beam_size = 1
    tokenizer = get_tokenizer()
    masking_factory = prefix_matching.PrefixMaskingFuncFactory(tokenizer, beam_size, PM_TYPE)

    id2token = masking_factory._mask_generator.id2token
    tokens_starting_with_a = list(filter(lambda x: x[1].startswith('a'), enumerate(id2token)))
    tokens_starting_with_a = [x[0] for x in tokens_starting_with_a]

    # The masking func modifies it's internal state on every step
    # so we need multiple copies to test different paths
    masking_func1 = masking_factory(prefix_batch)
    masking_func2 = masking_factory(prefix_batch)
    
    # Check than the only possible tokens at the first step are ' can' and ' canal'
    mask1_1 = masking_func1().nonzero()[:, 1].tolist()
    assert mask1_1 == [460, 29365]

    # If we chose ' can' (460), then on the next step we should have a mask
    # over all words staring with 'a'
    mask1_2 = masking_func1([0], [460]).nonzero()[:, 1].tolist()
    assert mask1_2 == tokens_starting_with_a
    # After that, there is nothing more we can infer so subsequent step should
    # be a positive mask over all tokens. We pass a random token starting with a
    mask1_3 = masking_func1([0], [64])
    assert mask1_3.sum() == len(tokenizer)

    # If we chose ' canal' (29365), then we cannot infer anything more from our prefix
    # so any subsequent step should return a positive mask over all tokens
    masking_func2()
    mask2_2 = masking_func2([0], [29365])
    assert mask2_2.sum() == len(tokenizer)

def test_simple_prefix_beam_size_2():
    prefix_batch = ['cana']
    beam_size = 2
    tokenizer = get_tokenizer()
    masking_factory = prefix_matching.PrefixMaskingFuncFactory(tokenizer, beam_size, PM_TYPE)

    id2token = masking_factory._mask_generator.id2token
    tokens_starting_with_a = list(filter(lambda x: x[1].startswith('a'), enumerate(id2token)))
    tokens_starting_with_a = [x[0] for x in tokens_starting_with_a]

    masking_func = masking_factory(prefix_batch)    
    
    # pdb.set_trace()
    mask1 = masking_func().nonzero()[:, 1].tolist()
    assert mask1 == [460, 29365]

    mask2 = masking_func([0, 1], [460, 29365])
    assert mask2[0].nonzero()[:, 0].tolist() == tokens_starting_with_a
    assert mask2[1].sum() == len(tokenizer)

def test_with_batch():
    prefix_batch = ['cana'] * 4
    beam_size = 1
    tokenizer = get_tokenizer()
    masking_factory = prefix_matching.PrefixMaskingFuncFactory(tokenizer, beam_size, PM_TYPE)

    id2token = masking_factory._mask_generator.id2token
    tokens_starting_with_a = list(filter(lambda x: x[1].startswith('a'), enumerate(id2token)))
    tokens_starting_with_a = [x[0] for x in tokens_starting_with_a]

    masking_func = masking_factory(prefix_batch)    

    batch_mask1 = masking_func()
    assert batch_mask1.size(0) == len(prefix_batch)
    for mask in batch_mask1:
        assert mask.nonzero()[:, 0].tolist() == [460, 29365]

    batch_mask2 = masking_func([0] * 4, [460] * 4)
    assert batch_mask2.size(0) == len(prefix_batch)
    for mask in batch_mask2:
        assert mask.nonzero()[:, 0].tolist() == tokens_starting_with_a





