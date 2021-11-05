from .prefix_masking_vocab import PrefixMaskingFuncVocabFactory
from .prefix_masking_sw import PrefixMaskingStartsWithFactory
from .constants import Constants

def PrefixMaskingFuncFactory(tokenizer, beam_size, pm_type, prefix_matching_cache_path=None, vocab_file=None, device='cpu', beam_implementation='megatron', **kwargs):
    if pm_type == "vocab":
        return PrefixMaskingFuncVocabFactory(tokenizer, beam_size, prefix_matching_cache_path, vocab_file, device, beam_implementation, **kwargs)
    elif pm_type == 'startswith':
        return PrefixMaskingStartsWithFactory(tokenizer, device)
