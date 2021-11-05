import copy

import torch
import numpy as np

from .mask_lookup_generator import MaskLookupGenerator
from .constants import Constants 


class PrefixMaskingFuncVocabFactory():
    # Factory for masking function over a batch of prefixes
    def __init__(self, tokenizer, beam_size, prefix_matching_cache_path=None, vocab_file=None, device='cpu', beam_implementation='megatron'):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.device = device
        self.beam_implementation = beam_implementation
        self.use_vocab = False

        self._mask_generator = MaskLookupGenerator(tokenizer, prefix_matching_cache_path, vocab_file, device=device)
        self.use_vocab = True
        
    def __call__(self, prefix_batch, **kwargs):
        lookups = self._mask_generator.get_lookups(prefix_batch)
        masking_func = PrefixMaskingVocab(lookups, len(self.tokenizer), self.beam_size, device=self.device, beam_implementation=self.beam_implementation)
        # !!! Add additional properties
        masking_func.tokenizer = self.tokenizer
        masking_func.prefix_batch = prefix_batch
        masking_func.force_prefix = True if 'force_prefix' in kwargs and kwargs['force_prefix'] else False
        masking_func.remove_space_marker = True if 'remove_space_marker' in kwargs and kwargs['remove_space_marker'] else False
        return masking_func

class PrefixMaskingVocab():
    def __init__(self, lookups, tokenizer_vocab_len, beam_size, device, beam_implementation):
        self.lookups = lookups
        self.tokenizer_vocab_len = tokenizer_vocab_len
        self.beam_size = beam_size
        self.first_step = True
        self.device = device
        self.beam_implementation = beam_implementation

    def _get_lookup(self, sequence_id, token_id):
        lookup = self.lookups[self._get_valid_sequence_id(sequence_id)]
        return lookup.get(token_id, {})

    def _get_masks(self, input_seq_index, next_token_ids):
        if not any(self.lookups):
            return None

        if not self.first_step:
            if torch.is_tensor(input_seq_index):
                input_seq_index = input_seq_index.cpu().numpy()
            if torch.is_tensor(next_token_ids):
                next_token_ids = next_token_ids.flatten().cpu().numpy()
            
            self.lookups = [self._get_lookup(sid, tid) for sid, tid in zip(input_seq_index, next_token_ids)]

        masks = torch.ones((len(self.lookups), self.tokenizer_vocab_len), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)
        for i, lookup in enumerate(self.lookups):
            if not lookup:
                continue
            prefix_heads = [k for k in lookup.keys() if k is not None]  # token ids match the prefix.
            mask = lookup.get(None)
            if prefix_heads:
                prefix_heads = torch.LongTensor(prefix_heads).to(self.device)
            if mask is None:
                if len(prefix_heads):
                    masks[i].fill_(0).index_fill_(0, prefix_heads, 1)
            else:
                if self.first_step:
                    # !!! Get space marke
                    space_marker = Constants.SPACE_MARKER
                    # !!! Get space marker token id
                    space_marker_token_id = self.tokenizer.convert_tokens_to_ids(Constants.SPACE_MARKER)
                    # !!! Add force prefix
                    if self.force_prefix:
                        prefix = self.prefix_batch[i].strip()
                        _prefix = space_marker + prefix
                        _prefix_token_id = self.tokenizer.convert_tokens_to_ids(_prefix)
                        if _prefix_token_id != self.tokenizer.unk_token_id:
                            mask.fill_(0).index_fill_(0, torch.tensor([_prefix_token_id]), 1)
                    # !!! Add remove space marker
                    if self.remove_space_marker:
                        mask.index_fill_(0, torch.tensor([space_marker_token_id]), 0)

                if len(prefix_heads):
                    # !!! Only fill for the following conditions
                    if not self.first_step or (not self.force_prefix and not self.remove_space_marker):
                        mask.index_fill_(0, prefix_heads, 1)
                masks[i] = mask
        # import pdb; pdb.set_trace()
        return masks
    
    def _get_valid_sequence_id(self, sid):
        # Megatron and HF don't assign beam sequence ids the same way, so this an ugly fix
        # William Buchwalter TODO: find a cleaner way to handle this
        if self.beam_implementation=='megatron':
            return sid // self.beam_size
        return sid

        
    def __call__(self, input_seq_index=None, next_token_ids=None):
        '''
        :param input_seq_index, next_token_ids: [LongTensor(shape=(d,))], where d=batch_size x beam_size(num_seq per sample), i.e., total number of sequences.
            `input_seq_index` = the index of input sequences where the `next_token_ids` is associated with.
            `next_token_ids` = the selected next token ids.
            Note: the two inputs should have the same shape and order as `next_token_ids`. All index/ids starts with 0.

            Example: with batch size=2, beam_size=2, the input_seq_index should have length=2x2=4.
                Let's assumed input_seq_index = [1, 0, 3, 3], next_token_ids = [300, 800, 90, 50]. It means
                - for sample 1: seq-0, seq-1 are selected next candidate, with 300 (for seq-0), 800 (for seq-1) as the best next token_ids respectively.
                - for sample 2: seq-3 is chosen (seq-2 is dropped), with 90 and 50 as best next token ids.
        '''
        masks = self._get_masks(input_seq_index, next_token_ids)

        self.first_step = False
        return masks