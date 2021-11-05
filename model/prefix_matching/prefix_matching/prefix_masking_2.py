import copy

import torch
import numpy as np

from .mask_lookup_generator import MaskLookupGenerator
#import .constants as constants
from .constants import Constants 


class PrefixMaskingFuncFactory():
    # Factory for masking function over a batch of prefixes
    def __init__(self, tokenizer, beam_size, prefix_matching_cache_path=None, vocab_file=None, space_marker=Constants.SPACE_MARKER, device='cpu', beam_implementation='megatron'):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.device = device
        self.beam_implementation = beam_implementation
        self._mask_generator = MaskLookupGenerator(tokenizer, prefix_matching_cache_path, vocab_file, space_marker, device=device)
    
    def __call__(self, prefix_batch):
        # Returns a masking func
        lookups = self._mask_generator.get_lookups(prefix_batch)
        masking_func = PrefixMasking(lookups, len(self.tokenizer), self.beam_size, device=self.device, beam_implementation=self.beam_implementation)
        return masking_func


class PrefixMasking():
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
            # If this is not the first generation step, then we move the root of the lookup tree to match
            # whatever we have generated at the previous step
            # For example, for 'Gestrang', if the model generate 'Gest' on the previous step, then this becomes our new root
            #     root  <--- root at step 1
            #      / \
            #     /   \
            #   None  Gest <--- root at step 2
            #          /\
            #         /  \
            #       None  r
            #              \
            #              None     
            # we do this independently for every item in the batch/beam   
            self.lookups = [self._get_lookup(sid, tid) for sid, tid in zip(input_seq_index, next_token_ids)]

        masks = torch.ones((len(self.lookups), self.tokenizer_vocab_len), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)
        for i, lookup in enumerate(self.lookups):
            if not lookup:
                continue
            
            # Here we build the final mask that will be used for the logits.
            # this mask is built by merging the value of the None child node, with the keys of all other nodes.
            # For example, for 'Gestrang' we have:
            #     root 
            #      / \
            #     /   \
            #   None  Gest
            #          /\
            #         /  \
            #       None  r
            #              \
            #              None     
            # So at the first generation step, the allowed tokens are any tokens that directly contains 'Gestrang' (this is what the mask at None contains)
            # + the token "Gest" 
            # So we get the mask from None, and modify it to also have Gest be positive.
            # 
            # If the model generated Gest at the first step, then at the second step Gest is our new root, and allowed values
            # are the mask at None under Gest that contains all tokens starting with 'rang', modified to also include the token 'r'
            prefix_heads = [k for k in lookup.keys() if k is not None]  # token ids match the prefix.
            mask = lookup.get(None)
            if prefix_heads:
                prefix_heads = torch.LongTensor(prefix_heads).to(self.device)
            if mask is None:
                if len(prefix_heads):
                    masks[i].fill_(0).index_fill_(0, prefix_heads, 1)
            else:
                if len(prefix_heads):
                    # Here we modify our mask from the None node to also include keys of sibling nodes
                    mask.index_fill_(0, prefix_heads, 1)
                masks[i] = mask
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