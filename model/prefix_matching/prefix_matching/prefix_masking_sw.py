import copy

import torch
import numpy as np
import bisect
import time

from .mask_lookup_generator import MaskLookupGenerator
from .constants import Constants 


def get_matching_range(sorted_list, prefix):
    '''Get a range of strings in `sorted_list` that start with `prefix`
    :return: a tuple of start(inclusive), end(exclusive).
    '''
    if not prefix:
        return 0, len(sorted_list)
    l = bisect.bisect_left(sorted_list, prefix)
    end_str = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    r = bisect.bisect_left(sorted_list, end_str, lo=l)
    return l, r

class PrefixMaskingStartsWithFactory():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        # we pre-generate some things that never change and pass them directly to
        # PrefixMasking to avoid generating them everytime
        self.one_mask =  torch.ones((len(self.tokenizer),), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)

        id2token = [tokenizer.decoder[i] for i in range(len(tokenizer.decoder))]
        if hasattr(tokenizer, 'added_tokens_decoder') and tokenizer.added_tokens_decoder:
            id2token += [w for _, w in sorted(tokenizer.added_tokens_decoder.items())]

        sorted_vocab = sorted(enumerate(id2token), key=lambda x: x[1])
        self.sorted_index, values = list(zip(*sorted_vocab))
        self.sorted_vocab = values
    
    def __call__(self, prefixes):
        return PrefixMaskingStartsWith(self.tokenizer, self.device, self.sorted_vocab, self.sorted_index, self.one_mask, prefixes)


class PrefixMaskingStartsWith():
    def __init__(self, tokenizer, device, sorted_vocab, sorted_index, one_mask, prefixes):
        self.tokenizer = tokenizer
        self.device = device
        self.one_mask =  one_mask       
        self.prefixes = [{-1: p} for p in prefixes]
        self.sorted_vocab = sorted_vocab
        self.sorted_index = sorted_index
    
    def get_mask(self, prefix_lookup, next_token_id, add_space_marker):
        '''
            This is a non-exhaustive masking function. Which means not all possible correct tokenizations of a prefix will be returned.
            This is because we found empirically that being exhaustive does not necessarily means better accuracy (and almost always means higher latency).
            Ultimately, we need to strike a balance between allowing the model to choose between all possible tokenization, and constraining the model to "likely" tokenizations.
            With this approach we are strongly contraining the model, leaving a lot of paths out. From our empirical tests this yields much better results with the model we used.

            A very strong model might benefit less from being constrained and this approach might be harmful, but for weaker model, constraining seem to be very beneficial.

            This algorithm has 2 major parts:
            * Lookup phase: we simply look for any token starting with our prefix. E.g. if the prefix is "Cana", we would get "Canada", "Canadians", "Canal" etc.
              We then add all those tokens to our mask.
            * Tokenization phase: If the lookup phase return nothing, then we simply tokenize the prefix, and add the first token to our mask. We save the remnant so that we can repeat the process
              for the next generation step.
              Again, we could always go through this phase, as the token yielded by this process is always a possible good candidate, but we found empirically that constraining the model even more by 
              only adding it when the lookup phase returned nothing was beneficial.
        '''
        token_id = next_token_id.item()
        if token_id not in prefix_lookup:
            return self.one_mask, {}
        
        prefix = prefix_lookup[token_id]

        if add_space_marker:
            lookup_key = Constants.SPACE_MARKER + prefix
        else:
             lookup_key = prefix
        # Lookup phase
        l, r = get_matching_range(self.sorted_vocab, lookup_key)
        valid_tokens = [self.sorted_index[t] for t in range(l, r)]
        
        # Tokenization phase (only if lookup phase yielded nothing)
        remnant_dict = {}     
        if len(valid_tokens) == 0:
            if add_space_marker:
                # reformat the prefix to force GPT2Tokenizer to generate a sequence
                # starting with Ġ to mark the beginning of a word (this is done for the first iteration of Prefix Matching)
                # Throw away the tokenization of the period
                tokenization = self.tokenizer.encode(". {}".format(prefix))[1:]
            else:
                # For any iteration after the first one we don't want to add the space marker anymore since we are now in the
                # middle of a word
                tokenization = self.tokenizer.encode(prefix)

            '''
            if our current tokenization is a single token long, we cannot infer anything
            valuable from it. All we know is that this token is one possible way of tokenizing the prefix.
            and we already found that token during the lookup phase, so no need to do anything here.

            if, however, the tokenization yielded more than one token, then we want to keep the first generated token
            and add it to our mask.
            This token would be shorter than our prefix, and thus, not captured during the lookup phase, hence why we need to capture
            it here.
            Also, in that case, we will need to go through multiple generation step with prefix matching, to fully match the prefix, so we 
            save the remant for the next step
            -------------------
            For example, if the prefix we get from a user is " Libert", the user might be trying to type:
            * Liberty, tokenized as "ĠLiberty"
            * Libertarian -> "ĠLibertarian"
            * Libertad    -> "ĠLiber", "t", "ad"
            (or many other things, but let's just consider those 3 for simplicity)

            From this, we know that "ĠLiber" is the a possible token that we need to consider, we call this the "stable token"
            Both ĠLiberty and ĠLibertarian will be recovered during the lookup phase
            '''
            stable_token = None
            if len(tokenization) > 1:
                stable_token = tokenization[0]
                # Save remnant for next step, if stable_token is chosen during the current iteration.
                # If any other token (from the ones found during lookup phase) is chosen, then our 
                # prefix will be fully matched so their will be no remnant
                remnant = self.tokenizer.decode(tokenization[1:])
                remnant_dict[stable_token] = remnant
                valid_tokens.append(stable_token)


        # faster than doing torch.LongTensor().to()
        if self.device == 'cuda':
            masks_idx = torch.cuda.LongTensor(valid_tokens)
        else:    
            masks_idx = torch.LongTensor(valid_tokens)

        masks = torch.zeros((len(self.tokenizer),), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)
        masks.index_fill_(0, masks_idx, 1)

        return masks, remnant_dict

    def __call__(self, input_seq_index=None, next_token_ids=None):       
        # At the first step we have no next_token_id
        if input_seq_index is None:
            add_space_marker = True
            # Do this so we can iterate on them and avoif if/else conditions 
            input_seq_index = list(range(len(self.prefixes)))
            # See: https://github.com/pytorch/pytorch/issues/24807
            # pylint: disable=E1102
            next_token_ids = [torch.tensor(-1)] * len(self.prefixes)
        else:
            add_space_marker = False

        outputs = [self.get_mask(self.prefixes[i], n_t, add_space_marker) for i, n_t in zip(input_seq_index, next_token_ids)]
        masks, self.prefixes = list(zip(*outputs))
        return torch.stack(masks)
