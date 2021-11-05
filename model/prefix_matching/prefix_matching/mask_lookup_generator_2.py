import os
import logging
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
import stat
import tempfile
import hashlib
import bisect
import re

import torch
# import pycedar 
import numpy as np
import nltk
from nltk.corpus import words

from .constants import Constants

def tree():
    return defaultdict(tree)

def get_cache_path(associate_path=None, version=None):
    if associate_path is not None:
        realpath = os.path.realpath(associate_path)
        st = os.stat(realpath)  
        uid = f'{realpath}:{st[stat.ST_MTIME]}'
    else:
        uid = f'None:{version}'
    tempdir = tempfile.gettempdir()
    uid = hashlib.sha256(uid.encode()).hexdigest()
    return os.path.join(tempdir, 'partial-' + uid)

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

class MaskBuilder(object):
    def __init__(self, vocab, space_marker=Constants.SPACE_MARKER, device='cpu'):
        # vocab here is the tokenizer vocab, decoded
        self.vocab = vocab
        self.space_marker = space_marker
        self.device = device
        sorted_vocab = sorted(enumerate(vocab), key=lambda x: x[1])
        # we enumerated our vocab and sorted it alphabetically
        # and split id and value
        sorted_index, values = list(zip(*sorted_vocab))
        self.sorted_index = torch.LongTensor(sorted_index).to(self.device)
        
        # ranks = [x[0] for x in sorted(enumerate(sorted_index), key=lambda x: x[1])]
        # self.ranks = np.asarray(ranks)
        self.sorted_vocab = values

        self.FULL_NEGATIVE_MASK = torch.zeros((len(vocab),), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)
        if space_marker:
            # Find all the tokens that starts with a space (^G) and positively mask them 
            self.START_WITH_SPACE_MASK = torch.from_numpy(np.asarray(
                    [x.startswith(self.space_marker) for x in vocab],  dtype=Constants.MASK_DTYPE)
                ).to(self.device)
        else:
            self.START_WITH_SPACE_MASK = torch.ones(len(vocab), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)

    def __call__(self, prefix):
        '''
        return: None if not found. Otherwise a torch.Tensor(shape=(vocab_size,)), with 1=match, 0=mis-match
        '''
        if not prefix:
            # This case should never happen
            return self.FULL_NEGATIVE_MASK
        if prefix == Constants.SPACE_MARKER:
            return self.START_WITH_SPACE_MASK

        l, r = get_matching_range(self.sorted_vocab, prefix)
        if l >= r:
            return None
        mask = torch.zeros((len(self.vocab),), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)
        mask.index_fill_(0, self.sorted_index[l:r], 1)
        return mask

class MaskLookupGenerator():
    def __init__(self, tokenizer, prefix_matching_cache_path=None, vocab_file=None, space_marker=Constants.SPACE_MARKER, load_from_cache=True, device='cpu'):
        self.tokenizer = tokenizer
        self.space_marker = space_marker

        self.sorted_vocab = None
        self.vocab_tokens = None

        # id2token maps BPE tokens id to their decoded string
        self.id2token = [tokenizer.decoder[i] for i in range(len(tokenizer.decoder))]
        if hasattr(tokenizer, 'added_tokens_decoder') and tokenizer.added_tokens_decoder:
            self.id2token += [w for _, w in sorted(tokenizer.added_tokens_decoder.items())]
        
        self.full_mask = torch.ones(len(tokenizer), dtype=Constants.TORCH_MASK_DTYPE, device=device)
        self.vocab_cache_path = prefix_matching_cache_path if prefix_matching_cache_path is not None else get_cache_path(vocab_file, nltk.__version__)
        self._load_vocab(vocab_file, load_from_cache)        

        self.get_mask = MaskBuilder(self.id2token, space_marker=space_marker, device=device)
   

    def _load_vocab(self, vocab_file, load_from_cache):
        if load_from_cache and os.path.exists(self.vocab_cache_path):
            try:
                self._load_vocab_from_cache()
                return
            except BaseException as e:
                logging.warning(f'Error at reading cache file at {self.vocab_cache_path}')
                logging.warning(str(e))

        self._build_vocab(vocab_file)


    def _load_vocab_from_cache(self):
        logging.info(f'Reading cached pre-defined vocabulary from {self.vocab_cache_path}.')
        cache = torch.load(self.vocab_cache_path)
        for k, v in cache.items():
            setattr(self, k, v)

    def _build_vocab(self, vocab_file=None):
        logging.info('Tokenizing pre-defined vocabulary...')
        if vocab_file:
            all_words = []
            with open(vocab_file) as f:
                for l in f:
                    kv = l.strip().split('\t')
                    if not kv:
                        continue
                    elif len(kv) == 1:
                        k, v = kv, 1
                    else:
                        k, v = kv[:2]
                    if re.match(r"[\w']+$", k):
                        all_words.append(k)
            all_words = sorted(enumerate(all_words), key=lambda x: x[1])
            idx, all_words = list(zip(*all_words))       
        else:
            logging.info('Use Nltk.corpus.words as vocab.')
            all_words = sorted(words.words())   
        
        cpus = max(1, cpu_count() - 1)

        with Pool(cpus) as process:
            batch_size = 1000
            all_words_iter = (all_words[i:i + batch_size] for i in range(0, len(all_words), batch_size))
            vocab_tokens = process.map(self._tokenize_batch, all_words_iter, chunksize=10)
        vocab_tokens = np.asarray([self.accum_lens(tuple(x)) for batch in vocab_tokens for x in batch])

        # remove words that contain one token only, as they are already in self.token_trie
        multi_token_word_idx = [i for i, ts in enumerate(vocab_tokens) if len(ts[0]) > 1]
        vocab_tokens = vocab_tokens[multi_token_word_idx]        
        sorted_vocab = [all_words[i] for i in multi_token_word_idx]

        # sorted_vocab contains all the words as strings in the vocab 
        # vocab_tokens contains all the words as token ids in the vocab
        self.sorted_vocab = sorted_vocab 
        self.vocab_tokens = vocab_tokens

        cache = {'vocab_tokens': vocab_tokens,
                 'sorted_vocab': sorted_vocab}
        torch.save(cache, self.vocab_cache_path)


    def _tokenize_batch(self, words):
        # faster than looping over all words directly
        tokens = self.tokenizer.tokenize('. ' + ' '.join(words))[1:]
        outputs = []
        tmp = []
        for t in tokens:
            if t.startswith(self.space_marker):
                if tmp:
                    outputs.append(tmp)
                    tmp = []
            tmp.append(t)
        if tmp:
            outputs.append(tmp)
        return outputs


    def _get_prefix_masks(self, prefix, max_prefix_groups=10, skip_vocab_matching=False, space_marker=None):
        prefix = prefix.strip()
        space_marker = self.space_marker if space_marker is None else space_marker
        _prefix = space_marker + prefix
        token_mask = self.get_mask(_prefix)

        if not prefix:
            return [[]], [token_mask]
        
        # outputs
        token_ids = []  # next tokens, up to the last piece
        masks = [] # mask of the last token piece

        # check if there are tokens in the tokenizer starting with the prefix
        if token_mask is not None:
            token_ids.append([])
            masks.append(token_mask)

        if not skip_vocab_matching and len(prefix) >= 2:
            # Now look into our Wikipedia/NLTK vocab
            s, e = get_matching_range(self.sorted_vocab, prefix)
        else:
            s, e = -1, -1

        if s >= e:  # prefix has only one character or not found, test if prefix is in the tokenizer vocab, otherwise directly tokenize it
            if token_mask is not None:
                return token_ids, masks
            #adding '. ' in front forces the tokenizer to generate the space marker
            tokens = self.tokenizer.tokenize(f'. {prefix}')[1:]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens[:-1])
            # build a mask over vocab tokens starting with the last token of our current prefix
            # for Cana, that would be 'a'
            mask = self.get_mask(tokens[-1])
            return [token_ids], [mask]

        # get up to MAX_EXAMPLE_WORDS words.
        # If we have more than 100 matches, we don't want to tokenize every single one as it would be too long
        # We only want MAX_EXAMPLE_WORDS.
        # We also don't want to take the first MAX_EXAMPLE_WORDS match, as vocab_tokens is ordered alphabetically, so items that are
        # close together are likely to be tokenized in the same way. So we try to sample uniformly from all matches, to get 
        # as many different ways of tokenizing as possible
        if Constants.MAX_EXAMPLE_WORDS is not None and e - s > Constants.MAX_EXAMPLE_WORDS:
            step = (e - s) // Constants.MAX_EXAMPLE_WORDS
        else:
            step = None
        word_info = self.vocab_tokens[s:e:step]
        # given a prefix and a tokenized matching word (e.g. cana and ['Gcan', 'ada])
        # get_prefix_parts will return the part of the tokenization that does not go over
        # the prefix length. E.g ['Gcan, 'ada'] is longer than the prefix, but 'Gcan' is not
        # so it will return 'Gcan'.
        # ["Gcanal"] is longer than "Gcan", so it will return at empty prefix group
        prefix_groups = Counter(self.get_prefix_parts(prefix, *x) for x in word_info)

        if len(prefix_groups) > max_prefix_groups:
            # If we have too many prefix groups, just keep the x most commons ones
            prefix_groups = sorted(prefix_groups.items(), key=lambda x: -x[1])[:max_prefix_groups]
        else:
            prefix_groups = prefix_groups.items()

        for k, _ in prefix_groups:
            if len(k) == 0:
                # The case were len(k) == 0, an empty prefix group, correponds to all tokenizations where the first token is equal or longer
                # than our entire prefix. E.g. 'canal' is a valid token and englobes our entire prefix.
                # For those cases, we can just look over our tokenizer vocab for any tokens that starts with our prefix
                # and return a mask that is positive for those values.
                # We skip the prefix vocab matching for this case 

                # I am actually unclear why we go through self._get_prefix_masks again, because we already have the result stored
                # in token_ids and masks from l. 198. 
                ids, ms = self._get_prefix_masks(prefix, skip_vocab_matching=True, space_marker=self.space_marker)
                token_ids += ids
                masks += ms
            else:
                # This is the case where a prefix group, needs multiple tokens to cover the entire prefix
                # for example if we have prefix "estrang", valid tokenizations could be (among others):   
                # "estrang" -> ["Ġest", "ranged"], ["Ġest", "r", "anger"]
                # which would give us 2 prefix groups: ["Ġest"] and ["Ġest", "r"]
                # for each prefix group, we find the remaining part of the prefix that is not cover, and we get a mask over
                # all the items in our tokenizer's vocab that match the remnant.
                # here, for the first group we would get a mask over rang* and for the second group over ang*
                token_ids.append(self.tokenizer.convert_tokens_to_ids(k))
                remnant_start = sum(len(t) for t in k) - len(space_marker)
                remnant = prefix[remnant_start:]
                # No point in doing the Wikipedia vocab matching, since we are dealing with only a part of a word now
                # so directly get whatever matches in the tokenizer's vocab instead. No space_marker since we are not at the start of a word
                # anymore, so we don't want the extra "Ġ" character
                _, ms = self._get_prefix_masks(remnant, skip_vocab_matching=True, space_marker='')
                masks.append(ms[0])

        return token_ids, masks

    def get_lookups(self, prefix_batch):
        # returns an array of lookups (one per element in the batch)
        lookups = []
        for prefix in prefix_batch:
            ids, masks = self._get_prefix_masks(prefix)

            # for each possible way of tokenizing, we create one node per step
            # and add the masks to the last step.
            # for 'Gestrang', we have 3 ways of tokenizing, one of them being ['Gest', 'r'] followed by any token staring with 'ang'
            # so in this case we would create a 'Gest' node below the root, an 'r' node below this one, and save the mask under a None 
            # node below this one: 'Gest' -> 'r' -> None = mask
            lookup = tree()
            for m, ids_i in zip(masks, ids):
                node = lookup
                for id in ids_i:
                    node = node[id]
                node[None] = m
            lookups.append(lookup)
        # In the case of "Gestrang" our lookup tree will look like this (with a mask saved as value of each None node)
        #     root
        #      / \
        #     /   \
        #   None  Gest
        #          /\
        #         /  \
        #       None  r
        #              \
        #              None        
        return lookups
    
    @staticmethod
    def get_prefix_parts(prefix, tokens, cum_lengths):
        i = bisect.bisect_right(cum_lengths, len(prefix))
        return tokens[:i]

    @staticmethod
    def accum_lens(tokens):
        lengths = np.cumsum(np.asarray([len(t) for t in tokens])).tolist()
        return tokens, lengths
