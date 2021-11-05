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
        self.vocab = vocab
        self.space_marker = space_marker
        self.device = device
        sorted_vocab = sorted(enumerate(vocab), key=lambda x: x[1])
        sorted_index, values = list(zip(*sorted_vocab))
        self.sorted_index = torch.LongTensor(sorted_index).to(self.device)
        ranks = [x[0] for x in sorted(enumerate(sorted_index), key=lambda x: x[1])]
        self.ranks = np.asarray(ranks)
        self.sorted_vocab = values

        self.FULL_NEGATIVE_MASK = torch.zeros((len(vocab),), dtype=Constants.TORCH_MASK_DTYPE, device=self.device)
        if space_marker:
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
        # !!! To get the vocabs from T5 and BERT, we need to use get_vocab()
        tokens = tokenizer.get_vocab()
        token_strs = list(tokens.keys())
        self.id2token = [token_strs[i] for i in range(len(tokens))]
        # !!! get_vocab() already contains added_tokens
        # if hasattr(tokenizer, 'added_tokens_decoder') and tokenizer.added_tokens_decoder:
        #     self.id2token += [w for _, w in sorted(tokenizer.added_tokens_decoder.items())]
        
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

        print(f"SystemLog: cpus: {cpus}.")

        # with Pool(cpus) as process:
        #     batch_size = 1000
        #     all_words_iter = (all_words[i:i + batch_size] for i in range(0, len(all_words), batch_size))
        #     vocab_tokens = process.map(self._tokenize_batch, all_words_iter, chunksize=10)
        # vocab_tokens = np.asarray([self.accum_lens(tuple(x)) for batch in vocab_tokens for x in batch])

        batch_size = 1000
        all_words_iter = (all_words[i:i + batch_size] for i in range(0, len(all_words), batch_size))
        vocab_tokens = map(self._tokenize_batch, all_words_iter)
        vocab_tokens = np.asarray([self.accum_lens(tuple(x)) for batch in vocab_tokens for x in batch])

        # remove words that contain one token only, as they are already in self.token_trie
        multi_token_word_idx = [i for i, ts in enumerate(vocab_tokens) if len(ts[0]) > 1]
        vocab_tokens = vocab_tokens[multi_token_word_idx]        
        sorted_vocab = [all_words[i] for i in multi_token_word_idx]
       
        self.vocab_tokens = vocab_tokens
        self.sorted_vocab = sorted_vocab 
        

        cache = {'vocab_tokens': vocab_tokens,
                 'sorted_vocab': sorted_vocab}
        torch.save(cache, self.vocab_cache_path)


    def _tokenize_batch(self, words):
        # faster than looping over all words directly
        # !!! T5 starts with "▁", so we need to start from 2
        tokens = self.tokenizer.tokenize('. ' + ' '.join(words))[2:]
        # !!! We need to loop over all words since T5 treats tokens in different sentences differently ?
        # tokens = []
        # for word in words:
        #     tokens.extend(self.tokenizer.tokenize(word))
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
            s, e = get_matching_range(self.sorted_vocab, prefix)
        else:
            s, e = -1, -1

        if s >= e:  # prefix has only one character or not found, test if prefix in tokenizer vocab, else tokenizer the prefix.
            if token_mask is not None:
                # !!! Get space marker token id
                space_marker_token_id = self.tokenizer.convert_tokens_to_ids(Constants.SPACE_MARKER)
                # !!! We need to patch the mask for single char cases, since T5 tokenizes 'a' as ['▁', 'a'] not ['▁a']
                prefix_tokens = self.tokenizer.tokenize(prefix)
                prefix_token_ids = self.tokenizer.convert_tokens_to_ids(prefix_tokens)
                if space_marker_token_id in prefix_token_ids:
                    masks[0].index_fill_(0, torch.tensor([space_marker_token_id]), 1)
                return token_ids, masks
            # !!! T5 starts with "▁", so we need to start from 2
            tokens = self.tokenizer.tokenize(f'. {prefix}')[2:]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens[:-1])
            mask = self.get_mask(tokens[-1])
            return [token_ids], [mask]

        # get up to MAX_EXMAPLE_WORDS words.
        if Constants.MAX_EXAMPLE_WORDS is not None and e - s > Constants.MAX_EXAMPLE_WORDS:
            step = (e - s) // Constants.MAX_EXAMPLE_WORDS
        else:
            step = None
        word_info = self.vocab_tokens[s:e:step]
        prefix_groups = Counter(self.get_prefix_parts(prefix, *x) for x in word_info)

        if len(prefix_groups) > max_prefix_groups:
            prefix_groups = sorted(prefix_groups.items(), key=lambda x: -x[1])[:max_prefix_groups]
        else:
            prefix_groups = prefix_groups.items()

        for k, _ in prefix_groups:
            if len(k) == 0:
                ids, ms = self._get_prefix_masks(prefix, skip_vocab_matching=True, space_marker=self.space_marker)
                token_ids += ids
                masks += ms
            else:
                token_ids.append(self.tokenizer.convert_tokens_to_ids(k))
                remnant_start = sum(len(t) for t in k) - len(space_marker)
                remnant = prefix[remnant_start:]
                _, ms = self._get_prefix_masks(remnant, skip_vocab_matching=True, space_marker='')
                masks.append(ms[0])

        return token_ids, masks

    def get_lookups(self, prefix_batch):
        # returns an array of lookups (one per prefix)
        lookups = []
        for prefix in prefix_batch:
            ids, masks = self._get_prefix_masks(prefix)
            lookup = tree()
            for m, ids_i in zip(masks, ids):
                node = lookup
                for id in ids_i:
                    node = node[id]
                node[None] = m
            lookups.append(lookup)
        return lookups
    
    @staticmethod
    def get_prefix_parts(prefix, tokens, cum_lengths):
        i = bisect.bisect_right(cum_lengths, len(prefix))
        return tokens[:i]

    @staticmethod
    def accum_lens(tokens):
        lengths = np.cumsum(np.asarray([len(t) for t in tokens])).tolist()
        return tokens, lengths
