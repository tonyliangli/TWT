# coding=utf-8
import torch

class Constants:
    # !!! We need to change the space marker for T5
    SPACE_MARKER = '▁'
    # !!! Change to int64 for to add to logits
    MASK_DTYPE = 'int64'
    TORCH_MASK_DTYPE = getattr(torch, MASK_DTYPE)
    MAX_EXAMPLE_WORDS = 100