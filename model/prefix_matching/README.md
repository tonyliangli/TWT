# Prefix Matching

A library to handle partial words completion for NLG scenarios.

Given some context finishing with an incomplete word, this library generates a mask over all 
the tokens that can possibly match this prefix in order to constrain the generation step to only relevant tokens.

### Example Scenario
Given the following context: `I live in Cana`.  
`cana` is the prefix of many different words (`canada`, `canadian`, `canal`, `canamadeupword` [as users might not always write words part of pre-defined vocabulary] etc.), but in all those cases the prefix `cana` has a limited number of different ways it can be tokenized into.  
`prefix_matching` will look into a pre-defined vocabulary for words starting with this prefix, tokenize them, and infer what are the different possible way to tokenize it.    
In this case it will find that there are two different ways:

1. `can` + `a*` + ...
1. `canal` + ...

Accordingly, `prefix_matching` will generate a mask for every generation step coherent with the inferred tokenizations.  
In this case:  
* 1st generation step:
    * `prefix_matching` will generate a mask with a positive value only for the logits representing `can` and `canal`.
* 2nd generation step:
    * If `can` was chosen, only tokens starting with `a` will be allowed.
    * If `canal` was chosen, then all tokens are allowed as we have already matched our prefix in it's entirety.
* subsequent generation steps: The mask is always positive everywhere since in both cases we have match our prefix entirely.

## Installation

If used within `agipt`, this library can be imported as you would expect (e.g. `from .inference.prefix_matching import *`).  
If you want to use it outside in an external project, this library contains a dedicated `setup.py`, so you can simply do `pip install .` and then `import prefix_matching`.

## How to use it

```python
from .inference.prefix_matching import PrefixMaskingFuncFactory
# OR 
from prefix_matching import PrefixMaskingFuncFactory

# The tokenizer must have the same interface as HuggingFace's tokenizer.
# In AgiPT, that would be `tokenizer.text_tokenizer`
prefix_masking_factory = PrefixMaskingFuncFactory(tokenizer, beam_size)

# prefix batch contains the prefixes of all our contexts in the batch
# e.g., if our batch is ["Hi how are yo", "Please schedu"]
# then prefix_batch  = ["yo", "schedu"]
prefix_masking_func = prefix_masking_factory(prefix_batch)

# returned tensor will be of dimension [batch_size, vocabulary_size]
prefix_mask = prefix_masking_func(beam.select_indice, tokens_beam)

# apply the mask on our logits
logits += -1e4 * (1 - prefix_mask.cuda())

```