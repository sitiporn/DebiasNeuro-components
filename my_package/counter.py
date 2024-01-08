from typing import List, Union
from nltk.sentiment.util import NEGATION_RE
from counter_utils.tokenizer import vanilla_tokenize
import sys, os
sys.path.append(os.path.join('..', 'utils'))

def count_negations(
    sent: Union[str, List[str]]
) -> int:
    sent = sent if isinstance(sent, list) else vanilla_tokenize(sent)
    negations = list(filter(
        lambda x: x is not None,
        [NEGATION_RE.search(x) for x in sent]
    ))
    
    return len(negations)
