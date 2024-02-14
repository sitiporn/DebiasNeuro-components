import re
from typing import List

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


LEMMATIZER = WordNetLemmatizer()


def __filter(_sent: List[str]) -> List[str]:
    _sent = list(filter(lambda x: re.match("\w+", x), _sent))
    return _sent


def vanilla_tokenize(_sent: str, _filter=__filter) -> List[str]:
    _words = [x.lower() for x in word_tokenize(_sent)]
    _words = _filter(_words)
    return _words


def lemmatized_tokenize(_sent: str, _lemmatizer=LEMMATIZER, _filter=__filter) -> List[str]:
    _words = [_lemmatizer.lemmatize(x.lower(), "v")
              for x in word_tokenize(_sent)]
    _words = _filter(_words)
    return _words

from typing import List, Callable



def get_ngram_doc(
    doc: str,
    n: int,
    tokenize: Callable[[str], List[str]] = vanilla_tokenize
)-> List[str]:
    tokenized_doc = tokenize(doc)
    length = len(tokenized_doc)-(n-1)
    return [
        '_'.join([tokenized_doc[i+j] for j in range(n)])
        for i in range(length)
    ]


def get_ngram_docs(
    docs: List[str],
    n: int,    
    tokenize: Callable[[str], List[str]] = vanilla_tokenize
) -> List[List[str]]:
    return [
        get_ngram_doc(doc=doc, n=n, tokenize=tokenize)
        for doc in docs
    ]

from typing import List, Union

from nltk.sentiment.util import NEGATION_RE


def count_negations(
    sent: Union[str, List[str]]
) -> int:
    sent = sent if isinstance(sent, list) else vanilla_tokenize(sent)
    negations = list(filter(
        lambda x: x is not None,
        [NEGATION_RE.search(x) for x in sent]
    ))
    return len(negations)

from typing import Dict, List

from math import log


def lmi(
    p_w_l: float,
    p_l_given_w: float,
    p_l: float
) -> float:
    return p_w_l * log(p_l_given_w/p_l)


def get_ngram_probs(
    ngram_docs: List[List[str]],
    labels: List[str],
    possible_labels: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
        Example of the output:
            {
                "SUPPORTS": {
                    "does_not": {
                        "p_w_l": 0.001,
                        "p_l_given_w": 0.005,
                        "p_l": 0.002
                    },
                    "get_in": {
                        "p_w_l": 0.001,
                        "p_l_given_w": 0.005,
                        "p_l": 0.002
                    }
                },
                "REFUTES": {
                    "did_not": {
                        "p_w_l": 0.001,
                        "p_l_given_w": 0.005,
                        "p_l": 0.002
                    }
                }
            }
    """
    possible_labels = possible_labels if possible_labels else list(set(labels))
    counter = {label: {} for label in possible_labels} # count(w, l)
    n_appear_labels = {label: 0 for label in possible_labels} # count(l)
    n_ngrams = {} # count(w)

    for ngram_doc, label in zip(ngram_docs, labels):
        for ngram in ngram_doc:
            counter[label][ngram] = counter[label].get(ngram, 0) + 1
            n_ngrams[ngram] = n_ngrams.get(ngram, 0) + 1
            n_appear_labels[label] += 1

    total_ngrams = sum([n for _, n in n_appear_labels.items()]) # D
    prob = {label: {} for label in possible_labels}

    for label in possible_labels:
        p_l = n_appear_labels[label] / total_ngrams
        for ngram in counter[label].keys():
            prob[label][ngram] = {
                "p_w_l": counter[label][ngram] / total_ngrams,
                "p_l_given_w": counter[label][ngram] / n_ngrams[ngram],
                "p_l": p_l
            }

    return prob


def compute_lmi(
    ngram_docs: List[List[str]],
    labels: List[str],
    possible_labels: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
        Example of the output:
            {
                "SUPPORTS": {
                    "does_not": 0.2,
                    "get_in": 0.1
                },
                "REFUTES": {
                    "did_not": 0.2
                }
            }
    """
    possible_labels = possible_labels if possible_labels else list(set(labels))
    ngram_probs = get_ngram_probs(
        ngram_docs=ngram_docs,
        labels=labels,
        possible_labels=possible_labels
    )
    for label in possible_labels:
        for ngrams in ngram_probs[label].keys():
            ngram_probs[label][ngrams] = lmi(**ngram_probs[label][ngrams])
    return ngram_probs

from typing import List, Union

import spacy


SPACY_NLP = spacy.load("en_core_web_sm")


def get_lexical_overlap(
    sent1: Union[str, List[str]],
    sent2: Union[str, List[str]]
) -> float:
    sent1 = sent1 if isinstance(sent1, list) else vanilla_tokenize(sent1)
    sent2 = sent2 if isinstance(sent2, list) else vanilla_tokenize(sent2)

    count = 0
    for w1 in sent1:
        for w2 in sent2:
            if w1 == w2:
                count += 1

    return count / max(len(sent1), len(sent2))


def get_entities_overlap(
    sent1: str,
    sent2: str
) -> int:
    doc1 = SPACY_NLP(sent1)
    doc2 = SPACY_NLP(sent2)

    count = 0
    for ent1 in doc1.ents:
        for ent2 in doc2.ents:
            if (ent1.text, ent1.label_) == (ent2.text, ent2.label_):
                count += 1

    return count

def inference_prob_to_index(x: List[Dict[str, float]]) -> List[float]:
    return [
        x["SUPPORTS"],
        x["NOT ENOUGH INFO"],
        x["REFUTES"]
    ]



import os
import operator
import pickle
from typing import Callable, Dict, List, Tuple

from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# from my_package.models.traditional import TraditionalML, FEATURE_EXTRACTOR
# from my_package.utils.handcrafted_features.mutual_information import compute_lmi
# from my_package.utils.ngrams import get_ngram_doc, get_ngram_docs
# from my_package.utils.tokenizer import vanilla_tokenize
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple


FEATURE_EXTRACTOR = Callable[[str, str], float]


class TraditionalML(ABC):
    @abstractmethod
    def fit(self, docs: List[Tuple[str, str]], labels: List[str]) -> None:
        ...

    @abstractmethod
    def inference(self, docs: List[Tuple[str, str]]) -> List[dict]:
        ...

    @abstractmethod
    def predict(self, docs: List[Tuple[str, str]]) -> List[str]:
        ...

DEFAULT_CONFIG = {
    "n_grams": [1, 2],
    "top_ks": [50, 50],
}

DEFAULT_MODEL = LogisticRegression(
    random_state=42,
    solver='saga',
    max_iter=500
)


class Classifier(TraditionalML):
    def __init__(
        self,
        possible_labels: List[str],
        feature_extractors: List[FEATURE_EXTRACTOR],
        tokenizer: Callable[[str], List[str]] = vanilla_tokenize,
        normalizer: TransformerMixin = MinMaxScaler(),
        model: RegressorMixin = DEFAULT_MODEL,
        config: dict = DEFAULT_CONFIG
    ) -> None:
        """
            Currently, tokenizer and normalizer can not be saved to the file.
            We have to inject it mannually.
        """
        self.map_labels = {lb: i for i, lb in enumerate(possible_labels)}
        self.model = model
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.feature_extractors = feature_extractors

        self.config = config
        self._validate_config()
        # internal states
        self.top_ngrams_sent1 = None
        self.top_ngrams_sent2 = None
        self.words_to_idx = None
        self.n_features = None

    def save(self, folder: str) -> None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        _model = {
            "map_labels": self.map_labels,
            "model": self.model,
            "normalizer": self.normalizer,
        }
        _states = {
            "config": self.config,
            "top_ngrams_sent1": self.top_ngrams_sent1,
            "top_ngrams_sent2": self.top_ngrams_sent2,
            "words_to_idx": self.words_to_idx,
            "n_features": self.n_features
        }
        pickle.dump(_model, open(os.path.join(folder, "model.pickle"), 'wb'))
        pickle.dump(_states, open(os.path.join(folder, "state.pickle"), 'wb'))

    def load(self, folder: str) -> None:
        _model = pickle.load(open(os.path.join(folder, "model.pickle"), 'rb'))
        self.map_labels = _model["map_labels"]
        self.model = _model["model"]
        self.normalizer = _model["normalizer"]
        _states = pickle.load(open(os.path.join(folder, "state.pickle"), 'rb'))
        self.config = _states["config"]
        self.top_ngrams_sent1 = _states["top_ngrams_sent1"]
        self.top_ngrams_sent2 = _states["top_ngrams_sent2"]
        self.words_to_idx = _states["words_to_idx"]
        self.n_features = _states["n_features"]

    def _validate_config(self):
        assert len(self.config.get("n_grams")) \
            == len(self.config.get("top_ks"))

    def _get_top_n_grams(self, docs: List[str], labels: List[str]) -> Dict[int, List[str]]:
        top_ngrams = {}
        top_k_lmis = {}
        for n, top_k in zip(self.config.get("n_grams"), self.config.get("top_ks")):
            ngram_docs = get_ngram_docs(
                docs=docs, n=n,
                tokenize=self.tokenizer
            )
            lmis = compute_lmi(ngram_docs=ngram_docs, labels=labels)
            top_k_lmis[n] = {
                label: dict(sorted(
                    lmi.items(), key=operator.itemgetter(1), reverse=True
                )[:top_k])
                for label, lmi in lmis.items()
            }
            if self.config.get("verbose", False):
                print("%d-gram LMI: " % n, top_k_lmis, "\n")
            top_ngrams[n] = []
            for _, lmi in top_k_lmis.items():
                top_ngrams[n].extend(lmi.keys())
        return top_ngrams, top_k_lmis

    def _transform(self, doc: Tuple[str, str]) -> List[float]:
        n_tokens = len(self.map_labels) * sum(self.config.get("top_ks"))
        vec_output = [0, ] * self.n_features

        for n in self.config.get("n_grams"):
            ngram_sent1 = get_ngram_doc(
                doc[0], n=n,
                tokenize=self.tokenizer
            )
            for i, token in enumerate(ngram_sent1):
                if token in self.words_to_idx:
                    idx = self.words_to_idx[token]
                    vec_output[idx] += 1

            ngram_sent2 = get_ngram_doc(
                doc[1], n=n,
                tokenize=self.tokenizer
            )
            for i, token in enumerate(ngram_sent2):
                if token in self.words_to_idx:
                    idx = self.words_to_idx[token]
                    vec_output[idx] += 1

        for i, f in enumerate(self.feature_extractors):
            idx = 2 * n_tokens + i
            vec_output[idx] = f(doc[0], doc[1])
        return vec_output

    def fit(self, docs: List[Tuple[str, str]], labels: List[str]) -> None:
        sent1s = [d[0] for d in docs]
        sent2s = [d[1] for d in docs]
        if self.config.get("verbose", False):
            print("------ Top N-grams for sentence 1 ------")
        self.top_ngrams_sent1, self.top_lmi_sent1 = self._get_top_n_grams(sent1s, labels)
        if self.config.get("verbose", False):
            print("------ Top N-grams for sentence 2 ------")
        self.top_ngrams_sent2, self.top_lmi_sent2 = self._get_top_n_grams(sent2s, labels)

        self.words_to_idx = {}
        for top_ngrams_sent_i in (self.top_ngrams_sent1,  self.top_ngrams_sent2):
            for n in top_ngrams_sent_i:
                for w in top_ngrams_sent_i[n]:
                    self.words_to_idx[w] = len(self.words_to_idx)

        self.n_features = 2 * \
            len(self.map_labels) * sum(self.config.get("top_ks")) + \
            len(self.feature_extractors)
        if self.config.get("verbose", False):
            print("n_features: %d" % self.n_features)

        x = [self._transform(d) for d in docs]
        y = [self.map_labels[lb] for lb in labels]

        x = self.normalizer.fit_transform(x)
        self.model.fit(x, y)

    def inference(self, docs: List[Tuple[str, str]]) -> List[dict]:
        x = [self._transform(doc) for doc in docs]
        x = self.normalizer.transform(x)
        y_preds = self.model.predict_proba(x)

        possible_labels = self.map_labels.keys()
        return [dict(zip(possible_labels, y_pred)) for y_pred in y_preds]

    def predict(self, docs: List[Tuple[str, str]]) -> List[str]:
        inverse_mapper = {v: k for k, v in self.map_labels.items()}
        x = [self._transform(doc) for doc in docs]
        x = self.normalizer.transform(x)
        y_preds = self.model.predict(x)
        return [inverse_mapper[y] for y in y_preds]

import json
from random import random
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import sys

# Config
DUMMY_PREFIX = "" # "sample_" for example and "" for the real one

TRAIN_DATA_FILE = "../../data/fact_verification/%sfever.train.jsonl"%DUMMY_PREFIX
VAL_DATA_FILE = "../../data/fact_verification/%sfever.val.jsonl"%DUMMY_PREFIX
DEV_DATA_FILE = "../../data/fact_verification/%sfever.dev.jsonl"%DUMMY_PREFIX
TEST_DATA_FILE = "../../data/fact_verification/fever_symmetric_v0.1.test.jsonl"

WEIGHT_KEY = "sample_weight"
OUTPUT_VAL_DATA_FILE = "../../data/fact_verification/%sweighted_feverv2.val.jsonl"%DUMMY_PREFIX
OUTPUT_TRAIN_DATA_FILE = "../../data/fact_verification/%sweighted_feverv2.train.jsonl"%DUMMY_PREFIX
SAVED_MODEL_PATH = "results/fever/bias_model"

DOC1_KEY = "claim"
DOC2_KEY = "evidence"
LABEL_KEY = "gold_label"

POSSIBLE_LABELS = ("SUPPORTS", "NOT ENOUGH INFO", "REFUTES")
BIAS_CLASS = "REFUTES"

MAX_SAMPLE = -1 # -1 for non-maximal mode or a finite number e.g. 2000
DROP_RATE = 0.0
TEST_FRAC = 0.2

MAX_TEST_SAMPLE = -1

def read_data(
    file: str = TRAIN_DATA_FILE,
    sent1_key: str = DOC1_KEY,
    sent2_key: str = DOC2_KEY,
    label_key: str = LABEL_KEY,
    drop_rate: float = 0.0
):
    docs = []
    labels = []

    N_SAMPLE = 0

    with open(file, 'r') as fh:
        line = fh.readline()
        while line:
            if random() > drop_rate:
                datapoint = json.loads(line)
                docs.append([datapoint[sent1_key], datapoint[sent2_key]])
                labels.append(datapoint[label_key])

                N_SAMPLE += 1
                if MAX_SAMPLE != -1 and N_SAMPLE == MAX_SAMPLE:
                    break
            line = fh.readline()
    print("# samples: ", N_SAMPLE)
    return docs, labels




if __name__ == "__main__":

    docs, labels = read_data(drop_rate=DROP_RATE)

    docs_train, docs_test, labels_train, labels_test = train_test_split(
        docs, labels,
        stratify=labels, test_size=TEST_FRAC,
        random_state=42
    )

    feature_extractors = [
        lambda s1, s2: count_negations(s1),
        lambda s1, s2: count_negations(s2),
        # get_lexical_overlap,
        # get_entities_overlap
    ]

    config = {
        "n_grams": [1, 2],
        "top_ks": [50, 50], # select by LMI
        "verbose": True,
    }    
    classifier = Classifier(
        possible_labels=POSSIBLE_LABELS,
        feature_extractors=feature_extractors,
        config=config
    )

    classifier.fit(docs_train, labels_train)
    print(classifier.top_lmi_sent1[2]['SUPPORTS'])

    import pickle

    top_lmi_sent1 = classifier.top_lmi_sent1

    with open('top_lmi_sent1.pickle', 'wb') as handle:
        pickle.dump(top_lmi_sent1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    top_lmi_sent2 = classifier.top_lmi_sent2

    with open('top_lmi_sent2.pickle', 'wb') as handle:
        pickle.dump(top_lmi_sent2, handle, protocol=pickle.HIGHEST_PROTOCOL)