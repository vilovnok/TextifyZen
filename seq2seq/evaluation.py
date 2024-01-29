import re

import rusyllab
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# from easse.easse.sari import corpus_sari
# from easse.easse.sari import corpus_sari
# from easse.easse.bleu import corpus_bleu


"""
Implements the evaluation metrics based on BLEU score
"""

import numpy as np
from typing import List

from sacrebleu.metrics import BLEU

# import easse.utils.preprocessing as utils_prep
# import easse.easse.utils.preprocessing as utils_prep


from typing import List

from importlib import import_module

import sacremoses

_TOKENIZERS = {
    "none": "tokenizer_base.BaseTokenizer",
    "13a": "tokenizer_13a.Tokenizer13a",
    "intl": "tokenizer_intl.TokenizerV14International",
}


def _get_tokenizer(name: str):
    """Dynamically import tokenizer as importing all is slow."""
    module_name, class_name = _TOKENIZERS[name].rsplit(".", 1)
    return getattr(import_module(f".tokenizers.{module_name}", "sacrebleu"), class_name)


def normalize(sentence: str, lowercase: bool = True, tokenizer: str = "13a", return_str: bool = True):
    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ["13a", "intl", "none"]:
        tokenizer_obj = _get_tokenizer(name=tokenizer)()
        normalized_sent = tokenizer_obj(sentence)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == "penn":
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent


# ////////////////////////////////

def corpus_bleu(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    smooth_method: str = "exp",
    smooth_value: float = None,
    force: bool = True,
    lowercase: bool = False,
    tokenizer: str = "13a",
    effective_order: bool = False,
):
    sys_sents = [normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[normalize(sent, lowercase, tokenizer) for sent in ref_sents] for ref_sents in refs_sents]

    bleu_scorer = BLEU(lowercase=False, force=force, tokenize="none", smooth_method=smooth_method, smooth_value=smooth_value, effective_order=effective_order)

    return bleu_scorer.corpus_score(
        sys_sents,
        refs_sents,
    ).score


def sentence_bleu(
    sys_sent: str,
    ref_sents: List[str],
    smooth_method: str = "floor",
    smooth_value: float = None,
    lowercase: bool = False,
    tokenizer: str = "13a",
    effective_order: bool = True,
):

    return corpus_bleu(
        [sys_sent],
        [[ref] for ref in ref_sents],
        smooth_method,
        smooth_value,
        force=True,
        lowercase=lowercase,
        tokenizer=tokenizer,
        effective_order=effective_order,
    )


def corpus_averaged_sentence_bleu(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    smooth_method: str = "floor",
    smooth_value: float = None,
    lowercase: bool = False,
    tokenizer: str = "13a",
    effective_order: bool = True,
):

    scores = []
    for sys_sent, *ref_sents in zip(sys_sents, *refs_sents):
        scores.append(
            sentence_bleu(
                sys_sent,
                ref_sents,
                smooth_method,
                smooth_value,
                lowercase=lowercase,
                tokenizer=tokenizer,
                effective_order=effective_order,
            )
        )
    return np.mean(scores)




# /////////////////////////////////////
# from easse.easse.bertscore import corpus_bertscore
# from easse.easse.quality_estimation import corpus_quality_estimation


from typing import List

from bert_score import BERTScorer



def get_bertscore_sentence_scores(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    lowercase: bool = False,
    tokenizer: str = "13a",
):
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    sys_sents = [normalize(sent, lowercase, tokenizer) for sent in sys_sents]
    refs_sents = [[normalize(sent, lowercase, tokenizer) for sent in ref_sents] for ref_sents in refs_sents]
    refs_sents = [list(r) for r in zip(*refs_sents)]

    return scorer.score(sys_sents, refs_sents)


def corpus_bertscore(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    lowercase: bool = False,
    tokenizer: str = "13a",
):
    all_scores = get_bertscore_sentence_scores(sys_sents, refs_sents, lowercase, tokenizer)
    avg_scores = [s.mean(dim=0) for s in all_scores]
    precision = avg_scores[0].cpu().item()
    recall = avg_scores[1].cpu().item()
    f1 = avg_scores[2].cpu().item()
    return precision, recall, f1



# /////////////////////////////////////////////

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import Counter
import itertools
from functools import lru_cache
import os

import Levenshtein
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.base import TransformerMixin
import torch
import torch.nn.functional as F

from tseval.embeddings import to_embeddings
from tseval.evaluation.readability import sentence_fre, sentence_fkgl
from tseval.evaluation.terp import get_terp_vectorizers
from tseval.evaluation.quest import get_quest_vectorizers
from tseval.utils.paths import VARIOUS_DIR
from tseval.text import (count_words, count_sentences, to_words, count_syllables_in_sentence, remove_stopwords,
                         remove_punctuation_tokens)
from tseval.models.language_models import average_sentence_lm_prob, min_sentence_lm_prob
from tseval.utils.helpers import yield_lines


@lru_cache(maxsize=1)
def get_word2concreteness():
    concrete_words_path = os.path.join(VARIOUS_DIR, 'concrete_words.tsv')
    df = pd.read_csv(concrete_words_path, sep='\t')
    df = df[df['Bigram'] == 0]  # Remove bigrams
    return {row['Word']: row['Conc.M'] for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def get_word2frequency():
    frequency_table_path = os.path.join(VARIOUS_DIR, 'enwiki_frequency_table.tsv')
    word2frequency = {}
    for line in yield_lines(frequency_table_path):
        word, frequency = line.split('\t')
        word2frequency[word] = int(frequency)
    return word2frequency


@lru_cache(maxsize=1)
def get_word2rank(vocab_size=50000):
    frequency_table_path = os.path.join(VARIOUS_DIR, 'enwiki_frequency_table.tsv')
    word2rank = {}
    for rank, line in enumerate(yield_lines(frequency_table_path)):
        if (rank+1) > vocab_size:
            break
        word, _ = line.split('\t')
        word2rank[word] = rank
    return word2rank


def get_concreteness(word):
    # TODO: Default value is arbitrary
    return get_word2concreteness().get(word, 5)


def get_frequency(word):
    return get_word2frequency().get(word, None)


def get_negative_frequency(word):
    return -get_frequency(word)


def get_rank(word):
    return get_word2rank().get(word, len(get_word2rank()))


def get_negative_log_frequency(word):
    return -np.log(1 + get_frequency(word))


def get_log_rank(word):
    return np.log(1 + get_rank(word))


# Single sentence feature extractors with signature method(sentence) -> float
def get_concreteness_scores(sentence):
    return np.log(1 + np.array([get_concreteness(word) for word in to_words(sentence)]))


def get_frequency_table_ranks(sentence):
    return np.log(1 + np.array([get_rank(word) for word in to_words(sentence)]))


def get_wordrank_score(sentence):
    # Computed as the third quartile of log ranks
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in get_word2rank()]
    if len(words) == 0:
        return np.log(1 + len(get_word2rank()))  # TODO: This is completely arbitrary
    return np.quantile([get_log_rank(word) for word in words], 0.75)


def count_characters(sentence):
    return len(sentence)


def safe_division(a, b):
    if b == 0:
        return b
    return a / b


def count_words_per_sentence(sentence):
    return safe_division(count_words(sentence), count_sentences(sentence))


def count_characters_per_sentence(sentence):
    return safe_division(count_characters(sentence), count_sentences(sentence))


def count_syllables_per_sentence(sentence):
    return safe_division(count_syllables_in_sentence(sentence), count_sentences(sentence))


def count_characters_per_word(sentence):
    return safe_division(count_characters(sentence), count_words(sentence))


def count_syllables_per_word(sentence):
    return safe_division(count_syllables_in_sentence(sentence), count_words(sentence))


def max_pos_in_freq_table(sentence):
    return max(get_frequency_table_ranks(sentence))


def average_pos_in_freq_table(sentence):
    return np.mean(get_frequency_table_ranks(sentence))


def min_concreteness(sentence):
    return min(get_concreteness_scores(sentence))


def average_concreteness(sentence):
    return np.mean(get_concreteness_scores(sentence))


# OPTIMIZE: Optimize feature extractors? A lot of computation is duplicated (e.g. to_words)
def get_sentence_feature_extractors():
    return [
        count_words,
        count_characters,
        count_sentences,
        count_syllables_in_sentence,
        count_words_per_sentence,
        count_characters_per_sentence,
        count_syllables_per_sentence,
        count_characters_per_word,
        count_syllables_per_word,
        max_pos_in_freq_table,
        average_pos_in_freq_table,
        min_concreteness,
        average_concreteness,
        sentence_fre,
        sentence_fkgl,
        average_sentence_lm_prob,
        min_sentence_lm_prob,
    ]


# Sentence pair feature extractors with signature method(complex_sentence, simple_sentence) -> float
def count_sentence_splits(complex_sentence, simple_sentence):
    return safe_division(count_sentences(simple_sentence), count_sentences(complex_sentence))


def get_compression_ratio(complex_sentence, simple_sentence):
    return safe_division(count_characters(simple_sentence), count_characters(complex_sentence))


def word_intersection(complex_sentence, simple_sentence):
    complex_words = to_words(complex_sentence)
    simple_words = to_words(simple_sentence)
    nb_common_words = len(set(complex_words).intersection(set(simple_words)))
    nb_max_words = max(len(set(complex_words)), len(set(simple_words)))
    return nb_common_words / nb_max_words


@lru_cache(maxsize=10000)
def average_dot(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    return float(torch.dot(complex_embeddings.mean(dim=0), simple_embeddings.mean(dim=0)))


@lru_cache(maxsize=10000)
def average_cosine(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    return float(F.cosine_similarity(complex_embeddings.mean(dim=0),
                                     simple_embeddings.mean(dim=0),
                                     dim=0))


@lru_cache(maxsize=10000)
def hungarian_dot(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    similarity_matrix = torch.mm(complex_embeddings, simple_embeddings.t())
    row_indexes, col_indexes = linear_sum_assignment(-similarity_matrix)
    # TODO: Penalize less deletion of unimportant words
    return float(similarity_matrix[row_indexes, col_indexes].sum() / max(len(complex_sentence), len(simple_sentence)))


@lru_cache(maxsize=10000)
def hungarian_cosine(complex_sentence, simple_sentence):
    complex_embeddings = to_embeddings(complex_sentence)
    simple_embeddings = to_embeddings(simple_sentence)
    similarity_matrix = torch.zeros(len(complex_embeddings), len(simple_embeddings))
    for (i, complex_embedding), (j, simple_embedding) in itertools.product(enumerate(complex_embeddings),
                                                                           enumerate(simple_embeddings)):
        similarity_matrix[i, j] = F.cosine_similarity(complex_embedding, simple_embedding, dim=0)
    row_indexes, col_indexes = linear_sum_assignment(-similarity_matrix)
    # TODO: Penalize less deletion of unimportant words
    return float(similarity_matrix[row_indexes, col_indexes].sum() / max(len(complex_sentence), len(simple_sentence)))


def characters_per_sentence_difference(complex_sentence, simple_sentence):
    return count_characters_per_sentence(complex_sentence) - count_characters_per_sentence(simple_sentence)


def is_exact_match(complex_sentence, simple_sentence):
    return complex_sentence == simple_sentence


def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)


def get_levenshtein_distance(complex_sentence, simple_sentence):
    return 1 - get_levenshtein_similarity(complex_sentence, simple_sentence)


def get_additions_proportion(complex_sentence, simple_sentence):
    n_additions = sum((Counter(to_words(simple_sentence)) - Counter(to_words(complex_sentence))).values())
    return n_additions / max(count_words(complex_sentence), count_words(simple_sentence))


def get_deletions_proportion(complex_sentence, simple_sentence):
    n_deletions = sum((Counter(to_words(complex_sentence)) - Counter(to_words(simple_sentence))).values())
    return n_deletions / max(count_words(complex_sentence), count_words(simple_sentence))


def flatten_counter(counter):
    return [k for key, count in counter.items() for k in [key] * count]


def get_added_words(c, s):
    return flatten_counter(Counter(to_words(s)) - Counter(to_words(c)))


def get_deleted_words(c, s):
    return flatten_counter(Counter(to_words(c)) - Counter(to_words(s)))


def get_kept_words(c, s):
    return flatten_counter(Counter(to_words(c)) & Counter(to_words(s)))


def get_lcs(seq1, seq2):
    '''Returns the longest common subsequence using memoization (only in local scope)'''
    @lru_cache(maxsize=None)
    def recursive_lcs(seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return []
        if seq1[-1] == seq2[-1]:
            return recursive_lcs(seq1[:-1], seq2[:-1]) + [seq1[-1]]
        else:
            return max(recursive_lcs(seq1[:-1], seq2), recursive_lcs(seq1, seq2[:-1]), key=lambda seq: len(seq))

    try:
        return recursive_lcs(tuple(seq1), tuple(seq2))
    except RecursionError as e:
        print(e)
        # TODO: Handle this case
        return []


def get_reordered_words(c, s):
    # A reordered word is a word that is contained in the source and simplification
    # but not in the longuest common subsequence
    c = c.lower()
    s = s.lower()
    lcs = get_lcs(to_words(c), to_words(s))
    return flatten_counter(Counter(get_kept_words(c, s)) - Counter(lcs))


def get_n_added_words(c, s):
    return len(get_added_words(c, s))


def get_n_deleted_words(c, s):
    return len(get_deleted_words(c, s))


def get_n_kept_words(c, s):
    return len(get_kept_words(c, s))


def get_n_reordered_words(c, s):
    return len(get_reordered_words(c, s))


def get_added_words_proportion(c, s):
    # TODO: Duplicate of get_addition_proportion, same for deletion
    # Relative to simple sentence
    return get_n_added_words(c, s) / count_words(s)


def get_deleted_words_proportion(c, s):
    # Relative to complex sentence
    return get_n_deleted_words(c, s) / count_words(c)


def get_reordered_words_proportion(c, s):
    # Relative to complex sentence
    return get_n_deleted_words(c, s) / count_words(s)


def only_deleted_words(c, s):
    # Only counting deleted words does not work because sometimes there is reordering
    return not is_exact_match(c, s) and get_lcs(to_words(c), to_words(s)) == to_words(s)


# @lru_cache(maxsize=1)
# def get_nlgeval():
#     try:
#         from nlgeval import NLGEval
#     except ModuleNotFoundError:
#         print('nlg-eval module not installed. Please install with ',
#               'pip install nlg-eval@git+https://github.com/Maluuba/nlg-eval.git')
#     print('Loading NLGEval models...')
#     return NLGEval(no_skipthoughts=True, no_glove=True)


# # Making one call to nlgeval returns all metrics, we therefore cache the results in order to limit the number of calls
# @lru_cache(maxsize=10000)
# def get_all_nlgeval_metrics(complex_sentence, simple_sentence):
#     return get_nlgeval().compute_individual_metrics([complex_sentence], simple_sentence)


# def get_nlgeval_methods():
#     """Returns all scoring methods from nlgeval package.

#     Signature: method(complex_sentence, simple_setence)
#     """
#     def get_scoring_method(metric_name):
#         """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
#         its current value."""
#         def scoring_method(complex_sentence, simple_sentence):
#             return get_all_nlgeval_metrics(complex_sentence, simple_sentence)[metric_name]
#         return scoring_method

#     nlgeval_metrics = [
#         # Fast metrics
#         'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr',
#         # Slow metrics
#         # 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore',
#     ]
#     methods = []
#     for metric_name in nlgeval_metrics:
#         scoring_method = get_scoring_method(metric_name)
#         scoring_method.__name__ = f'nlgeval_{metric_name}'
#         methods.append(scoring_method)
#     return methods


def get_nltk_bleu_methods():
    """Returns bleu methods with different smoothings from NLTK.
Signature: scoring_method(complex_sentence, simple_setence)
    """
    # Inline lazy import because importing nltk is slow
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    def get_scoring_method(smoothing_function):
        """Necessary to wrap the scoring_method() in get_scoring_method(), in order to set the external variable to
        its current value."""
        def scoring_method(complex_sentence, simple_sentence):
            try:
                return sentence_bleu([complex_sentence.split()], simple_sentence.split(),
                                     smoothing_function=smoothing_function)
            except AssertionError:
                return 0
        return scoring_method

    methods = []
    for i in range(8):
        smoothing_function = getattr(SmoothingFunction(), f'method{i}')
        scoring_method = get_scoring_method(smoothing_function)
        scoring_method.__name__ = f'nltkBLEU_method{i}'
        methods.append(scoring_method)
    return methods


# def get_sentence_pair_feature_extractors():
#     return [
#         word_intersection,
#         characters_per_sentence_difference,
#         average_dot,
#         average_cosine,
#         hungarian_dot,
#         hungarian_cosine,
#     ] + get_nlgeval_methods() + get_nltk_bleu_methods() + get_terp_vectorizers() + get_quest_vectorizers()


# Various
def wrap_single_sentence_vectorizer(vectorizer):
    '''Transform a single sentence vectorizer to a sentence pair vectorizer

    Change the signature of the input vectorizer
    Initial signature: method(simple_sentence)
    New signature: method(complex_sentence, simple_sentence)
    '''
    def wrapped(complex_sentence, simple_sentence):
        return vectorizer(simple_sentence)

    wrapped.__name__ = vectorizer.__name__
    return wrapped


def reverse_vectorizer(vectorizer):
    '''Reverse the arguments of a vectorizer'''
    def reversed_vectorizer(complex_sentence, simple_sentence):
        return vectorizer(simple_sentence, complex_sentence)

    reversed_vectorizer.__name__ = vectorizer.__name__ + '_reversed'
    return reversed_vectorizer


# def get_all_vectorizers(reversed=False):
#     vectorizers = [wrap_single_sentence_vectorizer(vectorizer)
#                    for vectorizer in get_sentence_feature_extractors()] + get_sentence_pair_feature_extractors()
#     if reversed:
#         vectorizers += [reverse_vectorizer(vectorizer) for vectorizer in vectorizers]
#     return vectorizers


def concatenate_corpus_vectorizers(vectorizers):
    '''Given a list of corpus vectorizers, create a new single concatenated corpus vectorizer.

    Corpus vectorizer:
    Given a numpy array of shape (n_samples, 2), it will extract features for each sentence pair
    and output a (n_samples, n_features) array.
    '''
    def concatenated(sentence_pairs):
        return np.column_stack([vectorizer(sentence_pairs) for vectorizer in vectorizers])
    return concatenated


class FeatureSkewer(TransformerMixin):
    '''Normalize features that have a skewed distribution'''
    def fit(self, X, y):
        self.skewed_indexes = [i for i in range(X.shape[1]) if skew(X[:, i]) > 0.75]
        return self

    def transform(self, X):
        for i in self.skewed_indexes:
            X[:, i] = boxcox1p(X[:, i], 0)
        return np.nan_to_num(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)




def get_average(vectorizer, orig_sentences, sys_sentences):
    cumsum = 0
    count = 0
    for orig_sentence, sys_sentence in zip(orig_sentences, sys_sentences):
        cumsum += vectorizer(orig_sentence, sys_sentence)
        count += 1
    return cumsum / count


def corpus_quality_estimation(
    orig_sentences: List[str], sys_sentences: List[str], lowercase: bool = False, tokenizer: str = '13a'
):
    orig_sentences = [normalize(sent, lowercase, tokenizer) for sent in orig_sentences]
    sys_sentences = [normalize(sent, lowercase, tokenizer) for sent in sys_sentences]
    return {
        'Compression ratio': get_average(get_compression_ratio, orig_sentences, sys_sentences),
        'Sentence splits': get_average(count_sentence_splits, orig_sentences, sys_sentences),
        'Levenshtein similarity': get_average(get_levenshtein_similarity, orig_sentences, sys_sentences),
        'Exact copies': get_average(is_exact_match, orig_sentences, sys_sentences),
        'Additions proportion': get_average(get_additions_proportion, orig_sentences, sys_sentences),
        'Deletions proportion': get_average(get_deletions_proportion, orig_sentences, sys_sentences),
        'Lexical complexity score': get_average(
            wrap_single_sentence_vectorizer(get_wordrank_score), orig_sentences, sys_sentences
        ),
    }







from collections import Counter
from typing import List


# import easse.easse.utils.preprocessing as utils_prep


from typing import List

from importlib import import_module

import sacremoses

_TOKENIZERS = {
    "none": "tokenizer_base.BaseTokenizer",
    "13a": "tokenizer_13a.Tokenizer13a",
    "intl": "tokenizer_intl.TokenizerV14International",
}


def _get_tokenizer(name: str):
    """Dynamically import tokenizer as importing all is slow."""
    module_name, class_name = _TOKENIZERS[name].rsplit(".", 1)
    return getattr(import_module(f".tokenizers.{module_name}", "sacrebleu"), class_name)


def normalize(sentence: str, lowercase: bool = True, tokenizer: str = "13a", return_str: bool = True):
    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ["13a", "intl", "none"]:
        tokenizer_obj = _get_tokenizer(name=tokenizer)()
        normalized_sent = tokenizer_obj(sentence)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == "penn":
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent


# ////////////////////////////////
NGRAM_ORDER = 4


def compute_precision_recall(sys_correct, sys_total, ref_total):
    precision = 0.0
    if sys_total > 0:
        precision = sys_correct / sys_total

    recall = 0.0
    if ref_total > 0:
        recall = sys_correct / ref_total

    return precision, recall


def compute_f1(precision, recall):
    f1 = 0.0
    if precision > 0 or recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_micro_sari(
    add_hyp_correct,
    add_hyp_total,
    add_ref_total,
    keep_hyp_correct,
    keep_hyp_total,
    keep_ref_total,
    del_hyp_correct,
    del_hyp_total,
    del_ref_total,
    use_f1_for_deletion=True,
):
    """
    This is the version described in the original paper. Follows the equations.
    """
    add_precision = [0] * NGRAM_ORDER
    add_recall = [0] * NGRAM_ORDER
    keep_precision = [0] * NGRAM_ORDER
    keep_recall = [0] * NGRAM_ORDER
    del_precision = [0] * NGRAM_ORDER
    del_recall = [0] * NGRAM_ORDER

    for n in range(NGRAM_ORDER):
        add_precision[n], add_recall[n] = compute_precision_recall(
            add_hyp_correct[n], add_hyp_total[n], add_ref_total[n]
        )
        keep_precision[n], keep_recall[n] = compute_precision_recall(
            keep_hyp_correct[n], keep_hyp_total[n], keep_ref_total[n]
        )
        del_precision[n], del_recall[n] = compute_precision_recall(
            del_hyp_correct[n], del_hyp_total[n], del_ref_total[n]
        )

    avg_add_precision = sum(add_precision) / NGRAM_ORDER
    avg_add_recall = sum(add_recall) / NGRAM_ORDER
    avg_keep_precision = sum(keep_precision) / NGRAM_ORDER
    avg_keep_recall = sum(keep_recall) / NGRAM_ORDER
    avg_del_precision = sum(del_precision) / NGRAM_ORDER
    avg_del_recall = sum(del_recall) / NGRAM_ORDER

    add_f1 = compute_f1(avg_add_precision, avg_add_recall)

    keep_f1 = compute_f1(avg_keep_precision, avg_keep_recall)

    if use_f1_for_deletion:
        del_score = compute_f1(avg_del_precision, avg_del_recall)
    else:
        del_score = avg_del_precision

    return add_f1, keep_f1, del_score


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> List[Counter]:
    ngrams_per_order = []
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        ngrams = Counter()
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams[ngram] += 1
        ngrams_per_order.append(ngrams)

    return ngrams_per_order


def multiply_counter(c, v):
    c_aux = Counter()
    for k in c.keys():
        c_aux[k] = c[k] * v

    return c_aux


def compute_ngram_stats(
    orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]]
):
    add_sys_correct = [0] * NGRAM_ORDER
    add_sys_total = [0] * NGRAM_ORDER
    add_ref_total = [0] * NGRAM_ORDER

    keep_sys_correct = [0] * NGRAM_ORDER
    keep_sys_total = [0] * NGRAM_ORDER
    keep_ref_total = [0] * NGRAM_ORDER

    del_sys_correct = [0] * NGRAM_ORDER
    del_sys_total = [0] * NGRAM_ORDER
    del_ref_total = [0] * NGRAM_ORDER

    for orig_sent, sys_sent, *ref_sents in zip(
        orig_sents, sys_sents, *refs_sents
    ):
        orig_ngrams = extract_ngrams(orig_sent)
        sys_ngrams = extract_ngrams(sys_sent)
        ref_sents = [sent for sent in ref_sents if sent.strip() != '']
        refs_ngrams = [Counter() for _ in range(NGRAM_ORDER)]
        for ref_sent in ref_sents:
            ref_ngrams = extract_ngrams(ref_sent)
            for n in range(NGRAM_ORDER):
                refs_ngrams[n] += ref_ngrams[n]
        num_refs = len(ref_sents)
        for n in range(NGRAM_ORDER):
            # ADD
            # added by the hypothesis (binary)
            sys_and_not_orig = set(sys_ngrams[n]) - set(orig_ngrams[n])
            add_sys_total[n] += len(sys_and_not_orig)
            # added by the references (binary)
            ref_and_not_orig = set(refs_ngrams[n]) - set(orig_ngrams[n])
            add_ref_total[n] += len(ref_and_not_orig)
            # added correctly (binary)
            add_sys_correct[n] += len(sys_and_not_orig & set(refs_ngrams[n]))

            # KEEP
            # kept by the hypothesis (weighted)
            orig_and_sys = multiply_counter(
                orig_ngrams[n], num_refs
            ) & multiply_counter(sys_ngrams[n], num_refs)
            keep_sys_total[n] += sum(orig_and_sys.values())
            # kept by the references (weighted)
            orig_and_ref = (
                multiply_counter(orig_ngrams[n], num_refs) & refs_ngrams[n]
            )
            keep_ref_total[n] += sum(orig_and_ref.values())
            # kept correctly?
            keep_sys_correct[n] += sum((orig_and_sys & orig_and_ref).values())

            # DELETE
            # deleted by the hypothesis (weighted)
            orig_and_not_sys = multiply_counter(
                orig_ngrams[n], num_refs
            ) - multiply_counter(sys_ngrams[n], num_refs)
            del_sys_total[n] += sum(orig_and_not_sys.values())
            # deleted by the references (weighted)
            orig_and_not_ref = (
                multiply_counter(orig_ngrams[n], num_refs) - refs_ngrams[n]
            )
            del_ref_total[n] += sum(orig_and_not_ref.values())
            # deleted correctly
            del_sys_correct[n] += sum(
                (orig_and_not_sys & orig_and_not_ref).values()
            )

    return (
        add_sys_correct,
        add_sys_total,
        add_ref_total,
        keep_sys_correct,
        keep_sys_total,
        keep_ref_total,
        del_sys_correct,
        del_sys_total,
        del_ref_total,
    )


def compute_precision_recall_f1(sys_correct, sys_total, ref_total):
    precision = 0.0
    if sys_total > 0:
        precision = sys_correct / sys_total

    recall = 0.0
    if ref_total > 0:
        recall = sys_correct / ref_total

    f1 = 0.0
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_macro_sari(
    add_sys_correct,
    add_sys_total,
    add_ref_total,
    keep_sys_correct,
    keep_sys_total,
    keep_ref_total,
    del_sys_correct,
    del_sys_total,
    del_ref_total,
    use_f1_for_deletion=True,
):
    """
    This is the version released as a JAVA implementation and which was used in their experiments,
    as stated by the authors: https://github.com/cocoxu/simplification/issues/8
    """
    add_f1 = 0.0
    keep_f1 = 0.0
    del_f1 = 0.0
    for n in range(NGRAM_ORDER):
        _, _, add_f1_ngram = compute_precision_recall_f1(
            add_sys_correct[n], add_sys_total[n], add_ref_total[n]
        )
        _, _, keep_f1_ngram = compute_precision_recall_f1(
            keep_sys_correct[n], keep_sys_total[n], keep_ref_total[n]
        )
        if use_f1_for_deletion:
            _, _, del_score_ngram = compute_precision_recall_f1(
                del_sys_correct[n], del_sys_total[n], del_ref_total[n]
            )
        else:
            del_score_ngram, _, _ = compute_precision_recall_f1(del_sys_correct[n], del_sys_total[n], del_ref_total[n])
        add_f1 += add_f1_ngram / NGRAM_ORDER
        keep_f1 += keep_f1_ngram / NGRAM_ORDER
        del_f1 += del_score_ngram / NGRAM_ORDER
    return add_f1, keep_f1, del_f1


def get_corpus_sari_operation_scores(orig_sents: List[str], sys_sents: List[str], refs_sents: List[List[str]],
                                     lowercase: bool = True, tokenizer: str = '13a',
                                     legacy=False, use_f1_for_deletion=True, use_paper_version=False):
    """The `legacy` parameter allows reproducing scores reported in previous work.
    It replicates a bug in the original JAVA implementation where only the system outputs and the reference sentences
    are further tokenized. 
    In addition, it assumes that all sentences are already lowercased. """
    if legacy:
        lowercase = False
    else:
        orig_sents = [
            normalize(sent, lowercase, tokenizer)
            for sent in orig_sents
        ]

    sys_sents = [
        normalize(sent, lowercase, tokenizer) for sent in sys_sents
    ]
    refs_sents = [
        [normalize(sent, lowercase, tokenizer) for sent in ref_sents]
        for ref_sents in refs_sents
    ]
    stats = compute_ngram_stats(orig_sents, sys_sents, refs_sents)

    if not use_paper_version:
        add_score, keep_score, del_score = compute_macro_sari(*stats, use_f1_for_deletion=use_f1_for_deletion)
    else:
        add_score, keep_score, del_score = compute_micro_sari(*stats, use_f1_for_deletion=use_f1_for_deletion)
    return 100. * add_score, 100. * keep_score, 100. * del_score


def corpus_sari(*args, **kwargs):
    add_score, keep_score, del_score = get_corpus_sari_operation_scores(*args, **kwargs)
    return (add_score + keep_score + del_score) / 3









# ///////////////////

def corpus_fkgl(sys_sents, sent_tokenize=sent_tokenize):
    """
    Считаем FKGL (https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
    Используется формула Обороневой (адаптированная для русского языка)
    :param sys_sents: Упрощеннные предложения, сгенерированные моделью
    :param sent_tokenize: Токенизатор для разделения текста на предложения
    :return:
    """
    word_pattern = re.compile('[А-яA-z]+')
    w = 0
    c = 0
    s = 0
    for sent in sys_sents:
        # считаем количество предложений (модель может дробить одно предложение на несколько)
        s += len(sent_tokenize(sent))
        # считаем количество слов (без учета знаков препинания)
        words = word_pattern.findall(sent)
        w += len(words)
        # считаем количество слогов
        syllables = rusyllab.split_words(''.join(words).split())
        c += sum(1 for syl in syllables if syl != ' ')
    return 206.836 - (65.14 * c / w) - (1.52 * w / s)


METRIC_FUNCS = {
    'bleu': {'func': corpus_bleu, 'requires_refs': True, 'requires_orig': False},
    'sari': {'func': corpus_sari, 'requires_refs': True, 'requires_orig': True},
    'bertscore': {'func': corpus_bertscore, 'requires_refs': True, 'requires_orig': False},
    'fkgl': {'func': corpus_fkgl, 'requires_refs': False, 'requires_orig': False},
}


def compute_corpus_metrics(orig, refs, simplification_func, compute_quality_estimation=True,
                           metrics=('bleu', 'sari', 'fkgl'), **kwargs):
    """
    Подсчет корпусных метрик
    :param orig: Сложные предложения
    :param refs: Упрощенные предложения из датасета
    :param simplification_func: Функция, упрощающая предложения
    :param compute_quality_estimation: Вычислять ли quality_estimation из библиотеки easse
    :param metrics: Список метрик, которые будут вычислены {'bleu', 'sari', 'fkgl', 'bertscore'}
    :return: computed_metrics: Словарь с метриками
             quality: Метрики качества из библиотеки easse
    """
    computed_metrics = {}
    quality = {}
    assert (len(refs) == len(orig))
    preds = simplification_func(orig, **kwargs)
    assert (len(preds) == len(orig))
    for metric in metrics:
        compute_metric = METRIC_FUNCS[metric]['func']
        kwargs = {'sys_sents': preds}
        if METRIC_FUNCS[metric]['requires_refs']:
            kwargs.update({'refs_sents': refs})
        if METRIC_FUNCS[metric]['requires_orig']:
            kwargs.update({'orig_sents': orig})
        # для метрики bertscore возвращаются значения precision, recall и f1. берем только f1
        if metric == 'bertscore':
            computed_metric = compute_metric(**kwargs)[2]
        else:
            computed_metric = compute_metric(**kwargs)
        computed_metrics.update({metric: round(computed_metric, 3)})
    if compute_quality_estimation:
        quality = corpus_quality_estimation(sys_sentences=preds, orig_sentences=orig)
    return computed_metrics, quality
