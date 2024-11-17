from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from collections import defaultdict
import numpy as np
import math

def precook(s, n=4):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    return precook(test, n)

class CiderScorer:
    def __init__(self, df_mode="corpus", n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.df_mode = df_mode
        self.document_frequency = defaultdict(float)
        self.ref_len = None

    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
            if test is not None:
                self.ctest.append(cook_test(test, self.n))
            else:
                self.ctest.append(None)

    def compute_doc_freq(self):
        for refs in self.crefs:
            for ngram in set(ngram for ref in refs for ngram in ref.keys()):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = self.counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = self.counts2vec(ref)
                score += self.sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            scores.append(np.mean(score) * 10.0 / len(refs))
        return scores

    def counts2vec(self, cnts):
        vec = [defaultdict(float) for _ in range(self.n)]
        length = 0
        norm = [0.0 for _ in range(self.n)]
        for ngram, term_freq in cnts.items():
            n = len(ngram) - 1
            vec[n][ngram] = term_freq
            norm[n] += term_freq ** 2
            if n == 1:
                length += term_freq
        norm = [np.sqrt(n) for n in norm]
        return vec, norm, length

    def sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
        delta = float(length_hyp - length_ref)
        val = np.array([0.0 for _ in range(self.n)])
        for n in range(self.n):
            for ngram, count in vec_hyp[n].items():
                val[n] += vec_hyp[n][ngram] * vec_ref[n].get(ngram, 0)
            if norm_hyp[n] != 0 and norm_ref[n] != 0:
                val[n] /= (norm_hyp[n] * norm_ref[n])
        return val

    def compute_score(self):
        if self.df_mode == "corpus":
            self.compute_doc_freq()
        return np.mean(self.compute_cider()), np.array(self.compute_cider())

class Cider:
    def __init__(self, n=4, df="corpus"):
        self._n = n
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        self.cider_scorer.clear()
        for res_id in res:
            hypo = res_id['caption']
            ref = gts[res_id['image_id']]
            self.cider_scorer.cook_append(hypo[0], ref)
        score, scores = self.cider_scorer.compute_score()
        return score, scores

    def method(self):
        return "CIDEr"

def compute_cider_interface(gts, res):
    cider = Cider()
    score, _ = cider.compute_score(gts, res)
    return score

