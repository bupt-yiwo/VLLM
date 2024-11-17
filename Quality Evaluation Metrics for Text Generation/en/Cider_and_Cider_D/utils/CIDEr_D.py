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
        self.ref_len = None
        self.document_frequency = defaultdict(float)

    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))

    def compute_doc_freq(self):
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            norm = [0.0 for _ in range(self.n)]
            length = 0
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score) / len(refs) * 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self):
        if self.df_mode == "corpus":
            self.compute_doc_freq()
        return np.mean(self.compute_cider()), self.compute_cider()

class CiderD:
    def __init__(self, n=4, sigma=6.0, df="corpus"):
        self._n = n
        self._sigma = sigma
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        tmp_cider_scorer = CiderScorer(df_mode=self._df, n=self._n, sigma=self._sigma)
        tmp_cider_scorer.clear()
        for res_id in res:
            hypo = res_id['caption']
            ref = gts[res_id['image_id']]
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            tmp_cider_scorer.cook_append(hypo[0], ref)
        score, scores = tmp_cider_scorer.compute_score()
        return score, scores

    def method(self):
        return "CIDEr-D"




