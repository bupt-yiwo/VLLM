import numpy as np
from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.spice.spice import Spice

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score
    :param reference: Reference text, string
    :param candidate: Generated text, string
    :return: BLEU score
    """
    from utils.BLEU import calculate_bleu  
    return calculate_bleu(reference, candidate)

def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE score
    :param reference: Reference text, string
    :param candidate: Generated text, string
    :return: Dictionary containing ROUGE1, ROUGE2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def calculate_kl_divergence(reference, candidate):
    """
    Calculate KL Divergence
    :param reference: Reference text, string
    :param candidate: Generated text, string
    :return: KL divergence value
    """
    def calculate_word_distribution(text):
        words = text.split()
        total_words = len(words)
        word_counts = Counter(words)
        return {word: count / total_words for word, count in word_counts.items()}
    
    p_dist = calculate_word_distribution(reference)
    q_dist = calculate_word_distribution(candidate)
    all_words = set(p_dist.keys()).union(set(q_dist.keys()))
    p = np.array([p_dist.get(word, 1e-10) for word in all_words])
    q = np.array([q_dist.get(word, 1e-10) for word in all_words])
    return np.sum(p * np.log(p / q))

def calculate_meteor(reference_list, candidate):
    """
    Calculate METEOR score
    :param reference_list: List of reference texts, each element is a tokenized reference sentence
    :param candidate: Generated text, tokenized list
    :return: METEOR score
    """
    return meteor_score(reference_list, candidate)

def calculate_spice(reference, candidate):
    """
    Calculate SPICE score
    :param reference: List of reference texts
    :param candidate: List of generated texts
    :return: SPICE score
    """
    gts = {0: reference}
    res = {0: candidate}
    scorer = Spice()
    score, _ = scorer.compute_score(gts, res)
    return score
