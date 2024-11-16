import numpy as np
from collections import Counter
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.spice.spice import Spice

def calculate_bleu(reference, candidate):
    """
    计算BLEU分数
    :param reference: 参考文本，字符串
    :param candidate: 生成的文本，字符串
    :return: BLEU分数
    """
    from utils.BLEU import calculate_bleu  # 假设 BLEU 实现存在于 utils.BLEU 模块中
    return calculate_bleu(reference, candidate)

def calculate_rouge(reference, candidate):
    """
    计算ROUGE分数
    :param reference: 参考文本，字符串
    :param candidate: 生成的文本，字符串
    :return: 包含ROUGE1, ROUGE2和ROUGE-L的分数字典
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def calculate_kl_divergence(reference, candidate):
    """
    计算KL散度
    :param reference: 参考文本，字符串
    :param candidate: 生成的文本，字符串
    :return: KL散度值
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
    计算METEOR分数
    :param reference_list: 参考文本列表，每个元素为一个分词后的参考句子
    :param candidate: 生成的文本，分词后的列表
    :return: METEOR分数
    """
    return meteor_score(reference_list, candidate)

def calculate_spice(reference, candidate):
    """
    计算SPICE分数
    :param reference: 参考文本列表
    :param candidate: 生成的文本列表
    :return: SPICE分数
    """
    gts = {0: reference}
    res = {0: candidate}
    scorer = Spice()
    score, _ = scorer.compute_score(gts, res)
    return score
