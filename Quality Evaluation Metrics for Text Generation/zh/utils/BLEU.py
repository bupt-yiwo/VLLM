import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

def preprocess_sentence(sentence):
    """处理标点符号和数字"""
    sentence = re.sub(r'[^\w\s]', '', sentence) 
    sentence = re.sub(r'\d+', ' <NUM> ', sentence)  
    sentence = sentence.strip()  
    return sentence

def calculate_bleu(reference, candidate):
    """
    计算两个句子的 BLEU 分数
    
    参数:
    reference (str): 参考句子
    candidate (str): 候选句子

    返回:
    float: BLEU 分数
    """
    reference_list = [preprocess_sentence(reference).split()]
    candidate_list = preprocess_sentence(candidate).split()
    
    smooth = SmoothingFunction().method0
    bleu_score = sentence_bleu(reference_list, candidate_list, smoothing_function=smooth)
    
    return bleu_score
