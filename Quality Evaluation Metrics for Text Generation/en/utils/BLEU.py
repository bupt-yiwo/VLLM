import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

def preprocess_sentence(sentence):
    """Handle punctuation and numbers"""
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    sentence = re.sub(r'\d+', ' <NUM> ', sentence)  # Replace digits with a placeholder
    sentence = sentence.strip()  # Remove leading and trailing whitespace
    return sentence

def calculate_bleu(reference, candidate):
    """
    Calculate the BLEU score between two sentences
    
    Parameters:
    reference (str): Reference sentence
    candidate (str): Candidate sentence

    Returns:
    float: BLEU score
    """
    reference_list = [preprocess_sentence(reference).split()]
    candidate_list = preprocess_sentence(candidate).split()
    
    smooth = SmoothingFunction().method0  # Apply smoothing to avoid zero scores for short sequences
    bleu_score = sentence_bleu(reference_list, candidate_list, smoothing_function=smooth)
    
    return bleu_score
