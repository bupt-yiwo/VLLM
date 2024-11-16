from scoring_functions import (
    calculate_bleu,
    calculate_rouge,
    calculate_kl_divergence,
    calculate_meteor,
    calculate_spice
)

# 示例文本
reference = "It is a guide to action which ensures that the"
candidate = "It is a guide to action which ensures that the military always obeys the commands of the party"

# 计算BLEU分数
bleu_score = calculate_bleu(reference, candidate)
print(f"BLEU Score: {bleu_score}")

# 计算ROUGE分数
rouge_scores = calculate_rouge(reference, candidate)
print(f"ROUGE Scores: {rouge_scores}")

# 计算KL散度
kl_score = calculate_kl_divergence(reference, candidate)
print(f"KL Divergence: {kl_score}")

# 计算METEOR分数
reference_list=[list("我说这是怎么回事，原来明天要放假了"),list("我说这是怎么回事")]
hypothesis=list("我说这是啥呢我说这是啥呢")
meteor_score1 = calculate_meteor(reference_list, hypothesis)
print(f"METEOR Score (Chinese Example): {meteor_score1}")

reference = ["this is a test".split()]
hypothesis = "this is test".split()
meteor_score2 = calculate_meteor(reference, hypothesis)
print(f"METEOR Score (English Example): {meteor_score2}")

# 计算SPICE分数
spice_reference = ["It is a dog"]
spice_candidate = ["It is a guide to action which ensures that the aa"]
spice_score = calculate_spice(spice_reference, spice_candidate)
print(f"SPICE Score: {spice_score}")

BLEU,ROUGE,KL散度,METEOR分数,SPICE分数