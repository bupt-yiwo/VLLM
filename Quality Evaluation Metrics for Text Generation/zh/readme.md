1. **BLEU (Bilingual Evaluation Understudy)**：主要用于机器翻译任务，通过计算生成文本与参考文本之间的n元组(n-gram)匹配程度来评估翻译质量，得分在0到1之间，越高表示生成文本越接近参考文本。通常会用加权的精确度来处理重复的情况。
2. 
3. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**：主要用于文本摘要任务，关注生成文本和参考文本之间的重叠部分，通常计算n元组、最长公共子序列（Longest Common Subsequence）等。常见的变种有ROUGE-N、ROUGE-L等，分别侧重于不同的重叠特性。
4. 
5. **KL散度 (Kullback–Leibler Divergence)**：用于衡量两个概率分布之间的差异。通常用于评价生成模型的分布与真实分布的接近程度。KL散度越小，表示生成的分布越接近真实分布。
6. 
7. **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**：一种文本评价指标，考虑了词形变化、同义词匹配等因素。相比BLEU，METEOR更注重单词顺序和语义相关性，常用于机器翻译和文本生成的评估。
8. 
9. **SPICE (Semantic Propositional Image Captioning Evaluation)**：主要用于图像描述生成任务，评估生成文本在语义层面上与参考文本的匹配程度。SPICE基于图形结构对文本进行解析，重视生成文本的语义内容。

