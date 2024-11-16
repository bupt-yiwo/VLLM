**BLEU (Bilingual Evaluation Understudy)**: Primarily used for machine translation tasks, it evaluates translation quality by calculating the n-gram overlap between the generated text and reference texts. The score ranges from 0 to 1, with a higher score indicating that the generated text is closer to the reference text. It typically uses weighted precision to handle repetition issues.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Mainly used for text summarization tasks, it focuses on overlapping units between generated and reference texts. It calculates n-gram overlap, longest common subsequence, and more. Common variants include ROUGE-N and ROUGE-L, each emphasizing different overlapping characteristics.

**KL Divergence (Kullback–Leibler Divergence)**: Measures the difference between two probability distributions, often used to evaluate how close the generated distribution is to the real distribution. Smaller KL divergence indicates that the generated distribution is closer to the true distribution.

**METEOR (Metric for Evaluation of Translation with Explicit ORdering)**: This metric considers word variations, synonyms, etc., and focuses more on word order and semantic correlation compared to BLEU. It is widely used for evaluating translations and text generation.

**SPICE (Semantic Propositional Image Captioning Evaluation)**: Primarily used for evaluating image caption generation, it measures semantic matching between generated and reference texts. SPICE parses text using graph structures and emphasizes the semantic content of the generated text.
