**BLEU (Bilingual Evaluation Understudy)**: Primarily used for machine translation tasks, it evaluates translation quality by calculating the n-gram overlap between the generated text and reference texts. The score ranges from 0 to 1, with a higher score indicating that the generated text is closer to the reference text. It typically uses weighted precision to handle repetition issues.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Mainly used for text summarization tasks, it focuses on overlapping units between generated and reference texts. It calculates n-gram overlap, longest common subsequence, and more. Common variants include ROUGE-N and ROUGE-L, each emphasizing different overlapping characteristics.

**KL Divergence (Kullback–Leibler Divergence)**: Measures the difference between two probability distributions, often used to evaluate how close the generated distribution is to the real distribution. Smaller KL divergence indicates that the generated distribution is closer to the true distribution.

**METEOR (Metric for Evaluation of Translation with Explicit ORdering)**: This metric considers word variations, synonyms, etc., and focuses more on word order and semantic correlation compared to BLEU. It is widely used for evaluating translations and text generation.

**SPICE (Semantic Propositional Image Captioning Evaluation)**: Primarily used for evaluating image caption generation, it measures semantic matching between generated and reference texts. SPICE parses text using graph structures and emphasizes the semantic content of the generated text.


In addition to the metrics mentioned earlier, CIDEr and CIDEr-D can sometimes also be used to evaluate the quality of text generation. However, they can have limitations in cases with a single reference answer or a single generated response.

**CIDEr (Consensus-based Image Description Evaluation)**: This metric is primarily used for image captioning tasks, evaluating the quality of generated descriptions by comparing them with reference descriptions. CIDEr utilizes a weighted TF-IDF representation of n-grams and measures the semantic similarity between generated and reference descriptions. This approach better captures the actual meaning of the text by accounting for different expressions that convey the same semantic content. By comparing with multiple reference sentences, it enhances the evaluation of description diversity and coverage.

**CIDEr-D**: CIDEr-D is an improved version of CIDEr that focuses on enhancing diversity and reducing repetition in generated descriptions. By penalizing repeated n-grams, CIDEr-D discourages excessive generation of simple phrases or word repetitions, thus encouraging more informative and diverse content. Additionally, CIDEr-D includes smoothing to increase robustness, making it more stable when handling low-frequency or unseen n-grams.
