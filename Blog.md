**Evaluating Confidence Calibration in LLM-as-Judge Classification Systems**

Large Language Models (LLMs) are increasingly deployed as classifiers in what's commonly referred to as "LLM-as-judge" frameworks. While these models aren't explicitly optimized for classification tasks, they have demonstrated remarkable reliability across diverse domains. However, unlike traditional probabilistic classifiers that output calibrated probability distributions, off-the-shelf LLMs typically provide only discrete predictions without accompanying confidence measures.

This limitation is significant because confidence scores enable crucial downstream applications—such as selective prediction, where low-confidence samples can be flagged for human review, or threshold optimization to achieve desired precision-recall trade-offs. The area under the ROC curve, for instance, can be substantially improved when reliable confidence estimates allow for selective decision thresholds.

A popular solution to this limitation is to prompt LLM judges to provide confidence scores alongside their classifications. However, these confidence estimates represent subjective, non-deterministic assessments that differ fundamentally from the posterior probabilities of traditional probabilistic models. LLM confidence scores emerge from the model's training on human-like reasoning patterns rather than explicit calibration objectives.

This raises critical questions about the reliability and calibration of LLM confidence estimates: Do these subjective confidence scores correlate meaningfully with actual classification accuracy? How consistent are confidence estimates across different representation formats? And do different model architectures exhibit varying degrees of calibration quality?

This technical note presents an empirical investigation into these questions, examining confidence consistency and calibration across multiple LLMs, confidence representation formats, and classification tasks.