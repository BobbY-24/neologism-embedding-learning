# Learning Behavioral Concepts as Single Token Embeddings in Frozen Language Models

> Instead of fine-tuning an entire language model, can we add one new token and optimize only its embedding from preference pairs—and when does that embedding begin to behave like a reusable word rather than a soft control vector?

This research-software repository studies that question with a frozen causal language model and one trainable vector. It preserves both positive and negative results; see the subsequent commits and documentation for the completed experiments.

The implementation is deliberately explicit: response-only teacher-forced log probabilities, a manually implemented DPO-style loss, and an optimizer whose only parameter is the new token embedding.

