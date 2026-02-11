# Refusal Directions in Instruction-Tuned Models

This project explores how refusal behavior is represented inside instruction-tuned language models with the central question being: is refusal encoded as a single linear direction in the residual stream, or as a higher-dimensional structure?

Across models, both patterns appear.

---

## Detailed Writeups

- [Qwen-1.8B-Chat](qwen.md)  
- [LLaMA-3.2-1B-Instruct](llama_1b.md)  
- [Refusal Structure Across Models](refusal_across_models.md)

These pages contain:

- Refusal vector extraction
- Cross-layer cosine analysis
- Runtime ablation experiments
- Subspace interventions (where applicable)
- Offline weight orthogonalization
- Benchmark evaluation
