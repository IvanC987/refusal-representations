# Refusal as a Linear Representation in Instruction-Tuned LLMs

Hi everyone, this project was created to explore how refusal is represented in LLMs and to answer a few questions, among which:

> Is refusal behavior in instruction-tuned language models represented as a single linear direction in the residual stream or something more distributed?

I recently came across an Alignment Forum [post](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ) (from 2024) showing that in some models, you can remove refusal by subtracting a single vector from the residual stream. As a result, the model starts complying with clearly harmful prompts, often with surprisingly little performance degradation.

Intrigued, I wanted to see how general that phenomenon really is by replicating the original work and extending it further.

---

## What I Found

Across several instruction-tuned models under ~10B parameters, two different patterns show up.

Some models behave like this:

* **Qwen-1.8B-Chat**
* **Qwen3-1.7B**
* **Gemma-2B-IT**
* **LLaMA-3.1-8B-Instruct**

In these models, refusal mostly collapses to a dominant direction in the residual stream. If you remove that direction:

* Harmful compliance often jumps to 80-95%
* Coherence stays high
* Benchmark performance drops only slightly

In other words, refusal looks linearly accessible and fairly centralized.

But other models look very different:

* **LLaMA-3.2-1B-Instruct**
* **LLaMA-3.2-3B-Instruct**
* **Phi-3-Mini** (partial)

Here, no single direction is enough. You can weaken refusal, but you don't break it. Compliance plateaus around ~15-25% under single-vector ablation. Stacking multiple vectors into a small subspace helps, but still doesn't saturate compliance.

In these models, refusal seems distributed across a low-rank subspace rather than concentrated along one axis.

So refusal structure is not universal.

---

## What's in This Repo

This repository contains:

* Refusal vector extraction (sequential and batched)
* Residual-stream runtime ablation
* Offline weight orthogonalization ("abliteration")
* Cross-layer cosine similarity analysis
* Cohen's d separability measurements
* Compliance / coherence scoring via an external LLM judge
* Benchmark evaluation using `lm-eval-harness`

There's also [this](https://ivanc987.github.io/refusal-representations/) primary GitHub page where I discuss extraction, ablation, qualitative behavior, and benchmarking results in detail.

---

## High-Level Method

For each model:

1. Construct paired harmful vs safe prompts.

2. Extract layer-wise refusal vectors from residual stream differences.

3. Measure separability (Cohen's d).

4. Analyze cross-layer cosine similarity.

5. Perform runtime ablation:

   r ← r − α⟨r, v⟩v

6. Score compliance and coherence using an LLM judge.

7. Orthogonalize model weights and benchmark downstream performance.

Nothing exotic, mostly linear algebra and careful bookkeeping.

---

## Limitations & Responsible Release Note

* All models tested are under ~10B parameters.
* Compliance scoring uses an LLM judge.
* Results depend on prompt construction.
* This describes empirical structure, not deep mechanistic causality.

This project studies the *representation* of refusal behavior in instruction-tuned models. Because the experiments involve harmful prompt categories, the harmful/safe prompt dataset used for extraction and evaluation is not included in this repository. The structure of the dataset is straightforward (see below), and anyone interested in reproducing the results can construct their own prompt pairs.

The intent of this work is to analyze safety representations, not to facilitate misuse.

---

## Dataset Format (Not Included)

The extraction dataset consists of paired harmful/safe prompts with the structure:

```
List[List[str, str]]
```

A list of ~1000 pairs, where each inner list contains:

```
[harmful_prompt, safe_prompt]
```

The evaluation set used for compliance scoring is:

```
List[str]
```

A list of harmful prompts used for generation and scoring.

The exact prompts are not distributed, but the format is simple to reproduce.

---

## Acknowledgements

This work builds on prior public discussions and resources, including:

* The Alignment Forum [post](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ) on linear refusal directions and its associated Colab notebook (2024)
* The Hugging Face blog [post](https://huggingface.co/blog/mlabonne/abliteration) by Maxime Labonne on weight orthogonalization ("abliteration")

These sources provided both conceptual motivation and practical reference points for residual stream intervention and offline orthogonalization.

This project started as a personal investigation and grew into a larger comparison across model families. A longer-form writeup will likely appear on LessWrong.

If you're mainly interested in the conclusions, start with `Refusal Structure Across Models` section in my github pages.


