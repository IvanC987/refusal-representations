# Refusal Structure Across Models

After digging into Qwen-1.8B-Chat and LLaMA-3.2-1B-Instruct in detail, the next obvious question is whether either of those behaviors is typical. In particular: is LLaMA-1B just a weird outlier, or are there genuinely two different "modes" of how refusal is implemented across models?

From the models I’ve tested so far, it looks like the latter. Some models implement refusal in a way that collapses cleanly onto a single dominant direction, while others spread refusal across a low-rank subspace.

When moving up to **LLaMA-3.1-8B-Instruct**, refusal behavior snaps back to looking much more like Qwen than LLaMA-1B. A single refusal direction is again mostly sufficient to control behavior, though the details are a bit more nuanced.

---

## LLaMA-3.1-8B-Instruct

The layer-wise cosine similarity heatmap for LLaMA-3.1-8B-Instruct is shown below.

<div align="center">
  <img src="images/llama_8b_heatmap.png" alt="Sequential RV cosine similarity heatmap" width="50%">
</div>

At a glance, this looks very similar to Qwen. A large block of highly similar refusal vectors appears in the later part of the network, indicating that refusal representations converge and remain stable across many layers. This is very different from LLaMA-1B, where similarity drops off quickly away from the diagonal.

However, when looking at *behavioral impact*, the strongest refusal vectors do **not** come from the very last layers. Instead, the most effective vectors show up earlier:

| Layer | Mean compliance | Mean coherence |
| ----: | --------------: | -------------: |
|     9 |           0.795 |          0.965 |
|    11 |           0.570 |          0.970 |
|    10 |           0.335 |          0.950 |
|    12 |           0.300 |          0.940 |
|     8 |           0.100 |          0.950 |

Despite the model having 32 layers, the peak leverage sits around layers 8-11. This is earlier than the “mid-late” region observed in Qwen and earlier than I initially expected based on the smaller LLaMA models.

My personal interpretation of the heatmap is that because layers roughly 15 and onward show extremely high cosine similarity with one another, the refusal direction has likely already been decided earlier and is simply being reused or lightly refined at this point. These later layers encode a *stable* refusal representation, but intervening on them has relatively weak causal impact. By contrast, layers near 9 appear to be where refusal is first decisively introduced.

Put differently: cosine similarity here reflects *stability*, not *decision power*. LLaMA-8B seems to decide on refusal early, then propagate that decision forward through many nearly identical layers. That appears to be why when model's has roughly a single refusal direction the cross-layer cosine similarity is high. The overall direction has been decided and later layers are merely refining it stylisically. For those that lives on a low rank subspace, there isn't a single, well defined direction and hence the heatmaps of those models usually wouldn't have a high cosine similarity for the late layers. 

---

## Interpreting LLaMA-1B vs LLaMA-8B

This contrast sharpens what’s going on in LLaMA-1B. In that model, refusal vectors never really collapse to a single stable direction. Each layer contributes a slightly different refusal-aligned component, producing a low-rank structure rather than a single axis. No single layer dominates, and refusal remains distributed across the network.

LLaMA-8B, by comparison, behaves much more like Qwen:

* refusal becomes well-defined early
* a single direction is largely sufficient for control
* later layers mostly propagate an already-decided signal

One plausible explanation is that LLaMA-1B’s distributed refusal is a consequence of training choices or architectural scale effects, rather than something fundamental about the LLaMA family. Supporting this, LLaMA-3.2-3B-Instruct behaves more like the 1B model than the 8B model, with peak compliance stuck around ~15% under single-direction ablation.

---

## Cross-model comparison

To put these results in context, I tested a small collection of other instruction-tuned models using the same basic methodology. Due to limited compute, all of these models are under ~10B parameters; extracting refusal vectors and generating large numbers of rollouts per layer gets expensive quickly. The table below summarizes whether a single refusal vector was sufficient, along with peak compliance achieved under runtime ablation.

| Model                  | Single RV sufficient? | Peak compliance | Notes                  |
| ---------------------- | --------------------- | --------------- | ---------------------- |
| Qwen3-1.7B             | Yes                   | ~96%            | Very strong single dir |
| Qwen-1.8B              | Yes                   | ~90%            | Clean baseline         |
| gemma-2b-it            | Yes                   | ~90%            | Matches AF post        |
| LLaMA-3.1-8B-Instruct  | Yes                   | ~80%            | Early decision, stable |
| phi-3-mini-4k-instruct | Partial               | ~39%            | Mixed behavior         |
| LLaMA-3.2-1B-Instruct  | No                    | ~21%            | Low-rank subspace      |
| LLaMA-3.2-3B-Instruct  | No                    | ~15%            | Similar to 1B          |

A few patterns stand out:

* Several models (Qwen, Gemma, LLaMA-8B) admit a clean single-direction refusal representation.
* LLaMA 1B and 3B variants behave differently, with refusal spread across a low-dimensional subspace.
* Phi lands in between, where single-direction ablation helps but clearly does not saturate compliance.


Overall, refusal structure is not universal across instruction-tuned models. Some models make an early, fairly discrete refusal decision that is then propagated forward, making them highly vulnerable to single-direction interventions. Others distribute refusal across layers, which makes it harder to remove cleanly and leaves deeper safety constraints intact even after subspace-level ablation.

Because these experiments focus on relatively small models (all under ~10B parameters), it’s still unclear how well this picture carries over to much larger systems. That said, the consistency across several families hints that scale and training choices both matter. Understanding *when* refusal collapses to a single direction, and *why*, feels like a key open question going forward.
