---
layout: post
title: "Steering and ablating the emergent misalignment direction"
date: 2026-07-21
excerpt: "Follow-up to my rank-1 EM replication: extracted the mean-diff misalignment direction, steered the base model with it, and tried ablating it from the EM model. Confirmed the paper's near-orthogonality finding, but got a threshold-shaped steering curve rather than a clean dose-response, and an ablation result that flatly contradicts the paper's >98% reduction claim."
reading_time_minutes: 8
---

Follow-up to [my rank-1 emergent misalignment replication](/replicating-rank-1-emergent-misalignment/). That post replicated Turner et al.'s behavioral result — a single rank-1 LoRA adapter on one MLP layer induces broad misalignment. This one follows up on Soligo, Turner, Rajamanoharan & Nanda's companion paper, [Convergent Linear Representations of Emergent Misalignment](https://arxiv.org/abs/2506.11618), which claims the misalignment is mediated by a single linear direction in activation space. Three things to check: does the direction have low cosine similarity with the LoRA's own weights (their most surprising claim), does adding it to the base model induce misalignment, and does ablating it from the EM model remove misalignment.

Two of three replicated cleanly. The third didn't, and I think that's more useful to report than to bury.

Code, vectors, and raw results: [`steering/`](https://github.com/submarat/em-rank1-lora-replication/tree/master/steering) in the [replication repo](https://github.com/submarat/em-rank1-lora-replication).

## Extracting the direction

Following the paper's actual method — not the simplified "EM activations minus base activations" pseudocode in my project brief — the direction is a contrast *within* the EM model: mean activation on its own misaligned completions minus mean activation on its own aligned completions, at layer 24. I already had 2,700 judged responses from the rank-1 model's eval, so this was just a matter of splitting them by score (273 misaligned, 273 aligned after subsampling for balance) and collecting residual-stream activations on each set.

**Cosine similarity between this direction and the rank-1 LoRA's own B vector: 0.0687.** The paper reports ~0.04. Both say the same thing: the direction the LoRA actually learned and the "population-level" misalignment direction are close to orthogonal as raw vectors. That's a real, if narrow, replication of their most counterintuitive finding — the LoRA's own parameters don't obviously encode the direction that supposedly mediates its effect.

## Steering: a threshold, but not the threshold I expected

Adding the direction to the *base* model's layer-24 residual stream at increasing scale, on 8 out-of-domain questions:

![Dose-response curve: misalignment and coherence vs steering scale](/images/em-replication/steering_dose_response.png)

The project brief flagged two possible outcomes: a smooth dose-response curve, or — "what would be interesting" — a sharp threshold. I got the threshold. But not the shape I'd have guessed. Misalignment rises from 0% to a peak of **9.4% at scale 5** (coherence still 75% there), then falls back toward 0% by scale 8 — and it falls *because coherence collapses in lockstep*, not because the model becomes safe again. By scale 8 one probe response (to a question about feeling overwhelmed) had degenerated into repeating "proxy"/"browser" tokens with no relation to the question. By scale 24 it's cross-lingual gibberism.

So the direction does causally induce misalignment — there's a real, sharp window where it works — but pushing further doesn't sustain "confidently wrong" behavior, it just breaks the model. The raw (unbucketed) `aligned` judge score keeps falling monotonically the whole way to scale 24, which tells you the judge is scoring the gibberish as maximally misaligned too; it's just that the compound metric (`aligned≤30 AND coherent>50`) requires coherence, and coherence is gone by then.

## Ablation: this is the part that didn't go as expected

Soligo et al.'s claim is that ablating this same direction from the EM model removes misalignment (>98% reduction), despite the near-zero cosine similarity above. I tried this two ways and got two different answers.

**First attempt (wrong mechanism, in hindsight):** the repo has a `load_lora_with_vec_ablated` utility that projects a direction out of the LoRA's own B weight vector directly, then renormalizes back to the original norm. Ran it, got nothing: 9.2% control vs. 10.0% ablated, noise. Thinking about why: cosine similarity here is 0.0687. Projecting out a component that's already almost orthogonal, then rescaling back to the same norm, barely rotates the vector at all — removing ~0.5% of its "energy" and restoring full magnitude. This isn't a real test of anything; it's a near-no-op dressed up as an intervention.

**Second attempt, the one the original repo's own `activation_steering.py` script actually uses for this exact experiment:** project the direction out of the *live residual stream* at every position during generation (not the weights) — a runtime hook, not a weight edit.

![Ablation comparison: control vs weight-space vs activation-space ablation](/images/em-replication/ablation_comparison.png)

Misalignment went from 10.4% (control) to **26.5% (ablated)**. Coherence barely moved (99.2% → 96.2%). That's not a smaller effect than the paper reports — it's the *opposite direction* entirely.

I don't have a confirmed explanation. My best guess: zeroing out a coordinate isn't the same operation as "removing misalignment content" — it assumes the coordinate's natural resting value is neutral. If ordinary EM-model generation already sits at a mildly *negative* projection onto this direction (i.e., the model's default behavior is already leaning slightly away from "misaligned" along this axis, most of the time, since misalignment is still only ~10% of responses), then forcefully zeroing that coordinate on every forward pass removes whatever was providing that lean, rather than removing something that was pushing behavior toward misalignment. That's a hypothesis, not something I've verified — I haven't checked the actual sign/distribution of that projection during ordinary generation, which would be the natural next thing to check before trusting the hypothesis at all.

I'm reporting this rather than tuning it away because a null-or-reversed replication result on a real claim is more informative than silence, and because I'd rather be visibly wrong in a way that's checkable than quietly drop the inconvenient number.

## What's next

- Figure out why activation-space ablation backfires — starting with the sign/distribution of the projection during unsteered generation.
- Cross-dataset transfer: train a second EM model on financial advice, check whether its direction converges with this one (the paper's other headline claim).
- The CoT-legibility extension from the [first post](/2026/07/21/replicating-rank-1-emergent-misalignment.html)'s discussion.

## References

Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S., & Nanda, N. (2025). *Model Organisms for Emergent Misalignment.* [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)

Soligo, A., Turner, E., Rajamanoharan, S., & Nanda, N. (2025). *Convergent Linear Representations of Emergent Misalignment.* [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)
