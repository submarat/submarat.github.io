---
layout: post
title: "Training an SAE on GPT-2, and steering it into gibberish on purpose"
date: 2026-09-22
excerpt: "With 30M residual-stream activations on disk, I trained a TopK sparse autoencoder on gpt2-xl's layer 7, read off what its features fire on, and then closed the loop the way interpretability is supposed to: I picked a feature by what it correlates with, clamped it on during generation, and watched the model produce exactly that. The feature I picked turned out to be the Unicode replacement character."
reading_time_minutes: 9
---

In the [last post](/collecting-activations-for-an-sae-roofline-and-a-writer) I built the boring-but-necessary plumbing: 30M residual-stream activations from gpt2-xl at `blocks.7.hook_resid_post`, packed into shards on disk. This post is the payoff — training a sparse autoencoder on those vectors and then doing the thing that makes interpretability feel real: finding a feature by what it *correlates* with, then *causally* forcing it and watching the model obey. The code for both stages is [on GitHub](https://github.com/submarat/activation_capture). As before, this was worked out with Claude alongside me.

## Why an SAE, and why TopK

The residual stream is dense and polysemantic: any single neuron participates in many unrelated computations, so you can't read a direction off it and say "this is the *X* feature." The sparse-autoencoder bet is that those activations are secretly a sparse combination of many more *interpretable* directions than there are dimensions — so you train an overcomplete autoencoder that reconstructs each activation using only a handful of active latents, and hope those latents are monosemantic.

I used a **TopK SAE** (the Gao et al. / OpenAI recipe): the encoder produces a pre-activation per latent, you keep only the top `k` and zero the rest, and the decoder reconstructs from those. The appeal is that there's no L1 sparsity coefficient to tune — sparsity is enforced *exactly* by `k`, and the loss is pure reconstruction MSE. For gpt2-xl's d_model of 1600 I used **16,384 features** (a 10× dictionary) and **k=32**, so every activation is explained by 32 of 16,384 possible directions.

Two implementation notes that matter. Normalize the inputs (I subtract the per-dimension mean and divide by a scalar scale ≈ 1.88 here) so the reconstruction loss is well-conditioned. And keep the decoder rows unit-norm after every step, so a feature's "activation" is a meaningful magnitude rather than something the optimizer can trade off against the decoder weight.

Training on the 30M activations for one epoch took a few minutes on an H100 and behaved itself:

![TopK SAE training: FVU and dead-feature fraction](/images/sae-activations/sae_training.png)

The metric to watch is **FVU — fraction of variance unexplained** — which is just reconstruction MSE divided by the variance of the data. It fell to **~0.07**, meaning the SAE reconstructs ~93% of the residual-stream variance from 32 active features per token. The other line is the **dead-feature fraction** — latents that stopped firing entirely — which stayed at **~4%**. Dead features are the classic SAE failure mode (the dictionary silently shrinks), so keeping it low without any resampling tricks is a good sign that k=32 and the learning rate were in a sane range.

## Reading the features

The real test isn't the loss — it's whether the latents *mean* anything. So I streamed fresh OpenWebText through gpt2-xl, encoded each token's residual with the SAE, and for a set of moderately-sparse features recorded the token contexts where they fired hardest. A sample of what layer 7 had learned:

| feature | fires on | top contexts |
|---|---|---|
| 2693 | the noun after "The" | The **activists**, The **Roman**, The **Obama** |
| 1598 | the noun after "An/A" | An **Idaho**, An **artist**, An **iPhone** |
| 10657 | the last item in a list | Wii, DSi and **Kinect**; North Korea and **Haiti** |
| 15353 | sentence onset after a paragraph break | `\n\n`While **the**, `\n\n`When **the** |
| 13867 | salient topical nouns | **solar**, **HIV**, **TCP**, **emergency** |
| 9326 | the `�` replacement character | It**�**, spectacular site,**�** |

This is roughly what you'd expect from a middle-early layer: a lot of it is *syntactic position* ("the head noun after a determiner", "the final element of a coordinated list", "the first word after a paragraph break") rather than abstract semantics. That's a feature, not a bug — it's the kind of structure layer 7 should be tracking, and it's legible. And tucked in among the grammar is feature **9326**, which fires specifically on `�`, the Unicode replacement character that shows up when text has a botched encoding. gpt2-xl has apparently allocated a dedicated direction to "this is mojibake."

## Closing the loop: steering

Max-activating contexts only show *correlation* — feature 9326 lights up *near* `�`. The stronger claim is causal: if I clamp that feature on, does the model *produce* `�`? This is the same move as Anthropic's Golden Gate Claude, just at gpt2-xl scale.

Mechanically it's a forward hook at layer 7 that adds `α · (scale · decoder_row[f])` to the residual stream during generation — the decoder row is the feature's direction in normalized space, and multiplying by the scale puts it back in real residual units. Same prompt, same seed, sweep α:

```
prompt: "My favorite thing about the weekend is"

[α = 0  baseline]  …that I don't have to be worried about my phone during the weekend.
[α = 10]           …that I don't have to be worried about my phone during the weekend.   (unchanged)
[α = 25]           …that I don't know what I would do with her. But I really feel like…   (drifting)
[α = 40]           …is����������������������������������
```

There it is. Below a threshold the steer does nothing; push it and coherence degrades; push it past ~40 and the model does nothing but emit `�`, forever. The feature that *correlates* with the replacement character *causes* the replacement character when you force it. Same experiment on the "final item in a list" feature (10657) instead collapses the model into looping short clauses — "We are here. and now. We are here." — a syntactic feature producing a syntactic pathology, which fits.

I'll admit I went in hoping for a tidy semantic feature — a "talk about San Francisco" latent — and at layer 7 the vocabulary is more grammatical than thematic, so the cleanest causal demo turned out to be an encoding artifact. That's honest and, I think, more interesting: the SAE surfaced a direction I'd never have gone looking for, and steering confirmed it does exactly one thing.

## What this bought

End to end: a 90 GB activation dataset, a TopK SAE that explains 93% of the residual variance with 32-of-16,384 sparsity and few dead features, a legible catalogue of what a middle-early layer is tracking, and a causal handle on one of those features. The pipeline from the last post made the front half fast and un-lose-able; the SAE made the back half interpretable.

The natural next steps are the interesting ones: run this at a later layer where features get more semantic, look for a genuinely conceptual feature to steer, and measure how much the SAE actually costs the model when you splice its reconstruction back into the forward pass (the "reconstruction fidelity → downstream loss" question). But finding a `�` neuron and being able to switch it on was a good enough place to stop for today.
