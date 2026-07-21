---
layout: post
title: "Reproducing GPT-2 Small: training, evaluating, and watching induction heads form"
date: 2026-07-21
excerpt: "I trained a 124M GPT-2 from scratch on 10B tokens of FineWeb-Edu on a single H100, benchmarked it against the real GPT-2, and traced how its capabilities — including an induction head — emerged over training. Notes on making the training practical, what the data choice did to the model, and where a from-scratch repro stops feeling like a demo."
reading_time_minutes: 15
---

A while ago I implemented GPT-2 from scratch — the attention, the MLP blocks, the layer norms — got the shapes right, watched a toy training loss go down, and loaded the real pretrained weights to generate text. It worked. But it never quite felt like a *reproduction*. The training loop only ever saw a few hundred steps of WikiText, and the coherent text came from OpenAI's weights, not mine. Two things were proven separately — "my loop can reduce a loss" and "my architecture can host GPT-2's weights" — but they never met in the middle: **weights that are GPT-2-quality because my code trained them.**

This post is about closing that gap. I trained a 124M-parameter GPT-2 from random initialization on 10B tokens of FineWeb-Edu on a single H100, evaluated it head-to-head against the public `gpt2`, and then traced *how* its capabilities appeared over training — ending with a genuine mechanistic-interpretability result: finding and watching an induction head form. The code is [on GitHub](https://github.com/submarat/gpt2-small-repro). As with my [post-training notes](/how-llms-go-from-base-models-to-assistants), a lot of this was worked out in conversation with Claude.

## Making training practical

A faithful GPT-2 run is ~10–300B tokens depending on which reproduction you target. The from-scratch implementation I started with was correct but slow — plain float32, hand-rolled einsum attention, no compilation. Before committing a GPU to it for hours, I did an optimization pass and, crucially, **benchmarked each step in isolation** so I could attribute the gains rather than cargo-cult them.

The measurement harness runs the model on synthetic fixed-shape batches and reports training-step throughput and peak memory. Applied cumulatively on an H100 at the full GPT-2-small config (batch 8, seq 1024):

| step | change | train tokens/sec | vs. previous |
|---|---|---|---|
| baseline | fp32 | 36,600 | — |
| + tf32 | TF32 matmuls | 66,600 | **1.82×** |
| + bf16 | bf16 autocast | 70,100 | 1.05× |
| + sdpa | fused/flash attention | 107,100 | **1.53×** |
| + compile | `torch.compile` | 163,900 | **1.53×** |
| + fused AdamW | fused optimizer | 184,800 | 1.13× |
| + weight tying | tie unembed to embed | 178,500 | ~1.0× |

End to end: **~4.9× faster training**, from 36.6K to 178.5K tokens/sec, and a 43% cut in peak memory. A few things surprised me:

- **TF32 was the single biggest win, and nearly free.** The fp32 baseline was doing matmuls off the tensor cores; one line — `torch.set_float32_matmul_precision("high")` — nearly doubled throughput. It keeps fp32's 8-bit exponent (full range) but truncates the mantissa to 10 bits for the multiply, accumulating in fp32. For training, that precision loss is irrelevant.
- **bf16 on top of TF32 barely moved throughput** — TF32 was already using the tensor cores. Its real value is unlocking the flash-attention kernel and halving activation memory.
- **SDPA and `torch.compile` were the two structural wins**, together roughly 2.3×, and they cut memory by not materializing the `[batch, heads, seq, seq]` attention-probability tensor.
- **Weight tying is a storage change, not a FLOPs change.** Tying the unembedding to the transposed embedding drops ~38M parameters (163M → 124.5M, matching real GPT-2) but the matmuls are identical, so throughput is flat. It's about parameter fidelity and freeing memory for a bigger batch, not speed.

At ~200K tokens/sec, 10B tokens is about 14 hours on one H100. That's an overnight run, which made the whole experiment feasible.

## The training run

I used the modern "speedrun" recipe rather than the faithful 300B-token one: **10B tokens of FineWeb-Edu**, the educational-quality-filtered CommonCrawl subset that current GPT-2 reproductions favor. The distinction matters and comes back later — FineWeb-Edu is *not* WebText (GPT-2's original data), and that difference is visible in the results.

Hyperparameters followed GPT-2 / nanoGPT: peak LR 6e-4 with a linear warmup then cosine decay to 6e-5, AdamW with β = (0.9, 0.95), weight decay 0.1 applied only to ≥2D tensors (not biases or LayerNorm gains), and gradient clipping at 1.0. The training objective is just next-token cross-entropy,

$$L = -\frac{1}{N}\sum_{i} \log P(x_{i+1} \mid x_1, \ldots, x_i)$$

To reach GPT-2's ~0.5M-token effective batch on one GPU, I used gradient accumulation: 32 sequences per micro-batch × 16 accumulation steps × 1024 tokens = 524,288 tokens per optimizer step. (I verified the accumulated gradient is bit-for-bit equivalent to a single large batch — scaling the loss by 1/N and accumulating matches to ~1e-9.)

The run took ~13 hours, held ~201K tokens/sec end-to-end (dataloader overhead was negligible — the loop is compute-bound), and drove training loss from 10.6 to 3.27. I checkpointed every 2000 steps, which — spoiler — turned out to be too coarse for one thing I wanted to see.

## Does it actually reproduce GPT-2?

To evaluate fairly, I converted my custom model into an equivalent HuggingFace `GPT2LMHeadModel` (verified numerically exact — max logit difference ~1e-6) so I could run the standard [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) against it and against the public `gpt2`. I picked six tasks where GPT-2 small scores clearly above chance (near-chance tasks like WinoGrande or MMLU are just noise at 124M and tell you nothing).

| Task | Metric | **Ours (10B FineWeb-Edu)** | GPT-2 (public) |
|---|---|---|---|
| arc_easy | acc_norm | **0.435** | 0.396 |
| sciq | acc_norm | **0.655** | 0.642 |
| hellaswag | acc_norm | 0.290 | **0.312** |
| piqa | acc_norm | 0.599 | **0.622** |
| lambada | acc | 0.188 | **0.309** |
| wikitext | word ppl (lower better) | 54.7 | **37.8** |

So: a GPT-2-class model that trades blows with the real thing. But the *pattern* of wins and losses is the interesting part — it's the **fingerprint of the training data**:

- **Wins on knowledge** (arc_easy, sciq) — educational filtering pays off directly on science/QA.
- **Near-parity on commonsense** (hellaswag, piqa within ~2 points).
- **Loses on the distribution-sensitive tasks** (lambada is narrative book text; wikitext is encyclopedic) — FineWeb-Edu's aggressive quality filtering *narrows* the distribution away from these, while GPT-2's broad WebText covers them well.

That's a more useful outcome than a single matching number. The data choice didn't just make the model "a bit better or worse" — it reshaped *which* capabilities it has.

## Watching capabilities emerge

Because I saved checkpoints throughout training, I could evaluate all ten of them and plot each metric against training tokens, with GPT-2 as a dashed reference:

![Capability emergence over training](/images/gpt2-capability-emergence.png)

Three distinct regimes show up:

1. **Knowledge (sciq, arc_easy)** rises fast and crosses *above* GPT-2 by ~4B tokens, then plateaus. The FineWeb-Edu advantage is banked early.
2. **Commonsense (hellaswag, piqa)** rises and then plateaus *just below* GPT-2. hellaswag is dead flat after ~2B tokens. The flatness is the point: **more FineWeb-Edu tokens would not close this gap.** It's a data-distribution ceiling, not a compute limit.
3. **lambada and wikitext are still improving at 10B.** These didn't saturate — more (and broader) data would keep helping.

So the honest answer to "was GPT-2 parity a ceiling or where the budget ran out?" is: *both, depending on the capability.* Commonsense hit a real ceiling from the data mix; knowledge was won outright; the distribution-sensitive tasks were simply under-trained. You can only see that decomposition by watching the trajectory, not the endpoint.

## The mechanistic capstone: an induction head

The part I most wanted to reach was mechanistic: could I find a concrete circuit inside a model *I* trained? The obvious target is an **induction head** — the attention head behind much of in-context learning, which implements "find the previous occurrence of the current token, and copy whatever came next" ([Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)).

The clean way to measure it is on a sequence of *random* tokens repeated twice, `[r₁…r_L][r₁…r_L]`. Random tokens matter: there's no bigram prior to exploit, so the only way to predict the second half is to genuinely look back. An induction head at position `p` in the second copy attends to position `p − L + 1` — the token that followed the previous occurrence of the current token, i.e. the correct next token. The induction score is the attention mass on that offset. I read the attention patterns directly out of my own model (running the non-fused attention path so it returns probabilities) via forward hooks — exact weights, no conversion.

Running this across all ten checkpoints:

![Induction head formation](/images/gpt2-induction-formation.png)

The result:

- **There's a clear induction head: layer 11, head 2**, with a score of ~0.80 (plus a couple of weaker ones in the same final layer). A real in-context-copying circuit, in a model trained from scratch on my own hardware.
- **It forms fast and saturates** — the score jumps from 0.37 to 0.75 between 1B and 2B tokens, then plateaus around 0.80.
- **But I missed the famous phase-change "bump."** In-context-learning score is already high at my first checkpoint (1B tokens) and drifts down afterward, rather than showing the sharp rise the literature describes.

That last point taught me the most. Induction heads form as an abrupt phase change — you can often see a kink in the loss curve where they snap into place — and at GPT-2-small scale that happens in roughly the **first ~1B tokens**, *before* my earliest checkpoint. By the time I first looked, the head had already largely formed. Two views of the same event line up nicely: the commonsense capabilities plateaued around ~2B tokens, and the induction head saturated in the same window — unsurprising, since induction underlies a lot of in-context pattern completion.

A striking fact from the literature is that this phase change occurs at a *similar absolute token count across model sizes* — it's driven more by data and optimization than by model capacity. So to actually watch the bump, I wouldn't need a bigger model; I'd need to checkpoint much more densely (every ~50–100 steps) over that first billion tokens. That's the clean next experiment, and it's cheap.

## What I took away

- **A from-scratch reproduction stops feeling like a demo the moment the capable weights are yours.** Generating fluent (if 124M-shaky) text from weights that were random noise 13 hours earlier is a categorically different feeling than loading someone else's checkpoint.
- **Benchmark each optimization.** I would have guessed bf16 was the big win; it was TF32. Measuring each step turned "make it faster" into a set of attributable facts.
- **The training data is a design decision with a visible signature.** FineWeb-Edu didn't make a uniformly better or worse GPT-2 — it made a *differently-shaped* one, strong on knowledge and weak on narrative, and the emergence curves showed exactly where each ceiling came from.
- **Checkpoints turn a training run into an experiment.** The capability-emergence and induction analyses were only possible because I saved intermediate weights — and the one thing I couldn't resolve (the induction bump) was purely a checkpoint-granularity problem, not a modeling one.

Everything — the optimized training loop, the eval harness, and the analysis scripts — is [in the repo](https://github.com/submarat/gpt2-small-repro).

## References

<a name="ref-1"></a>[1] Radford et al. *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.

<a name="ref-2"></a>[2] Karpathy. *nanoGPT* and *llm.c* — GPT-2 reproduction recipes.

<a name="ref-3"></a>[3] Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, 2024.

<a name="ref-4"></a>[4] Olsson et al. *In-context Learning and Induction Heads*, Anthropic, 2022.
