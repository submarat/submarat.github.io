---
layout: post
title: "Scaling up a DiT: replicating Peebles & Xie, then optimizing the training loop"
date: 2026-07-23
excerpt: "Went from a toy 2D spiral DDPM to a 130M-parameter class-conditional Diffusion Transformer on full ImageNet-1k, found an 8.3x training speedup with TF32 + bf16 + torch.compile, and completed the paper's full 400K-step schedule in ~24.6 hours instead of 4 days."
reading_time_minutes: 7
---

Following up on the [2D spiral diffusion post](/reverse-diffusion-on-a-2d-spiral/): the natural next step was scaling the same ideas up to a real architecture on real images — Peebles & Xie's [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (DiT), the paper that popularized replacing a diffusion U-Net with a plain transformer. Built in conversation with Claude.

## The architecture, faithfully

Rather than reuse a text-conditioned DiT variant I'd been building for an earlier (separate) text-to-image experiment, this is class-conditional, matching the paper directly:

- **Fixed 2D sin-cos positional embeddings** (ViT/MAE-style), not RoPE
- **Pure AdaLN-Zero conditioning** on `timestep embedding + class label embedding` — no cross-attention, no joint text/image token sequence, because there's no text at all
- **GELU MLP** blocks, standard ViT feedforward
- **DiT-B/2 config**: hidden dim 768, depth 12, 12 heads, patch size 2 → **130.7M parameters**, matching the paper's DiT-B exactly
- **EMA of the weights** (decay 0.9999), horizontal-flip augmentation, and 10% classifier-free-guidance label dropout to a null class — all part of the paper's actual training recipe

Data: the full ImageNet-1k train split (1,281,167 images, 1000 classes), latents pre-encoded through the standard Stable Diffusion VAE (8x spatial compression, 256px → 32x32x4 latents) so the diffusion model itself never touches raw pixels.

## First 50K steps

I started with a partial run — 50,000 of the paper's 400,000-step schedule (global batch 256, AdamW, constant lr 1e-4, no weight decay) — to sanity-check the whole pipeline before committing to the full multi-day schedule. One thing worth flagging for anyone replicating this: **the loss curve is a bad proxy for progress here.** It drops from ~0.56 to ~0.30 in the first thousand steps and then looks essentially flat for the rest of the run, while the actual generated samples keep visibly improving the whole time. A related gotcha: with only 50K steps and the paper's EMA decay of 0.9999, the EMA weights are still less than 50% "caught up" to the live model at that point (`1 - 0.9999^7000 ≈ 50%`) — so if you're checking progress by sampling from the EMA copy mid-run, it'll look far worse than reality. Sample from the raw weights while training is still short of the full schedule.

At 50K steps, guided sampling (classifier-free guidance) already produces genuine class-conditional structure for several classes — recognizable stripe patterns for tiger and zebra, a round shape with visible topping-like dots for pizza — while others (golden retriever, consistently, across every checkpoint I looked at) are still an undifferentiated blob:

![DiT-B/2 samples at 50K steps, raw vs EMA weights](/images/dit-imagenet/step50000_labeled.png)

Raw and EMA weights are complementary at this point rather than one being strictly better — EMA nails zebra's stripes more cleanly, raw wins on tiger.

## An 8.3x speedup, benchmarked in isolation

50K steps took ~12.7 hours unoptimized — the full 400K-step schedule the paper actually uses for its comparison table would have been close to 4 days on one GPU. Before committing to that, I benchmarked each optimization separately on the same training step, same methodology as an [earlier optimization pass on GPT-2](/reproducing-gpt2-small/):

| step | change | ms/step | img/s | vs. previous |
|---|---|---|---|---|
| baseline | fp32, no TF32 | 874.0 | 293 | — |
| + TF32 | `matmul_precision("high")` | 353.0 | 725 | **2.48x** |
| + bf16 | autocast | 194.6 | 1315 | **1.81x** |
| + fused AdamW | `fused=True` | 192.6 | 1329 | 1.01x |
| + `torch.compile` | | 105.1 | 2435 | **1.83x** |

**8.31x cumulative**, no FlashAttention-3 required — PyTorch's built-in `scaled_dot_product_attention` already gets a fused-kernel benefit once the activations are bf16, without installing anything extra. TF32 and bf16 were the two big individual wins (as they were for GPT-2), fused AdamW was noise, and `torch.compile` — which I hadn't tried on the earlier GPT-2 pass — added another clean 1.83x on top from kernel fusion.

The practical effect: the full 400K-step schedule goes from **~4 days to ~11.7 hours.** That turns "technically possible but impractical" into an overnight run.

## Full 400K schedule, complete

Training continued from the 50K checkpoint through the remaining 350K steps with the optimized loop (~11.9 hours), on top of the original 50K unoptimized (~12.7 hours) — full paper-matching schedule done in about a day of wall-clock time instead of four. Same 8 preview classes, same guidance scale (1.5), raw weights on top and EMA on the bottom:

![DiT-B/2 samples at the full 400K steps, raw vs EMA weights](/images/dit-imagenet/final400k_labeled.png)

This is a clear step up from the 50K checkpoint, not just "still noisy but less so." Flamingo and zebra are now full recognizable *scenes* — flamingos standing in water with correct posture, two zebras in a grassy field with correctly-shaped stripes — not just texture in roughly the right colors. Golden retriever, the one class that was a flat, undifferentiated blob at every single checkpoint I sampled up through 50K, is now a clean, well-formed dog face in both raw and EMA weights. Tiger shows correct fur/stripe patterning. Cheeseburger and pizza are the weakest of the eight, but still clearly food-shaped with layered structure, not abstract color fields.

Consistent with the paper's own numbers: DiT-B/2 at 400K steps is documented in their scaling table at FID ≈ 40s (without classifier-free guidance) — a meaningfully smaller model than their DiT-XL/2 flagship (675M params, FID ≈ 19-20 at the same 400K steps, and the ~2.27 headline number after ~7M steps with tuned guidance). So "correct semantics, visible roughness" is the expected ceiling for this exact config and schedule, not a shortfall — and that's what the samples above show.

Total for this replication: 130.7M-param DiT-B/2, full ImageNet-1k (1.28M images, all 1000 classes), full 400K-step paper-matching schedule, in ~24.6 hours of wall-clock time on one H100.
