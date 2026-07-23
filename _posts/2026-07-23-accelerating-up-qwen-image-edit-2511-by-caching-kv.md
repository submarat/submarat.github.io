---
layout: post
title: "Accelerating up Qwen-Image-Edit 2511 by caching KV"
date: 2026-07-23
excerpt: "Qwen-Image-Edit-2511 feeds a reference-image's VAE-encoded tokens through all 60 joint-attention blocks at every denoising step, discarding their own output every time. A per-block cosine-similarity study shows they're far more stable than they need to be recomputed from scratch — and a KV cache built on that finding gets a validated 1.3-1.7x speedup on 100 real edit tasks."
reading_time_minutes: 12
---

Qwen-Image-Edit-2511 edits an image by concatenating the input image's VAE-encoded latent
tokens onto the noisy target latent, and running the concatenation through 60 joint
self-attention blocks, once per denoising step. The reference tokens' own output gets
thrown away every single step — only the target (noisy) tokens get decoded into the final
image. That's a strong hint that the reference tokens' contribution to attention might be
far more stable across steps than the target's, and that recomputing it from scratch 60
times over might be mostly wasted work. This post is the study that checks that hunch, the
cache that exploits it, and the eval that says whether it's actually worth it. Code's
[on GitHub](https://github.com/submarat/qwen-image-edit-kv-cache); built in conversation
with Claude.

## The architecture, in one diagram

Qwen-Image-Edit-2511 is a dual-stream MMDiT: separate weights per block for the image and
text streams, but **joint** self-attention — everything attends to everything in one
`scaled_dot_product_attention` call. The image stream itself is a concatenation of two
different kinds of tokens sharing the same `img_in` projection and the same 60 blocks:

- **target tokens** — the noisy latent actually being denoised into the output image.
- **reference tokens** — the VAE-encoded input image, there purely as conditioning.

The pipeline slices the transformer's output back down to just the target portion
(`noise_pred = noise_pred[:, :latents.size(1)]`) before every scheduler step. The reference
tokens' own output row is computed and immediately discarded, every block, every step.

![Per-block KV cache: refresh step recomputes everything, cached step reuses reference K/V and skips the reference query row entirely](/images/qwen-kv-cache/architecture_diagram.png)

That's the mechanism this whole post exploits: if a "cached" step can reuse the reference
tokens' Key/Value from a prior step instead of recomputing them, it can also skip computing
the reference **query** row entirely — its output was never going to be used anyway. The
saving isn't just "skip a linear projection"; it shrinks the actual attention computation
(shorter query dimension) and skips the reference tokens' own MLP/residual update on that
step.

The open question is whether that reuse is actually *safe* — i.e., how much does the
reference tokens' Q/K/V actually drift, step to step, at a given block?

## Study: how stable are the reference tokens, really?

Mirroring an earlier internal methodology (a similar KV-cache-viability study on a
different image-editing DiT), I hooked `attn.to_q` / `attn.to_k` / `attn.to_v` on all 60
`QwenImageTransformerBlock`s, ran 5 real images through the full 50-step schedule
(`cfg=1.0`, so exactly one forward pass per step), and for each block computed cosine
similarity between the reference tokens' Q/K/V at step *i* and step *i+gap*, for every valid
gap from 1 to 49.

![Reference-token Q/K/V cosine similarity vs. gap, colored by block depth — block 0 stays at 1.0 forever, a volatile band around blocks 30-50, stable again toward the end](/images/qwen-kv-cache/qwen_qkv_cosine_by_layer.png)

The finding surprised me: **block 0 is provably frozen** — cosine similarity of exactly
1.0 at every single gap, for every image, for Q, K, and V. That's not noise; it falls out of
the architecture directly. Qwen-Image-Edit uses a `zero_cond_t` mechanism: the reference
tokens always get AdaLN modulation from a fixed zero-timestep embedding (never the real,
per-step timestep), and at block 0 specifically there's been zero prior cross-attention with
the (evolving) target tokens yet. Both the input and the modulation applied to it are
step-invariant, so the output is too.

That stability doesn't last, though. As you go deeper, each block's joint attention lets a
little bit of target-token information leak into the reference tokens' residual stream, and
it compounds:

![Block depth x refresh gap heatmap for Q, K, V — a clear low-similarity trough around blocks 30-50, especially severe for V](/images/qwen-kv-cache/qwen_qkv_cosine_heatmap.png)

The heatmap makes the shape obvious: a **U-shaped stability curve** across depth, not the
monotonic or uniform decay I'd expected. Blocks ~1-25 and ~50-59 stay highly similar even at
the largest gap tested (0.86-0.96 for Q/K, 0.91+ for V), while blocks ~26-49 are a genuinely
volatile middle band — down to 0.63-0.70 for Q/K and as low as **0.34** for V at gap=49.

That non-uniformity is the whole design constraint for what comes next: a single global
cache-refresh-cadence (cache every block the same way) is the wrong shape for this model.
It would either refresh often enough to protect the volatile middle band and waste most of
the savings available in the ~45 stable blocks, or refresh rarely enough to get real
speedup and eat real error specifically in that middle band.

## Implementation: a per-block cache

`qwen_kv_cache.py` monkeypatches each block's forward pass with a schedule-driven choice
between two paths:

- **Refresh step**: call the block's real, unmodified forward (bit-exact), and separately —
  redundantly, but safely — recompute just the reference tokens' post-RoPE K/V to stash for
  later.
- **Cached step**: run `to_q/to_k/to_v` and the FFN on the target tokens only, reuse the
  stashed reference K/V for the joint attention's key/value, and skip the reference query
  row and its residual update entirely (the reference hidden state stays frozen until the
  next refresh).

The refresh schedule is per-block, not global — an array of `refresh_every[block_id]`
values, so the frozen/stable bands (block 0, 1-25, 50-59) can go many steps between
refreshes while the volatile middle band (26-49) refreshes every step.

Getting this correct took two real bugs, both caught by insisting on bit-exact validation
*before* trusting any speedup number:

1. `QwenImageTransformerBlock.forward` returns `(encoder_hidden_states, hidden_states)` —
   text stream first. I had the tuple backwards.
2. The joint attention processor applies an output projection
   (`attn.to_out[0]`, `attn.to_add_out`) *after* the attention and *before* the residual add.
   I'd missed it entirely — the cached path produced pure noise even when fed the
   *correct*, non-stale reference K/V, because the output was never projected back into the
   right basis.

The gate that catches regressions like this: force every block to refresh every step
(`RefreshSchedule.uniform(60, 1)`) and diff against the true unpatched model. That
configuration is **100% pixel-identical** to baseline — only once that held did I trust any
result from an actual caching schedule.

## Eval: does it actually pay off?

Ran 100 real (image, instruction) pairs sampled from
[MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush) — its own held-out
human-verified eval split plus a random top-up — through four conditions at 28 steps: a
true no-cache baseline, two CM5-style global schedules (`uniform_5`: refresh every 5 steps;
`uniform_10`: every 10), and the depth-aware `adaptive` schedule the study above motivated
(block 0 cached forever, blocks 1-25/50-59 refresh every 10 steps, blocks 26-49 never
cached).

![Speedup vs. PSNR scatter across 100 real edit tasks, one color per scheme, X marks the mean](/images/qwen-kv-cache/speedup_vs_psnr_100.png)

![PSNR distribution boxplot: adaptive has a visibly tighter interquartile range than uniform_5, uniform_10 is worst on both axes](/images/qwen-kv-cache/psnr_boxplot_100.png)

| scheme | speedup | avg PSNR | median PSNR | worst-case tail (p10) | hit rate |
|---|---|---|---|---|---|
| uniform, refresh=5 | 1.59x | 33.9 dB | 33.7 dB | 22.6 dB | 80% |
| uniform, refresh=10 | 1.73x | 29.3 dB | 28.4 dB | 21.5 dB | 89% |
| adaptive per-band | 1.34x | 33.1 dB | 33.2 dB | **23.2 dB** | 54% |

`uniform_10` is dominated outright — fastest, but worst on quality by every measure.
Between `uniform_5` and `adaptive`, the interesting result is that they land at nearly the
same *average* PSNR — a smaller pilot on 5 hand-picked images had suggested adaptive would
be a clean winner on quality at every speed point, and that gap mostly closed at 100-image
scale on a more realistic, diverse task mix. What survives is the **distribution**:
adaptive's PSNR spread is visibly tighter (boxplot IQR ~11.4 dB vs. ~13.9 dB) and its
worst-case tail is a little better — it's the more predictable choice, at the cost of being
~19% slower.

On typical edits, both the cache and the baseline land in essentially the same place:

<div style="display:flex; gap:8px;">
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/blog_fish_input.png" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">input</figcaption></figure>
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/blog_fish_baseline.png" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">no cache</figcaption></figure>
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/blog_fish_adaptive.png" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">adaptive cache</figcaption></figure>
</div>
<p style="font-size:0.85em; color:#666; margin-top:4px;">"Change the baseball bat to a big fish" — nearly indistinguishable.</p>

<div style="display:flex; gap:8px; margin-top:20px;">
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/blog_flower_input.png" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">input</figcaption></figure>
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/blog_flower_baseline.png" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">no cache</figcaption></figure>
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/blog_flower_adaptive.png" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">adaptive cache</figcaption></figure>
</div>
<p style="font-size:0.85em; color:#666; margin-top:4px;">"Let's add a drawing of a flower to the fridge" — again, essentially the same result.</p>

But the single worst PSNR score in the whole 100-image run (16.4 dB) is worth looking at
directly, because it's a useful warning about the metric itself:

<div style="display:flex; gap:8px;">
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/cake_input.jpg" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">input</figcaption></figure>
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/cake_baseline.jpg" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">no cache</figcaption></figure>
<figure style="flex:1; margin:0;"><img src="/images/qwen-kv-cache/cake_cached.jpg" style="width:100%"><figcaption style="font-size:0.85em; text-align:center;">cached (uniform, refresh=5)</figcaption></figure>
</div>
<p style="font-size:0.85em; color:#666; margin-top:4px;">"Let a woman in a bridal gown stand near the cake" — PSNR says this is the worst result in the whole run. Look at it: the cached version is arguably <em>better</em> than the baseline, which has a visible ghosting/transparency artifact on the inserted figure.</p>

For an instruction this generative — inserting a whole new subject into the scene — the
model has a wide space of equally-valid outputs, and caching just nudges which one you land
on. Low PSNR-vs-baseline mostly reflects "different point in the output distribution," not
"quality collapsed." It's a fine *relative* signal for comparing caching schemes against
each other, but not a substitute for actually looking at the images, especially for
content-adding edits rather than style/color changes.

## Where this leaves things

- The reference-token KV cache is a real, "free" speedup — no retraining, and the only
  error source is the model's own token drift, not anything introduced on purpose.
- The depth-aware schedule is worth the extra complexity if predictability matters more
  than raw throughput; a blind uniform schedule is worth it if the reverse is true.
- This is a research prototype, monkeypatched onto a standalone `diffusers` install, tested
  at one resolution regime on 100 images on a single H100. Sensitivity to resolution/aspect
  ratio (which mattered a lot for the equivalent CM5 study) is still open.

Everything — the study script, the cache, both eval harnesses, the benchmark builder, and
the raw results — is [on GitHub](https://github.com/submarat/qwen-image-edit-kv-cache).
