---
layout: post
title: "How LLMs go from base models to assistants"
date: 2026-07-23
excerpt: "How language models are turned from raw next-token predictors into assistants — the three post-training stages (SFT, preference optimization, specialized/chat fine-tuning) — grounded in a worked example: I ran the whole pipeline on a 124M GPT-2 I trained from scratch. Includes real base→SFT→DPO→chat interactions and an in-browser demo you can try."
reading_time_minutes: 16
---

Notes I've accumulated while learning how language models are transformed from raw next-token predictors into useful assistants — now grounded in a **worked example**. I [trained a 124M GPT-2 from scratch](/reproducing-gpt2-small) on 10B tokens of FineWeb-Edu, then ran the canonical post-training pipeline on it: supervised fine-tuning, DPO, and a multi-turn chat fine-tune. It's a *toy* at 124M — shallow and it hallucinates — but every stage's effect is visible and instructive, which is the point.

**[Try it in your browser →](https://huggingface.co/spaces/submarat/gpt2-fineweb-chat)** — the base, SFT, DPO, and chat models all run client-side (transformers.js), so you can watch the pipeline's effect first-hand. As always, worked out in conversation with Claude.

## The post-training pipeline

A pretrained language model is essentially a document completer. It can produce fluent text, but it doesn't know how to follow instructions, refuse harmful requests, or call external tools. Post-training bridges this gap; the three-stage pipeline was formalized by the InstructGPT work [[1]](#ref-1):

1. **Supervised fine-tuning (SFT)** teaches the model to follow instructions
2. **Preference optimization** (RLHF or DPO) aligns the model's outputs with human judgments about quality
3. **Specialized fine-tuning** for tool calling, multi-turn chat, or domain-specific behavior

Each stage builds on the last. SFT teaches the format and style of helpful responses; preference optimization teaches the model to distinguish good responses from bad within that format.

## Supervised fine-tuning: same objective, different data

SFT uses the exact same training objective as pretraining — next-token prediction with cross-entropy loss. For a target sequence with tokens (y₁, y₂, ..., yₙ):

$$L = -\sum_{i=1}^{n} \log P(y_i \mid x, y_1, \ldots, y_{i-1})$$

The key difference from pretraining is the data: instead of broad internet text, SFT uses curated instruction-response pairs that demonstrate the behavior you want. The model learns "given this kind of instruction, produce this kind of response" through the same maximum likelihood mechanism it used to learn language in the first place.

### Prompt masking matters

One important practical detail: during SFT you mask the loss on the instruction tokens and only compute loss on the response tokens:

*Masking instruction tokens out of the loss*
```python
input_ids = [instruction_tokens + response_tokens]
labels = [-100] * len(instruction_tokens) + response_tokens
loss = cross_entropy(logits, labels, ignore_index=-100)
```

Without this masking, the model wastes capacity learning to reproduce the prompt rather than generate good responses. It can still attend to the instruction tokens during the forward pass — they're only masked in the loss, not in attention. (In the worked example I used TRL, where this is a one-flag switch: `completion_only_loss=True` on a prompt/completion dataset.)

### Cross-entropy and KL divergence: a subtle point

Something that tripped me up: SFT's cross-entropy loss against a one-hot target is mathematically equivalent to minimizing KL divergence from that one-hot distribution to the model's output distribution. Since the entropy of a one-hot distribution is zero, D_KL(q ∥ p) = H(q, p) − H(q) = H(q, p). So cross-entropy and KL divergence collapse to the same thing.

But when people say "KL divergence" in the fine-tuning literature, they usually mean comparing two *soft* distributions — a teacher's logits against a student's (distillation), or a trained policy against a reference policy (DPO/RLHF regularization). The one-hot case is degenerate: you're maximizing likelihood of specific tokens, not matching a full distribution.

### What SFT actually did

Theory in hand, here's the effect on the real model. I fine-tuned the 124M base on [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (~52k instruction-response pairs), completion-only loss, three epochs (~8 minutes on one H100). Instruction-following simply *appears*:

| Prompt | Base (document completer) | After SFT (instruction follower) |
|---|---|---|
| "What is the capital of France?" | drifts into a history-lesson-plan template | **"Paris"** |
| "List three tips for staying focused while studying." | vague musing about reading | a numbered 3-item list |

It still hallucinates — asked for a motivational quote, it invented a plausible-looking attribution ("Paulo Freirer, 2006") — but it's unmistakably following instructions now. The lesson: format and instruction-following are cheap to teach; factual reliability is not.

### Practical training considerations

A few things that matter more than you'd expect:

**Batching by sequence length** speeds up training versus shuffling wildly different lengths, since short sequences waste compute on padding. Group by length and shuffle within groups.

**Monitoring validation loss** is critical — instruction datasets are small relative to pretraining corpora, so overfitting is a real concern. Training loss hitting zero is a memorization warning.

**Context-length mismatch** is underappreciated: fine-tune a 32k-context model on only 1k-token examples and it can lose long-context ability as high positions go undertrained. The fix is packing short examples into longer sequences.

## From SFT to preference optimization

SFT gets you a model that produces responses in the right format, but it doesn't learn to distinguish mediocre from excellent responses to the same prompt. That's preference optimization.

### RLHF: the original approach

The classic RLHF pipeline: (1) start with an SFT model, (2) train a reward model on human preference pairs, (3) use PPO [[2]](#ref-2) to optimize the LM against the reward model, with a KL penalty to prevent drift from the SFT model. PPO's strengths are flexibility (any differentiable reward) and online learning; its downsides are cost, instability, and the reward model as an imperfect bottleneck.

### DPO: cutting out the middleman

DPO [[3]](#ref-3) reformulates preference learning as classification, eliminating the reward model and RL loop. Given a prompt and two responses (preferred, dispreferred), its loss encourages higher probability on the preferred response relative to a reference model, via the Bradley-Terry model and a log-probability ratio scaled by β (which controls deviation from the reference). In practice it's simpler, cheaper, and more stable — the default for most alignment work today.

### What DPO actually did (and the alignment tax, in miniature)

I ran DPO on top of the SFT model using [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) (15k pairs). The training metrics did exactly what they should — the reward margin rose and the model preferred "chosen" ~63% of the time — and the *style* shifted toward longer, more elaborate, more "assistant-like" responses (the well-documented UltraFeedback length/verbosity preference).

But here's the instructive part: **at 124M, verbosity isn't quality — it's more room to hallucinate.** On some prompts DPO *regressed*:

| Prompt | After SFT | After DPO |
|---|---|---|
| "What is the capital of France?" | **"Paris"** ✓ | a long, confident paragraph about Belgium, Luxembourg, and Louis XIV ✗ |

The SFT model gave the right one-word answer; the DPO model, optimized to *sound* more helpful, talked itself out of it. That's the **alignment tax / reward-vs-truth gap** — optimizing a preference proxy is not the same as improving correctness, especially when the base model lacks the capacity to back up the extra words. It's also a direct, if small, data point on two of the open questions I list below.

### When PPO still wins

PPO keeps advantages when optimizing multiple objectives simultaneously (helpful AND harmless AND honest), when the reward isn't pairwise, or when optimal behavior is far from the reference. The extra complexity buys flexibility.

## Specialized fine-tuning: multi-turn chat

The single-turn SFT above (Alpaca is one instruction → one response) isn't a chatbot — it has no notion of a conversation. Stage 3 is where you teach conversational and tool-use behavior. For chat, that means: a **conversational dataset**, a **chat template** that encodes role-tagged turns, and **loss on the assistant turns** across the whole dialogue.

I ran a separate SFT pass from the base model on [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (multi-turn, built for small models), with a ChatML-style template (`<|user|>` / `<|assistant|>`, each turn ending in EOS) and `assistant_only_loss`. The result holds a running conversation and references earlier turns:

> **User:** Give me three tips for studying.
> **Assistant:** *When you read your text, be sure to carefully listen with attention and focus on the words that are important…*
> **User:** Can you expand on the first one?
> **Assistant:** *While it's possible for readers to comprehend the first paragraph more effectively once they've read each other's sentences…*

It picks up "the first one" and runs with it — genuine (if wobbly) use of context. With a 1024-token window and 124M parameters, coherence degrades fast, so it's a chatbot in *form*, not reliability. Still: base → SFT → DPO → chat, a full pipeline you can [poke at in the browser](https://huggingface.co/spaces/submarat/gpt2-fineweb-chat).

## Tool calling: teaching models to use external systems

Tool calling is a harder specialized-fine-tuning task because it requires structured output — valid JSON function calls with correct syntax, parameter extraction, and error handling. Models are trained on tool-calling conversations: a request, a formatted tool call, the tool's response, and a final answer incorporating the result.

How small can this go? TinyAgent [[4]](#ref-4) showed a 1.1B model reaching 78.89% on function-calling after specialized fine-tuning (vs. 12.71% off the shelf), and a 7B reaching 83.09% — beating GPT-4-Turbo's 79.08%. Key techniques: LoRA on 40k real examples, ToolRAG to select only relevant tools per query, and quantization for edge deployment. Below ~1B parameters, models seem to lack the capacity for reliable structured generation — which is exactly why my 124M toy stops at chat and doesn't attempt tools.

## What I'm still thinking about

- **How robust is preference optimization really?** DPO trains on a fixed dataset of pairs; if deployment differs, does it degrade gracefully? (My toy's regression on a factual prompt suggests "not always gracefully.")
- **SFT quality vs. the alignment ceiling.** If SFT data has subtle errors, does preference optimization correct or amplify them? The DPO-verbosity result hints that preference signals can pull *away* from correctness when capacity is the binding constraint.
- **Scaling laws for tool calling.** TinyAgent put the floor near 1.1B. Hard architectural limit, or can better data/curricula push it lower?

These are best answered by running experiments rather than theorizing — which is why the worked example above exists, and why every model and script is public.

## Try it / resources

- **Demo:** [in-browser chat (base / SFT / DPO / multi-turn)](https://huggingface.co/spaces/submarat/gpt2-fineweb-chat)
- **Models:** [base](https://huggingface.co/submarat/gpt2-small-fineweb-edu-10b) · [SFT](https://huggingface.co/submarat/gpt2-small-fineweb-edu-10b-sft) · [DPO](https://huggingface.co/submarat/gpt2-small-fineweb-edu-10b-dpo) · [chat](https://huggingface.co/submarat/gpt2-small-fineweb-edu-10b-chat)
- **Code:** [gpt2-small-repro](https://github.com/submarat/gpt2-small-repro) (`posttraining/`) · **How the base model was trained:** [Reproducing GPT-2 Small](/reproducing-gpt2-small)

---

## References

<a id="ref-1"></a>**[1]** Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

<a id="ref-2"></a>**[2]** Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

<a id="ref-3"></a>**[3]** Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

<a id="ref-4"></a>**[4]** Erdogan, L.E., et al. (2024). *TinyAgent: Function Calling at the Edge.* EMNLP 2024 Demo. [arXiv:2409.00608](https://arxiv.org/abs/2409.00608)
