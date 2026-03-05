---
layout: post
title: "How LLMs go from base models to assistants"
date: 2025-06-24
excerpt: "Notes on how language models are transformed from raw next-token predictors into useful assistants. Covers the three main stages of post-training — supervised fine-tuning, reinforcement learning from human feedback, and direct preference optimization — along with practical details about tool calling and training dynamics."
reading_time_minutes: 12
---

Notes I've accumulated from conversations with Claude while learning how language models are transformed from raw next-token predictors into useful assistants. This post covers the three main stages of post-training — supervised fine-tuning, reinforcement learning from human feedback, and direct preference optimization — along with practical details about tool calling and training dynamics that aren't always obvious from the papers.

## The post-training pipeline

A pretrained language model is essentially a document completer. It can produce fluent text, but it doesn't know how to follow instructions, refuse harmful requests, or call external tools. Post-training is the process that bridges this gap, and the three-stage pipeline was formalized by the InstructGPT work [[1]](#ref-1), typically in stages:

1. **Supervised fine-tuning (SFT)** teaches the model to follow instructions
2. **Preference optimization** (RLHF or DPO) aligns the model's outputs with human judgments about quality
3. **Specialized fine-tuning** for tool calling, multi-turn chat, or domain-specific behavior

Each stage builds on the last. You can think of SFT as teaching the model the format and style of helpful responses, while preference optimization teaches it to distinguish between good and bad responses within that format.

## Supervised fine-tuning: same objective, different data

SFT uses the exact same training objective as pretraining — next-token prediction with cross-entropy loss. For a target sequence with tokens (y₁, y₂, ..., yₙ), the loss is:

$$L = -\sum_{i=1}^{n} \log P(y_i \mid x, y_1, \ldots, y_{i-1})$$

The key difference from pretraining is the data: instead of broad internet text, SFT uses curated instruction-response pairs that demonstrate the behavior you want. The model learns "given this kind of instruction, produce this kind of response" through the same maximum likelihood mechanism it used to learn language in the first place.

### Prompt masking matters

One important practical detail: during SFT, you typically mask the loss on the instruction tokens and only compute loss on the response tokens. The implementation is straightforward — set labels to -100 for instruction positions, and the loss function ignores them:

```python
input_ids = [instruction_tokens + response_tokens]
labels = [-100] * len(instruction_tokens) + response_tokens
loss = cross_entropy(logits, labels, ignore_index=-100)
```

Without this masking, the model wastes capacity learning to reproduce the prompt rather than learning to generate good responses. The model can still attend to the instruction tokens during the forward pass — they're only masked in the loss computation, not in attention.

### Cross-entropy and KL divergence: a subtle point

Something that tripped me up: SFT's cross-entropy loss against a one-hot target is mathematically equivalent to minimizing KL divergence from that one-hot distribution to the model's output distribution. Since the entropy of a one-hot distribution is zero, D_KL(q ∥ p) = H(q, p) − H(q) = H(q, p). So cross-entropy and KL divergence collapse to the same thing.

But when people say "KL divergence" in the fine-tuning literature, they usually mean comparing two soft distributions — a teacher model's logits against a student's (knowledge distillation), or a trained policy against a reference policy (DPO/RLHF regularization). The one-hot case is degenerate: you're maximizing likelihood of specific tokens, not matching a full distribution.

### Practical training considerations

A few things I've learned matter more than you'd expect:

**Batching by sequence length** dramatically speeds up training compared to shuffling sequences of wildly different lengths, since shorter sequences in a batch waste compute on padding. You can group sequences by length and shuffle within groups to get most of the speed benefit while preserving some randomness in gradient estimates.

**Monitoring validation loss** is critical. Instruction datasets are often small relative to pretraining corpora, making overfitting a real concern. If your training loss drops to zero on some batches, that's a sign of memorization. When training loss hits zero, the model is predicting correct tokens with 100% confidence — possibly a sign of overfitting, though it can also happen with numerical precision issues on easy examples.

**Context length mismatch** is an underappreciated problem. If you fine-tune a model with 32k context capability using only 1k-token examples, the model can lose its ability to handle longer contexts. Positional embeddings for positions beyond your training length become undertrained and drift. The practical fix is to pack multiple short examples into longer sequences (separated by special tokens) so the model maintains exposure to longer contexts.

## From SFT to preference optimization

SFT gets you a model that produces responses in the right format, but it doesn't learn to distinguish between mediocre and excellent responses to the same prompt. That's where preference optimization comes in.

### RLHF: the original approach

The classic RLHF pipeline has three stages:

1. Start with an SFT model
2. Train a separate reward model on human preference data (pairs of responses where humans indicate which is better)
3. Use PPO (Proximal Policy Optimization) [[2]](#ref-2) to optimize the language model against the reward model, with a KL penalty to prevent it from drifting too far from the SFT model

PPO's strengths include flexibility — it can work with any differentiable reward function, not just pairwise preferences — and online learning, where the model generates new responses and gets feedback on them during training. It also handles distribution shift better when the optimal policy is very different from the reference model.

The downsides are well-known: PPO is computationally expensive, training is unstable, and the reward model can become a bottleneck that doesn't perfectly capture human preferences.

### DPO: cutting out the middleman

DPO (Direct Preference Optimization) [[3]](#ref-3) reformulates preference learning as a classification task, eliminating the reward model and RL loop entirely. The key mathematical insight is a reparameterization that connects the optimal policy directly to the preference data.

Given a prompt and two responses (one preferred, one dispreferred), DPO's loss encourages the model to assign higher probability to the preferred response relative to a reference model. The loss uses the Bradley-Terry preference model and is computed on a per-token basis — it looks at log probability ratios between preferred and dispreferred completions, scaled by a hyperparameter β that controls how far the model can deviate from the reference.

In practice, DPO is simpler to implement, more computationally efficient, and more stable during training. It has become the default choice for most preference-based alignment work.

### When PPO still wins

PPO retains advantages in a few scenarios: when you want to optimize for multiple distinct objectives simultaneously (helpfulness AND harmlessness AND honesty), when your reward signal isn't based on pairwise preferences, or when the optimal behavior is very different from the reference model's behavior. The additional complexity buys you flexibility.

## Tool calling: teaching models to use external systems

Tool calling is one of the more interesting fine-tuning challenges because it requires the model to learn structured output generation — producing valid JSON function calls with correct syntax, proper parameter extraction, and appropriate error handling.

### How tool calling training works

Models are fine-tuned on datasets containing examples of tool-calling conversations: a human request, a properly formatted tool call, the tool's response, and the model's final answer incorporating the tool result. The model learns when to use tools, how to format calls correctly, and how to chain multiple tool calls together.

### The smallest tool-calling models

How small can a model be and still do tool calling? Research on TinyAgent [[4]](#ref-4) showed that a 1.1B parameter model achieved 78.89% success on function calling tasks after specialized fine-tuning — but the same model scored only 12.71% off the shelf. The 7B version reached 83.09%, actually beating GPT-4-Turbo's 79.08%.

Key techniques that made small-model tool calling work included LoRA fine-tuning on 40k real-world examples, ToolRAG to select only relevant tools per query (reducing prompt size), and quantization for edge deployment. Below about 1B parameters, models appear to lack the representational capacity for reliable structured output generation.

## Step-level credit assignment

One research direction I found particularly interesting: using Q-value models to improve LLM agent decision-making [[5]](#ref-5). The core problem is that agents typically receive sparse rewards (only at task completion), making it hard to learn which intermediate actions were good or bad.

The approach trains a separate Q-value model via step-level DPO — rather than comparing entire trajectories ("this sequence of actions succeeded, that one failed"), it compares individual actions at each step ("clicking product A here was better than clicking product B"). This granular credit assignment yields dramatically better results: 103% improvement on WebShop tasks, with enhanced agents outperforming GPT-4o-mini.

One interesting implementation detail: the Q-value calculation at inference time requires both the trained model and the original reference model, since Q-values are computed as the difference in log probabilities between them. This means you're running three models at inference: the agent, the Q-value model, and the reference model — though the latter two can be small (1.3B parameters).

## What I'm still thinking about

A few open questions from these explorations:

**How robust is preference optimization really?** DPO trains on a fixed dataset of preference pairs. If the preference distribution in deployment differs from training, does the model degrade gracefully or catastrophically?

**The relationship between SFT quality and alignment ceiling.** If your SFT data contains subtle errors or inconsistencies, does preference optimization correct for them, or does it amplify them?

**Scaling laws for tool calling.** TinyAgent [[4]](#ref-4) showed that 1.1B parameters is roughly the floor for reliable function calling. Is this a hard architectural limit, or could better training data and curricula push it lower?

These are the kinds of questions that I think are best answered by actually running experiments rather than theorizing — which is part of why I'm documenting my learning process publicly.

---

## References

<a id="ref-1"></a>**[1]** Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). *Training language models to follow instructions with human feedback.* NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

<a id="ref-2"></a>**[2]** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

<a id="ref-3"></a>**[3]** Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

<a id="ref-4"></a>**[4]** Erdogan, L.E., Lee, N., Jha, S., Kim, S., Tabrizi, R., Moon, S., Hooper, C., Anumanchipalli, G., Keutzer, K., & Gholami, A. (2024). *TinyAgent: Function Calling at the Edge.* EMNLP 2024 Demo. [arXiv:2409.00608](https://arxiv.org/abs/2409.00608)

<a id="ref-5"></a>**[5]** Zhai, Y., Yang, T., Xu, K., Feng, D., Yang, C., Ding, B., & Wang, H. (2024). *Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models.* AAAI 2025. [arXiv:2409.09345](https://arxiv.org/abs/2409.09345)
