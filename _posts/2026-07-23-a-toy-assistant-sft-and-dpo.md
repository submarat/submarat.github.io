---
layout: post
title: "A toy assistant: SFT and DPO on my from-scratch GPT-2"
date: 2026-07-23
excerpt: "I took the 124M GPT-2 I trained from scratch and ran the canonical post-training pipeline on it — supervised fine-tuning on Alpaca, then DPO on UltraFeedback. SFT gave a dramatic jump to instruction-following; DPO made it 'sound' more like an assistant while demonstrating the alignment tax in miniature. There's an in-browser demo."
reading_time_minutes: 8
---

Having [reproduced GPT-2 small from scratch](/reproducing-gpt2-small), I wanted to do something with the weights other than admire the loss curve. The obvious move — and the natural companion to my [post on how base models become assistants](/how-llms-go-from-base-models-to-assistants) — was to actually run the pipeline: **supervised fine-tuning, then DPO**. Could a 124M model become a chat assistant? Short answer: a *toy* one — it follows simple instructions and holds the format, but it's shallow and hallucinates confidently. That's exactly what makes it a good thing to have built and poked at. As before, worked out with Claude.

**[Try it in your browser →](https://huggingface.co/spaces/submarat/gpt2-fineweb-chat)** (runs the model client-side with transformers.js — compare base vs SFT vs DPO).

## Stage 1 — SFT: the big jump

SFT is, mechanically, the *same next-token objective as pretraining, on different data* — instruction→response pairs instead of raw web text. I used [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) (~52k examples) with TRL's `SFTTrainer`, formatting each row as a prompt/completion pair so the loss is masked to the response only (the model learns to *produce* answers, not echo prompts):

```
### Instruction:
{instruction}

### Response:
{output}          ← loss computed here only
```

Three epochs on the 124M base (~8 minutes on one H100), and instruction-following simply *appears*. The contrast with the base model is stark:

| Instruction | Base (rambles / ignores) | After SFT |
|---|---|---|
| "List three tips for staying focused while studying." | vague musing about reading | a numbered 3-item list |
| "What is the capital of France?" | drifts into a history lesson plan | **"Paris"** |
| "Summarize: *The sun is a star at the center of the solar system…*" | unrelated arithmetic | "Sun, it's just another super-ball of hot plasma" |

It still hallucinates — asked for a motivational quote it invented an attribution, *"— Paulo Freirer (2006)"* — but it's unmistakably following instructions now. That's the SFT payoff: format and instruction-following are cheap to teach; factual reliability is not.

## Stage 2 — DPO: the alignment tax, in miniature

Direct Preference Optimization nudges the SFT model toward preferred responses using (prompt, chosen, rejected) triples — no reward model, no PPO. I used [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) (15k pairs) with TRL's `DPOTrainer`. The training metrics did exactly what they should: the reward margin rose to ~0.28 and the model preferred "chosen" over "rejected" ~63% of the time.

And the behavior *did* change — DPO made responses **longer, more detailed, more "assistant-like"**. This is the well-documented UltraFeedback length/verbosity preference: annotators (and the models that labeled the data) tend to prefer fuller answers, so DPO pushes the model to be verbose.

Here's the catch, and the most interesting result of the whole exercise: **at 124M, verbosity isn't quality — it's more room to hallucinate.** DPO optimized the preference signal it was given, but that signal is a proxy for "good," not ground truth. So on some prompts it *regressed*:

| Instruction | SFT | After DPO |
|---|---|---|
| "What is the capital of France?" | **"Paris"** ✓ | a long, confident paragraph about Belgium, Luxembourg, and Louis XIV ✗ |
| "Write a motivational quote about learning." | short quote (+ fake attribution) | longer quote + a fabricated author *and* a fabricated TEDx talk |

The SFT model gave the right one-word answer; the DPO model, optimized to be more elaborate and "helpful-sounding," talked itself out of it. That's the **alignment tax / reward-vs-truth gap** you read about in the RLHF literature, reproduced here on a tiny model you can train in an afternoon: making a model *sound* more aligned to a preference is not the same as making it *more correct*, especially when the base model lacks the capacity to back up the extra words.

## Takeaways

- **SFT is where the transformation happens.** A near-instant jump from "document completer" to "instruction follower," using the same objective as pretraining.
- **DPO changes style, and style ≠ substance.** It reliably shifted the model toward longer, more assistant-like output — and just as reliably exposed that a 124M model has nothing to fill that extra length with but confident guesses.
- **Watching the alignment tax happen** on a model small enough to fully understand is worth more than reading about it. The reward metric went up; the answers, on balance, did not.

For a usable toy, the **SFT** model is the one to reach for. Both are on the Hub:
[SFT](https://huggingface.co/submarat/gpt2-small-fineweb-edu-10b-sft) ·
[DPO](https://huggingface.co/submarat/gpt2-small-fineweb-edu-10b-dpo) ·
[base](https://huggingface.co/submarat/gpt2-small-fineweb-edu-10b), with all the training code in
[the repo](https://github.com/submarat/gpt2-small-repro) under `posttraining/`.
