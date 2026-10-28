---
layout: post
title: "How big are frontier RL runs? A survey of open-weight models"
date: 2026-10-28
excerpt: "Rollout lengths, token budgets, environment counts, GPUs and costs for frontier-scale RL post-training, as disclosed by open-weight models in 2025-26. Plus which models release RL checkpoints you can actually study, how rollouts are batched for the update, and why MiMo-V2.6 stopped after 30 steps."
reading_time_minutes: 36
---

A frontier RL run in 2026 generates on the order of 10^11 tokens. That is roughly 25K trajectories of 100–150K tokens each per step, from 10^3–10^5 environment instances, on thousands of GPUs, for about a week and $1–3M. The best-documented calibration point is MiMo-V2.6 (Oct 2026), which is about what "frontier 6–9 months ago" looks like. Since then, open models have grown to 1.6–2.8T parameters.

**Scope.** This survey covers open-weight models: ones whose weights are public, and ideally their checkpoints, data and RL recipes too, so they can themselves be objects of study. Closed models (OpenAI, Anthropic, Google, xAI) appear only as reference points, where their labs have disclosed ratios or totals. Our own runs on MiMo-V2.6-Flash and OLMo 3 are included to calibrate the published numbers.

## At a glance

"Derived" marks my own arithmetic from disclosed numbers.

| Question | Typical frontier answer (2025–26) | Best-documented examples |
| --- | --- | --- |
| How long are rollouts? | Reasoning RL: a 32K→64K token cap, with means of 10–20K. Agentic RL: 100–150K tokens and 50–65 turns per trajectory, in contexts up to 1M. | MiMo-V2.6: about 110–150K tokens per sequence. Nemotron 3 Ultra: generation cap raised 48K→64K. OLMo 3: mean 14.6K under a 32K cap. |
| How many tokens? | About 3B tokens per step; 10^10–10^11 per run. | MiMo-V2.6: 2.7–3.7B per step over about 30 steps, about 100B in total (derived). OLMo 3.1: about 34B (derived). DeepSeek-R1-Zero: about 43–85B (derived). |
| How many environments? | 10^3–10^5 executable instances across 4–25 domain types. | MiMo-V2-Flash: about 320K agentic tasks. DeepSeek-V3.2: 1,827 synthesized environments and about 85K prompts. GLM-5: more than 10K. MiniMax M2.5: "hundreds of thousands". |
| How many GPUs? | RL: hundreds to a few thousand GPUs for days to weeks. Pretraining: 1K–32K+ GPUs for 1–2 months. | DeepSeek-R1 and MiniMax-M1: 512 H800. INTELLECT-3: 512 H200. MiMo-V2.6: "thousands of GPUs in a single run". Grok 4: RL on a 200K-GPU cluster. |
| What does RL cost? | $0.3–3M for open labs; RL is 5–20% of pretraining compute, rising to about 100% at xAI. | DeepSeek-R1: $294K. MiniMax-M1: $535K. MiMo-V2.6: $0.9M (Flash) and $2.6M (Pro). DeepSeek-V3.2: more than 10% of pretraining. LongCat: about 20%. |
| Largest open models? | 1–2.8T total parameters, trained on 30–48T tokens. | Kimi K3: 2.8T. Qwen3.8: 2.4T. DeepSeek-V4-Pro: 1.6T, 33T tokens. MiMo-V2.6-Flash: 48T tokens, the most disclosed. |

**Follow-up questions, short answers** (details in the sections below):

| Question | Short answer |
| --- | --- |
| What does the length distribution look like? | Right-skewed: p90 is 1.5–2.6× the median. OLMo 3 Think's mix is bimodal: math median about 17K tokens, chat and instruction following about 1K. |
| Batching, or tiny micro-batches with gradient accumulation? | Both: everyone accumulates gradients. RL stacks pack short sequences into token-budget rows; past about 32K tokens it is one sequence per micro-batch. Usually one optimizer step per rollout batch. |
| Which open models release checkpoints from within RL? | Only AI2: OLMo 3.1 32B Think (46), OLMo 3 7B Think (55), Tülu 3 405B (3). Large models give at most a before/after pair (DeepSeek-V4, R1-Zero, Nemotron 3 Ultra). |
| How did MiMo pretraining cost compare with its RL? | Estimated about $12M (Pro) and $7M (Flash), against $2.6M and $0.9M for RL: RL was about 13–20% of pretraining. |
| What do long-context mid-training and SFT achieve? | Mid-training adds dense math and code knowledge and a 256K–1M context. SFT teaches format and tool use, and starts pass rates where RL can learn. Most capability exists before RL. |
| Why full-parameter RL despite "LoRA Without Regret"? | No lab says. Likely: frontier batches exceed where LoRA was shown to work, episodes carry far more than one bit, and stacks are built for full weights. |
| Does Inkling use LoRA, and are its checkpoints out? | Not stated. Inkling is 975B / 41B, open weights, final checkpoints only. Tinker uses LoRA for customers' fine-tuning. |
| Who uses LoRA for RL in practice? | Tinker (LoRA-only; mainly research groups) and Fireworks RFT. No flagship model reports it. |
| Why did MiMo stop at 30 steps? | No reason given and still improving. Most likely a fixed budget timed to the release: livestreamed, both models stopped at exactly 30, weights out two days later. |
| Many small steps or few large ones? | Small and many learns more per rollout early; large and few reaches a higher ceiling, faster and more stably. Avoid several optimizer steps per rollout batch. |

## Rollout lengths

Reasoning RL settled on a 64K generation cap in late 2025. Agentic RL now runs trajectories of 100–150K tokens over dozens of turns, inside contexts of up to 1M.

**Single-response reasoning RL.** Caps doubled from 32K to 64K during 2025. Mean lengths sit at about a third to a half of the cap.

| Model (release) | RL generation cap | Typical / mean length |
| --- | --- | --- |
| [Nemotron 3 Ultra](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) (Jun 2026) | 48K, raised to 64K | not disclosed |
| [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) (Dec 2025) | 49K | not disclosed |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (Dec 2025) | 128K context; Speciale trained with a reduced length penalty | eval outputs 7–42K (V3.2) and 23–77K (Speciale) |
| [OLMo 3 Think 32B](https://arxiv.org/abs/2512.13961) (Nov 2025) | 32K | mean 14.6K |
| [INTELLECT-3](https://arxiv.org/abs/2512.16144) (Nov 2025) | 64K context | not disclosed |
| [Ring-1T](https://arxiv.org/abs/2510.18855) (Oct 2025) | 64K reasoning, 32K general | not disclosed |
| [LongCat-Flash-Thinking](https://arxiv.org/abs/2509.18883) (Sep 2025) | 64K for STEM; code staged 48K→56K→64K; agentic 48K | not disclosed |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) (Jul 2025) | 64K in one stage (beat a 16K→64K curriculum) | not disclosed |
| [MiniMax-M1](https://arxiv.org/abs/2506.13585) (Jun 2025) | 40K, staged up to 80K | above 20K late in training |
| [Magistral Medium](https://arxiv.org/abs/2506.10910) (Jun 2025) | 40K | length grew 16K→32K over training |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (Jan 2025) | 32K; R1-Zero raised to 64K at step 8.2K | length "jumps" at the cap change |

**Agentic RL.** Here a rollout is a whole multi-turn episode, and its length is the full context, not one reply.

- **MiMo-V2.6 (Oct 2026):** 2.7–3.7B tokens per step over about 25K sequences is "roughly 110K–150K tokens per sequence", in a 1M context ([paper §4.1](https://arxiv.org/html/2610.11959v1)).
  - Averages of about 57 turns (Flash) and 65 (Pro) come only from a secondary summary of Xiaomi's live dashboard.
- **Xiaomi's released RL config:** 262,144-token trajectory cap, 32,768 tokens per turn, 4,800 s timeout ([verl mimo-oss](https://raw.githubusercontent.com/XiaomiMiMo/verl/mimo-oss/scripts/code/train.sh)).
- **Other turn counts:** Kimi K2-Thinking reports 200–300 sequential tool calls at inference (secondary). INTELLECT-3's SWE environment caps episodes at 200 turns. MiniMax-M1 evaluated agents at 40 steps.
- **Trend:** labs now push length *down* as well as up: length penalties (DeepSeek-V3.2), Kimi K2.5's "Toggle" (−25–30% tokens), and per-task budgets (Kimi K3).

## Rollout length distributions

Every length distribution we have is right-skewed: the p90 is 1.5–2.6× the median, and the p99 is 2–5×. The public frontier reports give only means and caps, so the full distributions below come from our own runs.

![Rollout length percentiles for OLMo 3 and MiMo-V2.6 runs, log scale](/images/frontier-rl-runs/rollout_length_distributions.png)

*Ours: OLMo 3 7B GRPO runs on GSM8K and math, MiMo-V2.6-Flash agentic rollouts and LoRA training step. OLMo 3 Think 32B SFT traces: 1,200 sampled from Dolci-Think-SFT-32B on Hugging Face. Published: OLMo 3 report §4.4.3, MiMo-V2.6 report §4.1.*

- **No public OLMo 3 Think distribution exists.** The report gives a mean of 14,628 tokens under a 32,768 cap, and says static batching would waste "up to 54%" of compute on the tail ([OLMo 3 §4.4.3](https://arxiv.org/html/2512.13961)).
  - The chart's OLMo 3 Think 32B estimate uses the traces the model imitated in SFT just before RL: 1,200 from Dolci-Think-SFT-32B, reweighted to the RL mix (30% math, 21% code, 29% instruction following, 20% chat). The RL datasets on Hugging Face keep only final answers, not reasoning. The distribution is bimodal: math has a median of 17K tokens, instruction following and chat about 1K. Its mean, 7.8K, is about half the published RL mean.
  - Stretching the SFT lengths to reach the 14.6K RL mean would put 29–36% of rollouts at the cap, so the RL distribution probably differs in shape, not just scale. RL prompts were filtered for difficulty, and DPO lengthens outputs. Sampling OLMo 3's published RL checkpoints would settle it. Our own OLMo 3 7B rows are toy settings (a base model with 2K and 4K caps).
- **The tail sets wall-clock time.** Moonshot's Seer saw lengths from a few hundred tokens to 96K, with the last 10% of requests taking up to 50% of rollout time ([Seer](https://arxiv.org/html/2511.14617v3)). RollPacker measured the longest rollout at 25–32× the median ([RollPacker](https://arxiv.org/html/2509.21009v1)).
- **Lengths drift during RL.** In our OLMo 3 7B GSM8K run, median length rose from 208 to 371 tokens over 150 steps, and p99 from 1.2K to the 2K cap. OLMo 3 Think's length "first slightly dips and then slowly increases". MiniMax-M1 raised its cap from 40K to 80K whenever p99 approached it ([MiniMax-M1](https://arxiv.org/html/2506.13585)).
- **Agentic rollouts have three different "lengths":**
  - tokens the model writes (median 23K in our MiMo runs);
  - the final training sequence (median 63K);
  - the context re-read across all calls (median 1.2M), which drives generation cost unless the prefix cache hits.

## Batching rollouts for the policy update

At frontier lengths, gradient accumulation over small micro-batches is universal: no GPU holds a 25K-trajectory batch in one pass. The real choice is what fills a micro-batch.

**How common is each practice?**

- **Several rollouts per forward pass: universal where disclosed.**
  - Every frontier model that publishes its training stack packs rollouts into fixed-token rows: OLMo 3, MiMo-V2.6, GLM (slime), Nemotron (NeMo-RL) and INTELLECT-3 (prime-rl).
  - Padding survives in general-purpose trainers (TRL, OpenRLHF's default).
  - DeepSeek, Kimi, Qwen and MiniMax do not say.
- **How much packing actually batches depends on length.**
  - OLMo 3 Think's 35,840-token rows hold about 2–3 rollouts at its 14.6K mean. Long math rollouts often fill a row alone, while short chat and instruction-following ones share.
  - Agentic trajectories of 110–150K tokens are one per micro-batch at most, and context parallelism splits each across GPUs.
- **Several optimizer steps per rollout batch: now the minority.**
  - One step per batch is the async-era default: OLMo 3, MiMo, slime, NeMo-RL, prime-rl and TRL.
  - Several steps appear in verl and AReaL by default (4), OpenRLHF (8), and in DeepSeek-R1, DAPO and MiniMax-M1 (16). These are mostly earlier synchronous setups.

* **Packing.** RL-native stacks pack several short sequences into one row up to a token budget, so short rollouts share a forward pass and nothing is padded.
* **Padding.** Older or general trainers pad a fixed count of sequences per micro-batch, which wastes compute on padding tokens.
* **Long rollouts.** Once rollouts pass about 32K tokens, a micro-batch holds one sequence. Context parallelism then splits that sequence across GPUs.

| Stack or model | What fills a micro-batch | Key settings | Optimizer steps per rollout batch |
| --- | --- | --- | --- |
| [OLMo 3 32B Think](https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/32b_think_rl.sh) (open-instruct) | One packed row per GPU pass | 35,840-token rows; 128 prompts × 8 per the report (the launch script says 64); 64 learner GPUs | 1 |
| [MiMo verl release](https://raw.githubusercontent.com/XiaomiMiMo/verl/mimo-oss/scripts/code/train.sh) | Token-budget micro-batches of 1 (Megatron) | Up to 262,144 tokens per sequence; tensor parallel 8 | 1 |
| [MiMo-V2.6 production](https://arxiv.org/html/2610.11959v1) | Packed micro-batches, context parallel | 1M-token context; one packer per tensor-parallel group | not stated |
| [slime](https://raw.githubusercontent.com/THUDM/slime/main/slime/utils/arguments.py) (GLM) | Always packed, token budget per GPU | Budget ≈ max response length ÷ context-parallel size | 1 |
| [NeMo-RL](https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main/examples/configs/grpo_math_1B.yaml) | Packing on by default, or dynamic batching | Packed rows of 4 × sequence length | 1 |
| [prime-rl](https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/packages/prime-rl-configs/src/prime_rl/configs/trainer.py) (INTELLECT-3) | One packed sequence of fixed length per micro-batch | — | 1 |
| [verl](https://verl.readthedocs.io/en/latest/examples/config.html) defaults | Fixed micro-batch of 8; token budget optional | Mini-batch 256 of 1,024 | 4 |
| [verl DAPO 32B recipe](https://raw.githubusercontent.com/verl-project/verl-recipe/main/dapo/run_dapo_qwen2.5_32b.sh) | Token budget plus 8-way sequence parallel | 22,528 tokens per GPU; 512 prompts × 16 | 16 |
| [AReaL](https://raw.githubusercontent.com/inclusionAI/AReaL/main/areal/api/cli_args.py) | Token budget | — | 4 |
| [OpenRLHF](https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/main/openrlhf/cli/train_ppo_ray.py) | Padded micro-batch of 1 by default; packing optional | — | 8 |
| [TRL GRPOTrainer](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_config.py) | Padded micro-batches with gradient accumulation | — | 1 |
| [DeepSeek-R1](https://arxiv.org/html/2501.12948) | not stated | 8,192 outputs split into 16 mini-batches | 16 |
| [MiniMax-M1](https://arxiv.org/html/2506.13585) | not stated | "16 rounds of off-policy updates per generation batch" | 16 |
| Our MiMo-V2.6-Flash LoRA step | One sequence per pass, gradients accumulated over 88 | Sequences up to 118K tokens, layers split across 4 GPUs | 1 |

- **Padding is the cost packing removes.** NeMo-RL's design notes estimate that padding Zipf-like length distributions wastes 50–70% of compute ([source](https://raw.githubusercontent.com/NVIDIA-NeMo/RL/main/docs/design-docs/sequence-packing-and-dynamic-batching.md)).
- **Async and long-rollout systems take one optimizer step per rollout batch.** That keeps updates close to on-policy. Several mini-batches per batch (16 for R1, DAPO and M1) appear in synchronous setups that reuse one expensive generation batch, at the price of more off-policy correction.
- **The long tail is handled on the rollout side, not in update batching.** Systems use continuous batching, partial rollouts that carry unfinished sequences into the next step, and caps raised gradually.
- **Memory, not batching, is the binding limit for long sequences.** MiMo-V2.6 reports MoE expert load as the main risk inside a micro-batch: one expert-parallel rank reached over 30× the mean load.

## Tokens

Frontier RL runs generate about 10^10–10^11 tokens. Batches are modest, a few thousand to 25K rollouts per step, so the scale comes from rollout length.

| Model | Batch per step (prompts × samples) | Steps | Tokens per step | Total RL tokens |
| --- | --- | --- | --- | --- |
| [MiMo-V2.6](https://arxiv.org/html/2610.11959v1) (Oct 2026) | 1,568 × 16 = 25,088 | about 30 (training curves cover 30 steps) | 2.7–3.7B | about 100B (derived: 30 × 3.2B) |
| [Nemotron 3 Ultra](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) (Jun 2026) | 8,192 × 16 = 131,072 | not disclosed | not disclosed | not disclosed |
| [INTELLECT-3](https://arxiv.org/abs/2512.16144) (Nov 2025) | 256 × 16 = 4,096 | not disclosed | at most 0.27B (derived, 64K cap) | not disclosed |
| [OLMo 3.1 Think 32B](https://arxiv.org/abs/2512.13961) (Dec 2025) | 128 × 8 = 1,024 | 750 (OLMo 3), 2,300 (OLMo 3.1) | about 15M (derived: 1,024 × 14.6K) | about 11B (OLMo 3), about 34B (OLMo 3.1), both derived |
| [Ring-1T](https://arxiv.org/abs/2510.18855) (Oct 2025) | 480 × 8 = 3,840 | not disclosed | not disclosed | not disclosed |
| [Qwen3-235B](https://arxiv.org/abs/2505.09388) (May 2025) | not disclosed | 170 on 3,995 prompts | not disclosed | not disclosed |
| [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) (Jan 2025) | 32 × 16 = 512 | 10,400 | about 4–8M (derived) | about 43–85B (derived, assuming an 8–16K mean) |

- **Steps are few when rollouts are long.** MiMo-V2.6 took about 30 very large steps. OLMo took thousands of small ones.
- **Prompt pools are about 10^5.** DeepSeek-R1 used about 146K prompts, OLMo 3 Think 104,869, and DeepSeek-V3.2's agentic stage about 85K.
- **Context tokens far exceed generated tokens in agentic RL.** Each turn re-reads the whole history. Our MiMo-V2.6-Flash rollouts generated a median 23K tokens but processed 1.2M prompt tokens each (96% prefix-cache hits).

## Environments

Frontier runs use 10^3–10^5 executable environment instances, grouped into 4–25 domain types. Which of the two a headline counts changes the number by orders of magnitude.

Three senses of "environment" appear in reports:

- **Type:** a harness or domain, such as SWE, search, or math with a verifier.
- **Instance:** one executable task, such as a GitHub issue in its own Docker image.
- **Prompt:** one task text with a grader.

In SWE, one instance is usually one task, so "100K environments" means about 100K issues.

| Model or collection | Count (sense) | Domains and verifiers |
| --- | --- | --- |
| [MiMo-V2.6](https://arxiv.org/html/2610.11959v1) (Oct 2026) | "Thousands" of general environments; tens of thousands of cyber instances; about 7K tasks released (3K code, 1K cyber, 1K general, 2K visual) | Mix: 68% coding, 13% design, 12% tool use, 4% cyber, 3% context following. Tests, rule checks, rubric and visual grading. |
| [MiniMax M2.5](https://github.com/MiniMax-AI/MiniMax-M2.5) (Feb 2026) | "Hundreds of thousands" (unclear sense) | Coding, tool use, search, office work |
| [GLM-5](https://arxiv.org/abs/2602.15763) (Feb 2026) | More than 10K instances | SWE in 9 languages, terminal, multi-hop search; tests and rules |
| [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) (Jan 2026) | About 320K tasks: 120K code, 150K search, 50K general; plus about 30K terminal tasks | Real GitHub issues in 8 languages, run on more than 10K Kubernetes pods; unit tests |
| [LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725) (Jan 2026) | More than 10K instances in more than 20 domain types | Tool use and agentic tasks; executable checks |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (Dec 2025) | 1,827 synthesized instances plus tens of thousands of SWE instances; about 85K prompts | Code agent, search, general agent, code interpreter; synthesized verifier functions |
| [INTELLECT-3](https://arxiv.org/abs/2512.16144) (Nov 2025) | 6 types; more than 20K SWE repo images; about 73K prompts | Math, code, science, logic, deep research, SWE |
| [OLMo 3 Think](https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B) (Nov 2025) | 4 types; 102K prompts | Math, code, instruction following, chat; answer match, tests, LLM judge |
| [Kimi K2](https://arxiv.org/abs/2507.20534) (Jul 2025) | 3,000+ real MCP tools and 20,000+ synthetic tools; more than 10K concurrent sandboxes | Rule-based verifiers plus a rubric critic |
| [Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/) (Jul 2025) | 20K environments run in parallel (concurrency, not catalogue size) | SWE; execution |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (Jan 2025) | 5 types; about 146K prompts | Math, code, STEM, logic, general; rules plus reward models |

- **SWE drives the big numbers.** The six-figure counts are mostly GitHub issues with unit tests. Open SWE sets grew from 2.4K tasks ([SWE-Gym](https://www.emergentmind.com/topics/swe-gym), secondary) to 32K+ ([SWE-rebench V2](https://nebius.com/blog/posts/meet-swe-rebench-v2)).
- **Synthesized environments are now standard.** DeepSeek-V3.2, Kimi K2, MiMo and LongCat all use LLM pipelines that write the tools, tasks and verifiers.
- **Open hubs count types, not instances.** [Prime Environments Hub](https://arxiv.org/abs/2512.16144) lists 500+ and [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) about 200 configurations.
- **Environments are now a market.** Reported prices are $200–$2K per task, and about $20K for a website-replica environment ([Epoch AI](https://epoch.ai/gradient-updates/state-of-rl-envs)). Anthropic reportedly discussed spending more than $1B on environments ([TechCrunch](https://techcrunch.com/2025/09/16/silicon-valley-bets-big-on-environments-to-train-ai-agents/)).

## GPUs and compute

Disclosed open-model RL runs use 200–1,500 GPUs, or "thousands" at 1T scale, for days to weeks. Pretraining uses 1K–32K+ GPUs. RL's share of total compute rose from under 1% at the end of 2024 to 10–20% in 2026 for open labs.

| Model | Pretraining GPUs | RL GPUs and duration | RL cost or GPU-hours | RL share of pretraining |
| --- | --- | --- | --- | --- |
| [MiMo-V2.6](https://arxiv.org/html/2610.11959v1) (Oct 2026) | not disclosed | "Thousands of GPUs in a single run"; under 6 days (secondary) | $0.9M (Flash), $2.6M (Pro). Pro's split: 43.8% rollout, 43.5% training, 12.7% grading. | not disclosed |
| [Nemotron 3 Super](https://docs.nvidia.com/nemo/rl/latest/guides/nemotron-3-super.html) (Apr 2026) | not disclosed | Recipe nodes: 183 for RLVR, 64 for SWE-RL, 100 for RLHF (about 1,464 / 512 / 800 H100, derived at 8 per node) | not disclosed | not disclosed |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (Dec 2025) | not disclosed | not disclosed; "thousands of steps" | more than 279K H800-hours (derived) | more than 10% (stated) |
| [OLMo 3.1 Think 32B](https://arxiv.org/abs/2512.13961) (Dec 2025) | 1,024 H100 for 35 days, after 9.5 days on 512 | 224 H100 for 21 more days | about 113K GPU-hours for the extension (derived) | about 12% (derived) |
| [INTELLECT-3](https://arxiv.org/abs/2512.16144) (Nov 2025) | (GLM-4.5-Air base) | 512 H200 for about 2 months. Of 60 RL nodes, 16 trained and 44 generated. | at most 737K H200-hours, including SFT and ablations (derived) | not disclosed |
| [LongCat-Flash-Thinking](https://arxiv.org/abs/2509.18883) (Sep 2025) | "Tens of thousands of accelerators" | not disclosed | not disclosed | "nearly 20%" (stated) |
| [gpt-oss-120b](https://arxiv.org/abs/2508.10925) (Aug 2025) | H100s; 2.1M H100-hours for the whole run | not disclosed | not disclosed | not disclosed |
| [MiniMax-M1](https://minimax.io/news/minimaxm1) (Jun 2025) | not disclosed | 512 H800 for 3 weeks | $534,700; about 258K GPU-hours (derived) | not disclosed |
| [DeepSeek-R1](https://www.theregister.com/2025/09/19/deepseek_cost_train/) (Jan 2025) | 2,048 H800 (V3 base) | 512 H800: 198 h for R1-Zero, about 80 h for R1 | $294K; about 147K GPU-hours | about 5% (derived) |
| [Grok 4](https://x.ai/news/grok-4) (Jul 2025, closed) | 200K-GPU Colossus | RL "at pretraining scale" | Epoch: total about 5e26 FLOP, about $490M ([estimate](https://epoch.ai/data-insights/grok-4-training-resources)) | about 100% (xAI's chart, secondary reading) |

- **Generation dominates RL cost.** INTELLECT-3 gave inference 44 of its 60 nodes. OLMo 3.1's RL extension ran on 224 H100s. MiMo-V2.6 spent about as much on rollouts as on training, plus 13% on grading.
- **Everyone runs RL asynchronously now** (partial rollouts with bounded staleness). Async frameworks such as AReaL, ROLL Flash and slime report 2–3× throughput gains.
- **Weight sync matters at 1T parameters.** Moonshot's checkpoint-engine cut a 1T broadcast from about 10 minutes to about 20 seconds across thousands of GPUs.
- **Closed labs give ratios, not GPU counts.** OpenAI's o3 used about 10× o1's RL compute. Epoch assumes GPT-5's RL was 10–200% of its pretraining ([Epoch](https://epochai.substack.com/p/notes-on-gpt-5-training-compute)).

## Largest and most data-rich open-weight models

Four 2026 open-weight models now exceed 1.5T parameters: Kimi K3 (2.8T), Qwen3.8 (2.4T), DeepSeek-V4-Pro and LongCat-2.0 (1.6T each). MiMo-V2.6-Flash has the largest disclosed pretraining corpus, 48T tokens. Openness of the RL recipe is uneven: only Nemotron 3 Ultra and MiMo-V2.6 publish concrete RL numbers at frontier scale.

| Model (release) | Total / active params | Pretraining tokens | Base model released? | RL recipe disclosed |
| --- | --- | --- | --- | --- |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) (Jul 2026) | 2.8T / 104B | not disclosed | no | mechanisms only, no numbers |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) (Aug 2026) | 2.4T / 95B | not disclosed | no | none (blog only) |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (Apr 2026) | 1.6T / 49B | 33T | yes | low: GRPO specialists, then on-policy distillation |
| [LongCat-2.0](https://huggingface.co/meituan-longcat/LongCat-2.0) (Jul 2026) | 1.6T / about 48B | more than 35T | not stated | none |
| [MiMo-V2.6-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) (Sep 2026) | 1.02T / 42B | 30T | no (RL checkpoint) | **high**: batch, tokens per step, cost, compute split; about 7K environments and the RL framework released |
| [Kimi K2 / K2.5 / K2.6](https://arxiv.org/abs/2507.20534) (Jul 2025–Apr 2026) | 1T / 32B | 15.5T, plus about 15T more for K2.5 | K2 base: yes | medium |
| [Ling-2.5-1T](https://huggingface.co/inclusionAI/Ling-2.5-1T) (Feb 2026) | 1T / 63B | 29T plus 9T continued | not linked | low |
| [Tencent Hy4-preview](https://huggingface.co/tencent/Hy4-preview) (Aug 2026) | 770B / 49B | not disclosed | no | none |
| [GLM-5](https://huggingface.co/zai-org/GLM-5) (Feb 2026) | 744B / 40B | 28.5T | not stated | medium; the slime RL framework is open |
| [Mistral Large 3](https://azure.microsoft.com/en-us/blog/introducing-mistral-large-3-in-microsoft-foundry-open-capable-and-ready-for-production-workloads/) (Dec 2025) | 675B / 41B | not disclosed | — | none |
| [DeepSeek-V3.x](https://arxiv.org/abs/2512.02556) (Dec 2024–Dec 2025) | 671B / 37B | 14.8T | yes | medium |
| [Nemotron 3 Ultra](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) (Jun 2026) | 550B / 55B | 20T | **yes, plus pretraining and post-training data and RL environments** | **high**: batch 8,192 × 16, generation cap 48K→64K |
| [MiMo-V2.6-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (Sep 2026) | 310B / 15B | **48T** (26T text, 22T omni) | no (RL checkpoint) | high, as Pro |
| [OLMo 3.1 Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think) (Dec 2025) | 32B dense | 5.9T | **yes, every stage** | **full**: data, code, hyperparameters |

**Most pretraining tokens (disclosed):**

1. MiMo-V2.6-Flash: 48T, then Thinking Machines' Inkling (975B total, 41B active, Jul 2026): 45T
2. Llama 4 Scout: about 40T
3. Ling-2.5-1T: 38T including continued pretraining
4. Qwen3: 36T
5. LongCat-2.0: more than 35T
6. DeepSeek-V4-Pro: 33T
7. MiMo-V2.6-Pro: 30T
8. GLM-5: 28.5T

Kimi K3, Qwen3.8, Qwen3.5 and gpt-oss do not disclose token counts.

**Best for studying RL openly:**

- **OLMo 3.1** is fully reproducible.
- **Nemotron 3 Ultra** is the largest model with its base, data, environments and RL recipe all open.
- **MiMo-V2.6** is the only 1T-scale release that open-sources its RL environments and training dynamics.
- **INTELLECT-3** has every environment public on the Prime Hub.

The next section covers which of these release checkpoints from partway through RL.

Some figures come from secondary sources only: Kimi K3's weights date, LongCat-2.0's release date, and the [Step 5 Preview](https://www.marktechpost.com/2026/10/16/stepfun-releases-step-5-preview-open-weights-600b-moe-27b-active-1m-context/) (about 600B, Oct 2026) and MiniMax M3 (about 428B) sizes. The MiMo-V2.6 token counts are from the [paper §3.1](https://arxiv.org/html/2610.11959v1).

## Intermediate RL checkpoints

No frontier-scale lab releases checkpoints from partway through RL; MiMo-V2.6, DeepSeek, Qwen, Kimi, GLM, MiniMax, INTELLECT and Nemotron 3 publish final weights only. AI2 is the exception. OLMo 3.1 32B Think has 46 RL checkpoints, every 50 steps up to step 2,300, alongside its open data, code and pre-RL starting points.

Checkpoints are Hugging Face branches, counted via each repo's refs listing on 27 Oct 2026. Load one with `revision="step_XXXX"`.

| Model or suite | Size | RL checkpoints (spacing) | Also released |
| --- | --- | --- | --- |
| [OLMo 3.1 32B Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think) | 32B | **46**: steps 50–2,300 (every 50) | Data, code, SFT and DPO stages; logs "coming soon" |
| [OLMo 3 32B Think](https://huggingface.co/allenai/Olmo-3-32B-Think) | 32B | 15: steps 50–750 (every 50) | Same as above |
| [OLMo 3 7B Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | 7B | **55**: steps 25–1,375 (every 25) | Same as above |
| [OLMo 3 / 3.1 7B RL-Zero](https://huggingface.co/allenai/Olmo-3-7B-RL-Zero-Math) (Math, Code, IF, General, Mix) | 7B | 8–33 per run (every 50–100) | Seven runs from one base model, one per domain |
| [OLMo 3 7B Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | 7B | 8: steps 50–400 | Instruct SFT and DPO stages |
| [Tülu 3 405B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B) | 405B | 3: steps 25, 50, 75 | The largest RL step series found; PPO-style RLVR |
| [Tülu 3.1 8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B) | 8B | 61: steps 40–2,440 (every 40) | GRPO |
| [OLMo 2 32B Instruct](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct) | 32B | 17: steps 20–340 | Also 13B (13 checkpoints) and 1B (13) |
| [Nemotron-Cascade-8B](https://huggingface.co/nvidia/Nemotron-Cascade-8B-Intermediate-ckpts) | 8B | 5 stage ends: RLHF → IF-RL → math RL → code RL → SWE RL | RL datasets |
| [RLVR-Linearity](https://huggingface.co/Miaow-Lab/RLVR-Linearity-Checkpoints) | 1.5B | 13: steps 100–1,300 | Paper on weight-update linearity |
| [ProRL Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) | 1.5B | 3 versions of one long run (v1, v2, BroRL) | — |

- **For frontier-scale agentic RL, there are no checkpoints, but there are metrics and environments.** MiMo-V2.6's "training dynamics" release is a live per-step dashboard (reward, entropy, KL, cost) at [mimo.xiaomi.com/rl](https://mimo.xiaomi.com/rl) ([TechNode](https://technode.com/2026/09/18/xiaomi-livestreams-mimo-v2-6-reinforcement-learning-runs/)). Its RL environments are released too. The weights are one RL checkpoint and one after the later distillation stage, per size.
- **Best substrate for behavior-over-RL studies:** OLMo 3.1 32B Think, with OLMo 3 7B Think for denser sampling at a smaller size. Both share one recipe and release the SFT and DPO models the RL started from.
- **Domain-controlled comparison:** the seven OLMo 3 / 3.1 7B RL-Zero runs. They start from the same base and differ only in RL data domain.
- **Size within one recipe:** Tülu 3 has 8B (10 checkpoints), 70B (1) and 405B (3).
- **Checked and final-only:** MiMo-V2.6 (Pro, Flash, Distill-Qwen-9B), MiMo-7B-RL, INTELLECT-2/3/3.1, Nemotron 3 Nano, AceReason, DeepSeek-R1, QwQ-32B, Kimi-K2-Thinking, GLM-4.6, MiniMax-M1, DeepScaleR (W&B logs are public), DeepCoder, Open-Reasoner-Zero, SimpleRL-Zoo, DAPO, Skywork-OR1, SkyRL, Apertus. OLMo 3.1 32B Instruct has only its final weights, although its card mentions a step branch.

## How our MiMo-V2.6-Flash runs compare

Our one LoRA step was about 1/500 of one Xiaomi production step, in tokens and in dollars. Per trajectory, our rollouts matched theirs in kind but ran shorter, because our caps were half of theirs.

|  | Our runs (Oct 2026) | Xiaomi MiMo-V2.6-Flash RL | Ratio |
| --- | --- | --- | --- |
| Trajectories per step | 256 generated, 88 trained on (11 mixed groups) | 25,088 (1,568 × 16) | about 1/100 (generated), 1/285 (trained) |
| Tokens per step | 5.75M | 2.7–3.7B | about 1/550 |
| Rollouts per prompt | 8 | 16 | 1/2 |
| Context cap / per-turn cap | 128K / 16K | up to 1M (262K in released config) / 32K | 1/2 per turn |
| Length per trajectory | median 31 model calls, 23K generated tokens | about 110–150K tokens; about 57 turns (secondary) | shorter |
| Training | rank-64 LoRA, attention only, 1 step | full-parameter, async GRPO, router frozen, about 30 steps | — |
| Hardware and cost per step | 4× H200; about $30 to generate and $30 to train | "thousands of GPUs"; about $30K per step (derived: $0.9M over about 30 steps) | about 1/500 |

- **The per-trajectory picture is representative.** We used Xiaomi's own harnesses on real SWE and terminal tasks, so behaviors per rollout should resemble production, apart from truncation at our 16K per-turn cap.
- **The batch is not representative.** One production step has about 100× more trajectories. Reward gaps measured on 11 mixed groups are far noisier than what drives the production update.
- **Ten production-like steps at our scale would cost about $600.** Matching one real step would cost about $30K (derived).

## LoRA vs full-parameter RL

No frontier report explains why it trains all weights; they do it without comment. The likeliest reasons are that frontier batches are far past where Thinking Machines saw LoRA hold up, and that their training stacks are built for full weights. Thinking Machines' own Inkling does not say whether its RL used LoRA, and only its final weights are public.

**What ["LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/) (Thinking Machines, Sep 2025) claims:**

- "LoRA fully matches the learning performance of FullFT" for policy-gradient RL, even at rank 1. The reason is that an episode's advantage carries only "O(1) bits" of information.
- **Adapt the MLP layers**, including MoE experts. "Attention-only LoRA significantly underperforms MLP-only LoRA". This is why our attention-only MiMo-Flash adapter needed the expert fix.
- Use about 10× the full fine-tuning learning rate.
- **Its stated limit:** "LoRA is less tolerant of large batch sizes than FullFT".

**Likely reasons frontier RL stays full-parameter.** All are inferred unless marked stated.

| Reason | Evidence |
| --- | --- |
| Batches far beyond the tested regime | MiMo-V2.6 trains on 25K trajectories and about 3B tokens per step, where the blog itself warns LoRA degrades. |
| More than one bit per episode | Multi-domain, multi-grader, 100-turn agentic episodes are far richer than the single-task math runs where LoRA was tested. Tiny adapters do suffice for narrow tasks: [Tina](https://arxiv.org/abs/2504.15777), and [TinyLoRA](https://arxiv.org/abs/2602.04118) at about 91% on GSM8K with 13 parameters. |
| RL updates are sparse but nearly full-rank (stated finding) | 5–30% of parameters change, in "nearly full-rank" updates ([Mukherjee et al.](https://arxiv.org/abs/2505.11711)). Some academic RLVR benchmarks show LoRA a few points behind full fine-tuning ([2512.23165](https://arxiv.org/abs/2512.23165)). |
| LoRA's main systems benefit is small at scale (stated for Cursor) | Cursor's Composer 2 RL on Kimi K2.5 (1T) is full-parameter; its weight diffs "compress to a handful of gigabytes" ([Composer 2](https://arxiv.org/html/2603.24477)). |
| Serving and quantization support | vLLM gained fused-MoE expert LoRA only in early 2026 ([vLLM](https://vllm.ai/blog/2026-02-26-multi-lora)). Production stacks train MoE experts in FP8 or FP4 kernels written for full weights. |
| RL is one stage of a full-parameter pipeline | Pretraining, mid-training, SFT and distillation are full-parameter, so RL reuses the same trainer. |

**Who uses LoRA for RL:** [Tinker](https://tinker-docs.thinkingmachines.ai) (LoRA-only; mainly research users) and [Fireworks RFT](https://docs.fireworks.ai/fine-tuning/how-rft-works). No frontier or open-weight flagship reports LoRA for its main RL run.

**Inkling** ([announcement](https://thinkingmachines.ai/news/introducing-inkling/), 15 Jul 2026):

- **What it is:** Thinking Machines' first model trained from scratch. 975B total and 41B active parameters, 45T pretraining tokens across text, image, audio and video, 1M context, Apache-2.0 on Hugging Face.
- **Post-training:** SFT on synthetic data from open models "including Kimi K2.5", then RL "over 30M rollouts" in "two long continuous runs". **Whether that RL used LoRA is not stated.** LoRA appears only as the fine-tuning route for users, via Tinker.
- **[Inkling-Small](https://thinkingmachines.ai/news/inkling-small/)** (Jul 2026): 276B total, 12B active. Distilled from Inkling, then "agentic coding RL for two weeks" on GB300 NVL72. The method is again not stated.
- **Checkpoints:** final BF16 and NVFP4 weights only, for [Inkling](https://huggingface.co/thinkingmachines/Inkling) and Inkling-Small. No base model, intermediate RL checkpoints or adapters, per the Hugging Face API on 27 Oct 2026.

## Open-weight models with RL checkpoints

Only AI2 releases checkpoints from within an RL run (see Intermediate RL checkpoints above). Several larger models release a before-RL and an after-RL pair, which supports a two-point comparison but not dynamics.

| Model | Before RL | After RL | Within-RL checkpoints |
| --- | --- | --- | --- |
| [OLMo 3.1 32B Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think) | [Think-SFT](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT), [Think-DPO](https://huggingface.co/allenai/Olmo-3-32B-Think-DPO) | final model | **46**, every 50 steps |
| [OLMo 3 7B Think](https://huggingface.co/allenai/Olmo-3-7B-Think) and 7 RL-Zero runs | SFT, DPO, base | final models | 8–55 per run |
| [Tülu 3 405B / 70B / 8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B) | SFT, DPO | final | 3 / 1 / 10 |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (1.6T) | V4-Pro-Base | V4-Pro | none |
| DeepSeek-R1-Zero / R1 (671B) | DeepSeek-V3-Base | R1-Zero (RL only), R1 | none |
| [Nemotron 3 Ultra](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) (550B) | base | post-trained model, plus its generative reward model | none |
| [MiMo-V2.6-Pro / Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | not released | -RL (end of RL), then -MOPD (after distillation) | none; per-step metrics on a dashboard |
| [INTELLECT-3](https://arxiv.org/abs/2512.16144) (106B) | GLM-4.5-Air base | INTELLECT-3, 3.1 | none |
| Kimi K2 (1T) | K2-Base | K2-Instruct, K2-Thinking | none |

- **For behavior-over-RL studies at 32B, OLMo 3.1 Think is the only option**.
- **At 0.5–1.6T, the best you can do is a before/after pair** (DeepSeek-V4, R1-Zero, Nemotron 3 Ultra), or an after-RL/after-distillation pair (MiMo-V2.6).
- **MiMo's "-RL" and "-MOPD" pair isolates the distillation step, not RL.** Its pre-RL checkpoint is not public.

## Why MiMo-V2.6 stopped at about 30 steps

Xiaomi gives no reason. Nothing in the report or its public statements describes a plateau, collapse or hacking problem; the curves were still rising. The evidence points to a fixed, budgeted run timed to a release.

**Stated facts:**

- **Timeline.** Both runs started on 15 Sep 2026, at about 126 minutes per step for Pro ([Intelligent Living](https://www.intelligentliving.co/mimo-v2-6-rl-training/)).
  - On 17 Sep, Pro was at step 12 and Flash at step 17 ([36Kr](https://eu.36kr.com/en/p/3986962267765767)).
  - On 19 Sep, the live dashboard showed Flash stopped after 30 steps.
  - Weights shipped on 21–22 Sep, and both models "each completed 30 RL Steps" ([Woshipm](https://www.woshipm.com/?p=6468320)).
- **No total step count in the report.** The failure timeline (Fig. 12) spans 30 steps. A separate code-only run kept improving "through step 52", so Xiaomi has run longer in other settings ([report §4.3.2, §5](https://arxiv.org/html/2610.11959v1)).
- **Still improving at the end.** DeepSWE and other held-out tasks improved "despite fluctuations between checkpoints", with no discussion of saturation (§5.3).
- **Problems were fixed, not fatal.** The trainable router drifted into "a severe load-collapse problem" within about 20 steps, and freezing it fixed that (§5.4). GPU memory errors, a cluster crash, an unreachable grader and out-of-memory events all led to restarts (§5.5).
- **More training came after RL.** A multi-teacher distillation stage (MOPD2) combined domain specialists after the mixed RL. The report's final evaluation results are after it (§5.6).

**Likely explanations (inference, strongest first):**

1. **A preset budget and step count tied to a release.** Round, pre-disclosed costs ($2.6M Pro, $0.9M Flash), a livestreamed run, both models stopping at exactly 30, and release within about two days.
2. **A one-shot showcase.** "You Only RL Once" is the title of §5: one mixed run over every domain, shown live, with its environments released, rather than a push to convergence. Further gains were left to distillation and to the open-sourced recipe.
3. **Each step is large.** 30 steps are 750K trajectories and about 100B tokens. Each extra step would cost about $90K for Pro and $30K for Flash (derived from the totals).
4. **Plateau or instability:** no supporting evidence.

In compute terms, 30 steps at a staleness of 4 is only about 7–8 fully fresh policy generations. The run is better read as an early slice of a curve that was still rising than as a converged result.

## Many small steps or few large ones?

Neither wins outright. Small batches with many on-policy steps learn more per rollout and lead early. Large batches with few steps reach a higher ceiling, stay more stable and finish sooner in wall-clock time. The larger risk is not batch size but taking several gradient steps on each rollout batch.

| Source | Setting | Finding |
| --- | --- | --- |
| [ScaleRL](https://arxiv.org/html/2510.13786v1) (Meta, Oct 2025) | 8B and 17B×16 MoE, math, about 7K steps | A larger batch "is slower in training but settles at a higher asymptote": final score 0.645 at batch 2,048 vs 0.61 at 512–768, but it takes about 4× the GPU-hours to reach the halfway point. Splitting a fixed batch into more prompts or more rollouts per prompt changes little. |
| [BroRL](https://arxiv.org/html/2510.01180v1) (NVIDIA) | Continuing a 3,000-step ProRL model that had plateaued | 128 prompts × 512 rollouts beat 512 × 16 in 107 steps and 98 h, against 535 steps and 134 h. Throughput doubled; learning rate scaled by the square root of the batch increase. |
| [Li et al.](https://arxiv.org/abs/2608.29296) (2026, abstract only) | GRPO | With Adam and square-root learning-rate scaling, per-sample curves barely depend on batch size up to a limit. Larger batches win on time-to-target (up to 29%) only if the learning rate is retuned. |
| [Bansal, Kakade et al.](https://icml.cc/virtual/2026/72312) (abstract only) | GRPO, gradient-noise model | Noise within each prompt's group dominates. Reusing rollouts off-policy raises the batch size beyond which more samples stop helping. |
| [Skywork-OR1](https://arxiv.org/html/2505.22312) | 32B, 64 × 16, 2,700 steps | Batch and group size made no difference to entropy. More than one gradient step per rollout batch collapsed entropy within a few steps, and a larger batch did not prevent it. |
| [Magistral](https://arxiv.org/html/2506.10910) | GRPO, asynchronous | Performance "doesn't depend strongly on batch size" but degrades with more than 2 mini-batches per batch; batches of 1,024 or fewer were less stable. |
| [Scaling behaviors of LLM RL](https://arxiv.org/html/2509.25300) | Qwen2.5 0.5–72B | Results track total data; reusing data up to 25× costs nothing. Best group size grows with compute (4–8 low, 32 high). |
| [MiMo-V2.6 §4.1](https://arxiv.org/html/2610.11959v1) | 1,568 × 16, about 30 steps | The stated reasons are systems ones: "A large batch is good for scaling out GPUs" and "amortizes the re-prefill" when resuming partial rollouts. Staleness of 4 steps. |

**When to choose which:**

- **Small batch, many steps:** cheap rollouts, one domain, a tight budget, and runs judged early. Keep each step on-policy.
- **Large batch, few steps:** long agentic rollouts, many domains mixed in one batch, and runs taken near convergence. Async generation with 4–8 steps of staleness and importance-weight corrections makes large batches practical.
- **Either way, retune the learning rate when the batch changes** (about square-root scaling), and avoid many optimizer steps on one rollout batch.

Our MiMo-Flash LoRA step sits at the small end: 88 sequences, one step. MiMo's own run is at the extreme large end, and no public source tests whether 30 large steps leave gains unrealized that more, smaller steps would have found.

## Caveats and open questions

The numbers are reliable to an order of magnitude, not to a factor of two. Disclosure is uneven and getting thinner.

- **Disclosure is falling.** DeepSeek-V4, Qwen3.5 and 3.8, Kimi K2.x and K3, GLM-5 and MiniMax M2.5 give no RL compute figures. The fullest accounts come from AI2, Prime Intellect, NVIDIA and Xiaomi.
- **Derived totals rest on assumptions.** DeepSeek-R1-Zero's token total assumes an 8–16K mean length. MiMo-V2.6's total assumes about 30 steps; the paper shows 30 steps in its figures but never states a step count.
- **Secondary-only figures:**
  - MiMo-V2.6: turn counts and the under-6-day run length.
  - MiniMax-M2: the 100K-environment count.
  - Grok 4: RL compute about equal to pretraining.
  - Release dates for Kimi K3, LongCat-2.0, Step 5 and MiniMax M3.
- **Closed labs:** only ratios are public (o3 at about 10× o1's RL; Grok 4 at "pretraining scale"). Anthropic and Google disclose nothing quantitative.
- **Open question:** MiMo-V2.6's "training dynamics" turned out to be a per-step metrics dashboard, not checkpoints or rollouts (see Intermediate RL checkpoints above). Whether Xiaomi would share per-step rollouts or checkpoints on request is worth asking.

## Sources

Primary sources (technical reports, model cards, official repositories):

- [MiMo-V2.6 technical report (arXiv 2610.11959)](https://arxiv.org/html/2610.11959v1) and the [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) and [Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) model cards
- [Xiaomi verl mimo-oss RL config](https://raw.githubusercontent.com/XiaomiMiMo/verl/mimo-oss/scripts/code/train.sh)
- [MiMo-V2-Flash report (arXiv 2601.02780)](https://arxiv.org/abs/2601.02780)
- [DeepSeek-R1 (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948), [DeepSeek-V3.2 (arXiv 2512.02556)](https://arxiv.org/abs/2512.02556), [DeepSeek-V4-Pro card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [OLMo 3 report (arXiv 2512.13961)](https://arxiv.org/abs/2512.13961), [OLMo 3 blog](https://allenai.org/blog/olmo3), [Dolci-Think-RL data](https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B)
- [INTELLECT-3 report (arXiv 2512.16144)](https://arxiv.org/abs/2512.16144)
- [Nemotron 3 Ultra report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf), [Nemotron 3 Nano (arXiv 2512.20848)](https://arxiv.org/abs/2512.20848), [Nemotron 3 Super RL recipe](https://docs.nvidia.com/nemo/rl/latest/guides/nemotron-3-super.html)
- [MiniMax-M1 (arXiv 2506.13585)](https://arxiv.org/abs/2506.13585), [MiniMax-M1 announcement](https://minimax.io/news/minimaxm1), [MiniMax-M2.5 repo](https://github.com/MiniMax-AI/MiniMax-M2.5)
- [GLM-4.5 (arXiv 2508.06471)](https://arxiv.org/abs/2508.06471), [GLM-5 (arXiv 2602.15763)](https://arxiv.org/abs/2602.15763)
- [Kimi K2 (arXiv 2507.20534)](https://arxiv.org/abs/2507.20534), [Kimi K3 card](https://huggingface.co/moonshotai/Kimi-K3)
- [LongCat-Flash-Thinking (arXiv 2509.18883)](https://arxiv.org/abs/2509.18883), [LongCat-Flash-Thinking-2601 (arXiv 2601.16725)](https://arxiv.org/abs/2601.16725)
- [Ring-1T (arXiv 2510.18855)](https://arxiv.org/abs/2510.18855), [Qwen3 (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388), [Qwen3-Coder blog](https://qwenlm.github.io/blog/qwen3-coder/), [Magistral (arXiv 2506.10910)](https://arxiv.org/abs/2506.10910), [gpt-oss model card (arXiv 2508.10925)](https://arxiv.org/abs/2508.10925)
- [Grok 4 announcement](https://x.ai/news/grok-4)

Analysis and secondary sources:

- [Epoch AI: Grok 4 training resources](https://epoch.ai/data-insights/grok-4-training-resources), [Epoch AI: GPT-5 training compute](https://epochai.substack.com/p/notes-on-gpt-5-training-compute), [Epoch AI: state of RL environments](https://epoch.ai/gradient-updates/state-of-rl-envs)
- [The Register on DeepSeek-R1 cost](https://www.theregister.com/2025/09/19/deepseek_cost_train/), [TechCrunch on RL environments](https://techcrunch.com/2025/09/16/silicon-valley-bets-big-on-environments-to-train-ai-agents/)
- [SWE-rebench V2](https://nebius.com/blog/posts/meet-swe-rebench-v2), [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)


---

*Compiled in October 2026 with research help from Claude: parallel literature searches, Hugging Face checks of checkpoint branches, and our own MiMo-V2.6-Flash and OLMo 3 runs for calibration. Numbers marked "derived" are my arithmetic; "secondary" means not confirmed against a primary source.*
