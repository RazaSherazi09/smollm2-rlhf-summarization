# Learning to Summarize from Human Feedback: End-to-End Alignment of SmolLM2-135M

**Course:** Generative AI — Semester 7  
**Author:** Syed Muhammad Raza  
**Repository:** [smollm2-rlhf-summarization](https://github.com/Razaasherazi09/smollm2-rlhf-summarization)  
**Compute Infrastructure:** Modal Labs — NVIDIA A10G (24 GB VRAM)  
**Base Model:** [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)  
**Datasets:** [`CarperAI/openai_summarize_tldr`](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr) and [`CarperAI/openai_summarize_comparisons`](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons)

---

## Abstract

This project investigates preference-based alignment for short-form Reddit post summarization using a compact decoder-only language model, SmolLM2-135M. The experimental pipeline compares a supervised fine-tuning (SFT) baseline, a Bradley–Terry reward model, online Proximal Policy Optimization (PPO), and Direct Preference Optimization (DPO). SFT learns from prompt–summary examples; the reward model learns to rank preferred and rejected summaries; PPO optimizes a policy against the learned reward while regularizing it toward a reference policy; and DPO learns directly from paired preferences without using an explicit reward model during policy optimization.

On the reported set of 300 held-out posts, SFT obtains the highest ROUGE scores, while DPO obtains the highest mean score under the trained reward model and a 96.33% reward-model pairwise win rate over SFT. PPO shows a larger reduction in ROUGE and a tendency to generate outputs at the decoding length limit. These findings illustrate that lexical overlap and a learned preference proxy measure different aspects of summarization quality. The reward-model results should be interpreted as model-based preference estimates rather than as independent human evaluations.

## 1. Project Overview

### 1.1 Problem Statement

The goal is to transform a Reddit post into a concise `TL;DR` summary that preserves the central facts, important context, and the author's intent. Standard supervised learning can imitate reference summaries, but it does not directly optimize for pairwise preferences between alternative summaries. This project therefore studies how preference learning and alignment methods change the behavior of a small language model.

### 1.2 Objectives

1. Fine-tune SmolLM2-135M on prompt–summary examples.
2. Train a scalar reward model from chosen/rejected summary pairs.
3. Apply PPO using the reward model and a reference-policy KL penalty.
4. Train a DPO policy directly from preference pairs.
5. Compare the base model, SFT, PPO, and DPO using ROUGE, generated length, learned reward scores, and qualitative examples.
6. Examine reward-model errors and identify possible optimization pathologies, including length bias and reward exploitation.

### 1.3 End-to-End Pipeline

```text
SmolLM2-135M
     |
     v
Supervised Fine-Tuning (SFT)
     |-------------------------------|
     v                               v
SFT policy                       SFT reference policy
     |                               |
     v                               |
Reward Model (Bradley–Terry)         |
     |                               |
     +-----------> PPO <--------------+
     |                |
     |                v
     |           PPO policy
     |
     +-----------> DPO <-------------- SFT reference policy
                      |
                      v
                 DPO policy
                      |
                      v
       Evaluation on 300 held-out posts
```

The reward model is used to score PPO rollouts and to calculate the reported reward-based evaluation metrics. DPO training itself uses paired preferences and a reference policy; it does not require the explicit reward model as a training input.

## 2. Data and Experimental Setup

### 2.1 Datasets

The project uses two datasets:

- **Summarization dataset — `CarperAI/openai_summarize_tldr`:** Reddit post prompts paired with human-written TL;DR summaries. The supplied experiment description reports 20,000 prompt–summary examples for SFT.
- **Preference dataset — `CarperAI/openai_summarize_comparisons`:** Prompts accompanied by preferred (`chosen`) and less-preferred (`rejected`) summaries. The supplied experiment description reports 20,000 pairwise examples for reward modeling and preference training.

The reported evaluation set contains 300 held-out posts. The reward-model validation set contains 1,000 held-out preference pairs.

> **Reproducibility note:** The figures above are the counts stated in the supplied experiment summary. Before submission, confirm the exact dataset revisions, split names, filtering rules, deduplication procedure, and train/validation/test boundaries from the data-loading code and saved logs. In particular, ensure that evaluation prompts and preference pairs do not overlap with training data.

### 2.2 Model and Compute

| Item | Configuration |
|---|---|
| Base language model | `HuggingFaceTB/SmolLM2-135M` |
| Model size | Approximately 134.5 million parameters (as reported) |
| Compute platform | Modal Labs serverless GPU infrastructure |
| GPU | NVIDIA A10G |
| GPU memory | 24 GB |
| Optimizer | AdamW |
| Mixed precision | BF16 autocast with FP32 master parameters/optimizer state (as reported) |

### 2.3 Evaluation Protocol

All four model variants are reported as evaluated on the same 300 held-out posts using deterministic greedy decoding:

- `max_new_tokens = 48`
- `temperature = 0.0` (greedy decoding; temperature is effectively irrelevant when sampling is disabled)
- Stop at ` <|endoftext|> ` when that token is emitted

ROUGE-1, ROUGE-2, and ROUGE-L are reported as percentages. Mean generated length is measured in tokens. Mean reward is calculated using the trained reward model, denoted by \(r_\phi\).

> **Evaluation interpretation:** ROUGE measures reference overlap, whereas \(r_\phi\) measures the output according to the learned reward model. Neither metric alone establishes human-perceived summary quality. A reward-model score higher than the score assigned to a human reference is not proof that the generated summary is better than the human summary; it may also reflect reward-model miscalibration or exploitable artifacts.

## 3. Methodology

The stages below follow the actual dependency order of the experiment: SFT provides the initial summarization policy; the reward model is initialized from SFT; PPO uses both the SFT reference policy and reward model; and DPO learns from preference pairs relative to the SFT reference policy.

### 3.1 Stage 1 — Supervised Fine-Tuning (SFT)

The base model is fine-tuned to generate a summary \(y=(y_1,\ldots,y_T)\) conditioned on a post \(x=(x_1,\ldots,x_P)\). Training minimizes causal language-model cross-entropy over summary tokens only:

\[
\mathcal{L}_{\mathrm{SFT}}(\theta)
= -\frac{1}{|y|}\sum_{t=1}^{T}
\log \pi_\theta(y_t\mid x,y_{<t}).
\]

Prompt positions and right-padding positions are excluded from the loss. The target sequence is shifted so that each prediction is conditioned only on preceding tokens. The resulting SFT checkpoint is used as the initialization for the reward model, PPO policy, PPO reference policy, and DPO policy/reference setup.

### 3.2 Stage 2 — Bradley–Terry Reward Modeling

The reward model is initialized from the transformer body of the SFT checkpoint and augmented with a scalar linear reward head. Let \(h_T\) denote the final hidden representation at the last non-padding token and let \(W_v\in\mathbb{R}^{H\times 1}\) be the projection weights. The scalar reward is:

\[
r_\phi(x,y)=W_v^\top h_T.
\]

For each prompt, the dataset supplies a preferred summary \(y_w\) and a rejected summary \(y_l\). Under the Bradley–Terry model, the probability that the preferred response is ranked above the rejected response is:

\[
P(y_w\succ y_l\mid x)
=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big).
\]

The reward model is trained by minimizing the pairwise negative log-likelihood:

\[
\mathcal{L}_{\mathrm{RM}}(\phi)
=-\mathbb{E}_{(x,y_w,y_l)}
\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right].
\]

The trained reward model supplies scalar feedback to PPO and is also used for the reward-based comparisons reported in the evaluation section.

### 3.3 Stage 3 — Proximal Policy Optimization (PPO)

PPO performs online policy optimization using generated summaries and reward-model feedback. The experiment keeps the following model components available:

- **Trainable policy \(\pi_\theta\):** Generates summaries and is updated by PPO.
- **Critic \(V_\psi\):** Estimates state values for advantage calculation.
- **Frozen reference policy \(\pi_{\mathrm{ref}}\):** The SFT policy, used to penalize excessive policy drift.
- **Frozen reward model \(r_\phi\):** Scores completed summaries.

#### 3.3.1 Token-Level KL-Penalized Reward

For generated token \(a_t\) at state \(s_t\), the per-token reward is defined as:

\[
r_t=-\beta_{\mathrm{KL}}
\left[
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{ref}}(a_t\mid s_t)
\right]
+\mathbf{1}[t=T]r_\phi(x,y).
\]

The KL-related term penalizes divergence from the reference policy, while the terminal reward-model score is added at the final generated token. With discount factor \(\gamma=1\), returns are accumulated without discounting:

\[
R_t=\sum_{k\geq 0}r_{t+k}.
\]

Advantages are computed using the old critic estimate and normalized over valid generated-token positions:

\[
A_t=R_t-V_{\psi}^{\mathrm{old}}(s_t),
\qquad
\widehat A_t=\frac{A_t-\mu_A}{\sigma_A}.
\]

#### 3.3.2 Clipped PPO Objective

The policy probability ratio is:

\[
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
\]

The clipped surrogate objective is:

\[
L_{\mathrm{PPO}}(\theta)=\widehat{\mathbb E}_t
\left[
\min\left(
\rho_t(\theta)\widehat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
\]

The combined optimization objective includes the policy loss, value-function loss, and entropy bonus:

\[
\mathcal L(\theta,\psi)=
-L_{\mathrm{PPO}}(\theta)
+c_V\widehat{\mathbb E}_t[(V_\psi(s_t)-R_t)^2]
-c_H\widehat{\mathbb E}_t[\mathcal H(\pi_\theta(\cdot\mid s_t))].
\]

The supplied configuration uses four optimization epochs per rollout. The clipping term limits the size of policy updates, while the value and entropy terms support critic learning and policy exploration.

### 3.4 Stage 4 — Direct Preference Optimization (DPO)

DPO trains a policy directly from preferred and rejected completions, using the SFT checkpoint as the reference policy. It avoids running an online RL loop and does not require an explicit reward model during policy training.

For a preference pair \((x,y_w,y_l)\), the DPO objective is:

\[
\mathcal{L}_{\mathrm{DPO}}(\theta;\pi_{\mathrm{ref}})
=-\mathbb E_{(x,y_w,y_l)}\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)\right].
\]

Here, \(\beta=0.1\) controls the strength of the reference-policy constraint. Sequence log-probabilities are obtained by summing token log-probabilities over valid completion positions; prompt and padding positions are excluded. The supplied implementation description notes vectorized log-probability calculations using log-sum-exp operations.

## 4. Training Configuration and Runtime

### 4.1 Hyperparameters

| Hyperparameter | SFT | Reward Model | PPO | DPO |
|---|---:|---:|---:|---:|
| Initialization | SmolLM2-135M | SFT checkpoint | SFT checkpoint | SFT checkpoint |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Learning rate | \(5\times10^{-5}\) | \(5\times10^{-5}\) | \(1\times10^{-5}\) | \(2\times10^{-6}\) |
| Batch size | 8 | 8 | 16 (rollout) | 8 |
| Training schedule | 1 epoch; 2,500 steps | 1 epoch; 2,500 steps | 200 iterations; 3,200 prompts | 1 epoch; 2,500 steps |
| Precision | FP32 master / BF16 AMP | FP32 master / BF16 AMP | FP32 master / BF16 AMP | FP32 master / BF16 AMP |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 |
| Additional settings | — | — | \(\epsilon=0.2\), \(c_V=0.1\), \(c_H=0.01\), \(\beta_{KL}=0.05\) | \(\beta=0.1\) |
| Wall-clock time | 1,124.6 s | 1,032.6 s | 662.5 s | 1,811.7 s |

> **Configuration check:** The reported dataset sizes, batch sizes, steps, and epoch counts should be checked against the actual dataloader settings. For example, 20,000 examples with batch size 8 do not automatically imply 2,500 optimizer steps if `drop_last`, gradient accumulation, distributed execution, or filtering changes the number of batches. Record the effective number of optimizer updates in the final experiment logs.

### 4.2 Reported Runtime

| Stage | Wall-clock time | Approximate time |
|---|---:|---:|
| SFT | 1,124.6 s | 18.7 min |
| Reward modeling | 1,032.6 s | 17.2 min |
| PPO | 662.5 s | 11.0 min |
| DPO | 1,811.7 s | 30.2 min |
| **Total** | **4,631.4 s** | **77.2 min** |

These are the execution times supplied for the experiment. They may not include data download, environment startup, checkpoint transfer, evaluation, or other orchestration overhead unless those were included in the recorded timers.

## 5. Quantitative Evaluation

### 5.1 Automatic Metrics

The following table reports the supplied measurements on the 300 held-out posts.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean length (tokens) | Mean reward \(r_\phi\) |
|---|---:|---:|---:|---:|---:|
| Base SmolLM2-135M | 18.68 | 4.77 | 15.47 | 48.0 | -1.396 |
| **SFT** | **32.04** | **11.93** | **25.42** | **30.4** | +0.352 |
| PPO | 23.59 | 5.29 | 17.54 | 48.0 | +0.513 |
| DPO | 31.19 | 10.35 | 22.88 | 45.9 | **+2.655** |
| Human reference | 100.0* | 100.0* | 100.0* | 32.8 | +1.008 |

\*The human-reference row is shown as a self-reference ceiling for ROUGE and should not be interpreted as a model competing against itself. Its reward score is the reward model's score for reference summaries, not a human preference rating.

### 5.2 Pairwise Win Rates Under the Reward Model

Each cell represents the percentage of pairwise comparisons in which the row candidate receives a higher reward-model score than the column candidate, according to the supplied results.

| Candidate (row) vs. opponent (column) | Base | SFT | PPO | DPO |
|---|---:|---:|---:|---:|
| Base | 50.00% | 12.00% | 11.33% | 0.67% |
| SFT | 87.00% | 50.00% | 44.67% | 3.33% |
| PPO | 88.67% | 55.33% | 50.00% | 4.67% |
| DPO | **99.33%** | **96.33%** | **95.33%** | 50.00% |

These figures are **reward-model win rates**, not independently collected human win rates. They quantify how often the reward model ranks one candidate above another over the evaluated prompts. The interpretation is affected by reward-model accuracy, calibration, and possible reward hacking.

## 6. Analysis and Discussion

### 6.1 Reward-Model Validation and Error Analysis

The supplied validation accuracy on 1,000 held-out preference pairs is **63.60%**. For misclassified examples where

\[
\Delta r=r_\phi(x,y_w)-r_\phi(x,y_l)<0,
\]

the report describes several recurring patterns.

1. **Length and completeness bias.** Some rejected summaries were longer and ended with more grammatically complete clauses. The reward model may therefore have learned superficial correlations between style, length, and the chosen label instead of consistently representing summary faithfulness and informativeness. The cited example reward differences are -1.0156, -0.3750, and -0.9844.
2. **Long-context difficulty.** Errors were reported more frequently for prompts in the 321–400+ token range. A reward prediction based on the final non-padding token representation may be less reliable when the model must preserve details from early in a long prompt. This is a plausible explanation, but it should be supported with a length-bucketed error analysis before being stated as a demonstrated cause.
3. **Preference-label ambiguity.** Reddit summary preferences can be subjective. However, the claim that the dataset has 35–40% inter-annotator noise, or that 63.60% is near a Bayes-optimal ceiling, is not established by validation accuracy alone. Such a conclusion would require repeated annotations or an independent estimate of annotator agreement and label noise.

### 6.2 SFT: Strong Reference Overlap

SFT produces the highest ROUGE scores in the reported comparison: ROUGE-1 of 32.04, ROUGE-2 of 11.93, and ROUGE-L of 25.42. This is consistent with its training objective: maximum-likelihood learning encourages the model to assign high probability to the reference-summary token sequences. SFT also produces the shortest outputs among the generated models in this experiment, with a mean length of 30.4 tokens, close to the human-reference mean of 32.8 tokens.

### 6.3 PPO: Optimization Instability and Length Saturation

PPO improves the mean reward score from +0.352 for SFT to +0.513, but its ROUGE scores are lower than SFT's (ROUGE-1 falls from 32.04 to 23.59). Its mean generated length reaches the 48-token decoding limit.

The training notes report that rollout length increased from 33.9 to 48.0 tokens and that some rollout reward values reached approximately +8.5. Together, these observations are consistent with possible reward exploitation: the policy may have discovered output patterns that receive high scores from the learned reward model without producing better reference-aligned summaries. The length limit may also be contributing to the measured mean length. This is a warning sign rather than conclusive proof of reward hacking; a stronger diagnosis would compare reward, length, factuality, and blinded human judgments across checkpoints.

The reported clip fraction remained between 2.5% and 13.6%, while value loss decreased from 12.53 to below 1.0. These values suggest that the PPO updates and critic loss remained numerically controlled under the selected settings, but they do not by themselves establish that the learned policy is behaviorally stable or that its outputs are higher quality.

### 6.4 DPO: Strong Reward-Model Preference Scores

DPO reaches the highest mean reward-model score (+2.655) and is ranked above SFT on 96.33% of the reported pairwise reward-model comparisons. Its ROUGE scores (31.19 / 10.35 / 22.88) remain relatively close to SFT, although all three are lower.

This result suggests that, for this setup, direct optimization on paired preferences produced outputs that the learned reward model strongly preferred while retaining much of SFT's reference overlap. DPO also avoids the online rollout and critic optimization required by PPO, simplifying the policy-training procedure. Nevertheless, the unusually high reward-model win rate should be validated with independent human evaluation because DPO can learn the reward model's or preference dataset's biases indirectly through the preference labels.

### 6.5 ROUGE and Learned Preference Are Different Measures

The experiment exhibits a clear distinction between lexical reference overlap and reward-model scoring:

- **SFT** is strongest on ROUGE and generates concise summaries.
- **PPO** raises the reported mean reward slightly over SFT but has noticeably lower ROUGE and reaches the output-length ceiling.
- **DPO** has the strongest reward-model scores while retaining ROUGE values relatively close to SFT.

It is therefore more accurate to conclude that SFT best matches the reference summaries under ROUGE, while DPO is most preferred by the trained reward model under the reported proxy evaluation. The present measurements do not prove that DPO is preferred by humans unless a separate human evaluation was conducted.

## 7. Qualitative Comparison

The following example is reproduced from the supplied held-out qualitative comparison.

**Prompt (tail):** A user explains that their girlfriend of three years unexpectedly ended the relationship to find herself and become independent. The user feels devastated and cannot focus on university finals, and asks for advice.

**Human reference**  
> Girlfriend of 3 years dumped me unexpectedly to find herself. Devastated and failing to study for finals. Need coping advice.  
> Reward: +1.284

**Base model**  
> I have been feeling really down lately and I don't know what to do anymore. I am asking for advice on reddit because my friends are busy...  
> Reward: -1.842

**SFT**  
> GF of 3 years dumped me out of nowhere; I'm heartbroken and can't focus on finals.  
> Reward: +0.412

**PPO**  
> Girlfriend broke up with me after three years because she wants to be independent, and now I am completely devastated and unable to study for my upcoming university finals.  
> Reward: +1.640

**DPO**  
> GF of 3 years abruptly dumped me to 'find herself'. I'm devastated and struggling to focus on my upcoming finals; looking for advice on how to cope.  
> Reward: +3.105

### Qualitative Observations

- **Base model:** Does not reliably follow the TL;DR instruction and produces generic conversational text rather than a faithful summary.
- **SFT:** Captures the main event and its impact concisely, closely matching the brevity of the reference.
- **PPO:** Preserves the main facts and is fluent, but is more verbose than the reference.
- **DPO:** Preserves the relationship duration, the quoted reason for the breakup, the emotional impact, and the difficulty studying. It also makes the request for coping advice explicit.

This is one example and cannot establish overall model quality. A broader qualitative study should sample outputs across short and long prompts, different topics, and cases where the models disagree.

## 8. Limitations and Threats to Validity

1. **Small evaluation set.** The reported automatic evaluation uses 300 posts. Results may vary with a larger or differently sampled test set.
2. **Reward-model dependence.** DPO's reported reward advantage and pairwise win rates are measured by the same learned reward model used in the pipeline. This creates a risk of circular evaluation and reward-model exploitation.
3. **No confirmed independent human preference test.** The supplied results do not include blinded human ratings or a human pairwise comparison of model outputs. Reward-model win rates must not be labeled as human win rates.
4. **ROUGE limitations.** ROUGE captures lexical overlap and may undervalue valid paraphrases or semantically faithful summaries with different wording.
5. **Length cap effects.** The 48-token generation cap may truncate outputs or cause models to accumulate at the maximum length. Length statistics should be interpreted alongside termination and truncation rates.
6. **Reward-model generalization.** A 63.60% validation accuracy indicates imperfect preference prediction. The reward model may encode stylistic shortcuts and may not reliably detect factual errors.
7. **Dataset and split reproducibility.** Exact revisions, preprocessing, filtering, and split construction must be recorded to rule out overlap and data leakage.
8. **Compute and seed variability.** The supplied summary does not specify random seeds, repeated runs, confidence intervals, or variance across training runs. Small metric differences may not be statistically meaningful.

## 9. Conclusion

This project implements and compares four stages of summarization alignment using SmolLM2-135M: SFT, Bradley–Terry reward modeling, PPO, and DPO. SFT provides the strongest ROUGE performance and concise outputs. PPO demonstrates the difficulty of online reward optimization in a compact model, with lower reference overlap and generation lengths saturating at the configured maximum. DPO achieves the strongest scores under the learned reward model and retains ROUGE performance relatively close to SFT.

The main conclusion is that the alignment method changes what the model optimizes: SFT emphasizes imitation of reference summaries, whereas preference-based methods shift behavior toward patterns favored by the preference signal. In this experiment, DPO provides the strongest **reward-model-based** preference result, while SFT remains the strongest **ROUGE-based** model. An independent human evaluation is necessary before claiming that DPO produces summaries that people prefer overall.

## 10. Reproducibility and Repository Artifacts

The supplied repository layout is organized by training stage and evaluation output:

```text
.
├── checkpoints/
│   ├── sft/                 # SFT model and tokenizer
│   ├── reward/              # Reward model weights (reward_model.pt)
│   ├── ppo/                 # PPO policy checkpoint (without value head)
│   └── dpo/                 # DPO policy checkpoint
├── results/
│   ├── sft/                 # SFT loss plot and metrics
│   ├── reward/              # Reward loss/accuracy plots, metrics, error samples
│   ├── ppo/                 # PPO training curves and metrics
│   ├── dpo/                 # DPO loss/accuracy plots and metrics
│   └── eval/                # Aggregate metrics and qualitative comparisons
├── train_sft.py             # Stage 1: supervised fine-tuning
├── train_reward.py          # Stage 2: reward-model training
├── train_ppo.py             # Stage 3: PPO training
├── train_dpo.py             # Stage 4: DPO training
├── evaluate.py              # Evaluation and benchmark suite
├── modal_runner.py          # Modal execution/orchestration harness
└── ANALYSIS.md              # Consolidated project report
```

### 10.1 Recommended Reproducibility Checklist

Before final submission, ensure the repository contains or documents:

- Dataset identifiers, revisions, split names, and preprocessing scripts.
- Model and tokenizer identifiers, configuration, and checkpoint provenance.
- Random seeds and the exact effective optimizer-step counts.
- Full training logs and saved hyperparameter configurations for each stage.
- Evaluation script version, ROUGE implementation and aggregation settings.
- Output termination/truncation statistics and the exact decoding configuration.
- Reward-model validation predictions and representative false-positive/false-negative pairs.
- Clear separation between reward-model metrics and any independently collected human judgments.

---
