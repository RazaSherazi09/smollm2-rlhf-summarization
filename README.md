# SmolLM2-135M: RLHF & DPO for Reddit Summarization

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Compute-Modal%20%7C%20NVIDIA%20A10G-6C5CE7)](https://modal.com/)
[![Hugging Face](https://img.shields.io/badge/Checkpoints-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/razaasherazi/smollm2-135m-rlhf-checkpoints)

An end-to-end preference-alignment project that adapts [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) for Reddit TL;DR summarization. The project implements and compares **Supervised Fine-Tuning (SFT)**, **Bradley–Terry Reward Modeling**, **Proximal Policy Optimization (PPO)**, and **Direct Preference Optimization (DPO)**.

> **Reproducibility and interpretation:** The metrics and correctness-check outcomes below are reported project results. They should be considered verified only after checking the committed implementation, raw result files, and execution logs. Reward-model scores are learned proxy signals; they are not human ratings.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Alignment Pipeline](#alignment-pipeline)
- [Datasets and Infrastructure](#datasets-and-infrastructure)
- [Methodology](#methodology)
- [Results](#results)
  - [Dataset Statistics](#dataset-statistics)
  - [Training Summary](#training-summary)
  - [Automatic Evaluation](#automatic-evaluation)
  - [Reward-Model Pairwise Comparisons](#reward-model-pairwise-comparisons)
- [Human Evaluation](#human-evaluation)
- [Implementation Correctness Checks](#implementation-correctness-checks)
- [Interactive Demo](#interactive-demo)
- [Repository Structure](#repository-structure)
- [Reproducing the Project](#reproducing-the-project)
- [Checkpoints and Deliverables](#checkpoints-and-deliverables)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

| Item | Description |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M` |
| Model size | Approximately 134.5 million parameters |
| Task | Reddit post → concise TL;DR summary |
| Supervised dataset | `CarperAI/openai_summarize_tldr` |
| Preference dataset | `CarperAI/openai_summarize_comparisons` |
| Alignment methods | SFT, reward modeling, PPO, and DPO |
| Reported compute | Modal serverless GPU, NVIDIA A10G (24 GB VRAM) |
| Author | Syed Muhammad Raza |
| Course | Generative AI — Semester 7 |

### Project Objectives

1. Adapt a compact decoder-only language model to instruction-conditioned summarization.
2. Learn a scalar reward function from chosen/rejected summary pairs using the Bradley–Terry preference model.
3. Study online policy optimization with PPO and reference-policy KL regularization.
4. Train a preference-aligned policy directly from offline comparisons using DPO.
5. Compare model outputs using ROUGE, reward-model scores, and a small blinded human evaluation.

---

## Alignment Pipeline

```text
                         Reddit TL;DR data
                                |
                                v
                       SmolLM2-135M base
                                |
                                v
                         SFT policy
                         /        \
                        /          \
                       v            v
                Reward model    DPO training
                       |        (preference pairs)
                       v
                 PPO training
                       |
                       v
                   PPO policy

              Base / SFT / PPO / DPO
                       |
                       v
          Automatic and human evaluation
```

The two preference-alignment paths are distinct:

- PPO path: SFT policy → reward model → PPO policy.
- DPO path: SFT reference policy + preference pairs → DPO policy.

## Datasets and Infrastructure

### Datasets

- `CarperAI/openai_summarize_tldr` — Reddit posts paired with TL;DR summaries for supervised fine-tuning and evaluation.
- `CarperAI/openai_summarize_comparisons` — chosen/rejected response comparisons for preference learning and reward modeling.

The project notes report 20,000 supervised examples and 20,000 preference comparisons for training. Exact dataset revisions, filtering rules, split construction, and random seeds should be recorded alongside the code for fully reproducible runs.

### Compute

Training was reported as orchestrated on [Modal](https://modal.com/?utm_source=chatgpt.com) using an NVIDIA A10G GPU with 24 GB VRAM. The reported precision setup used BF16 autocast with FP32 master parameters.

## Methodology

### 1. Supervised Fine-Tuning (SFT)

The base model is fine-tuned on Reddit post–summary examples using causal language-model cross-entropy. The loss is applied to target summary tokens; prompt and padding positions are masked so they do not contribute to the supervised objective.

### 2. Bradley–Terry Reward Modeling

The reward model is initialized from the SFT transformer and extended with a scalar reward head. Given a prompt $x$, a preferred response $y_w$, and a rejected response $y_l$, the model is trained to assign a higher score to the preferred response:

$$
\mathcal{L}_{\mathrm{RM}} = -\mathbb{E}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right].
$$

Here, $r_\phi$ is the learned scalar reward function and $\sigma$ is the sigmoid function.

### 3. Proximal Policy Optimization (PPO)

PPO optimizes a trainable policy using sampled responses and reward-model feedback. The implementation uses a value function and frozen reference-policy/reward-model copies. A token-level KL penalty regularizes the policy against the reference policy to discourage excessive drift.

Reported PPO coefficients:

| Parameter | Value |
|---|---:|
| Clip range $\epsilon$ | 0.2 |
| Value coefficient $c_V$ | 0.1 |
| Entropy coefficient $c_H$ | 0.01 |
| KL coefficient $\beta_{KL}$ | 0.05 |

### 4. Direct Preference Optimization (DPO)

DPO optimizes the policy directly from chosen/rejected response pairs relative to the SFT reference policy. Unlike the PPO path, DPO does not require a separately trained reward model or value network during its optimization objective.

The reported DPO temperature parameter is $\beta=0.1$.

## Results

The tables below reproduce the project-reported values. These values should be cross-checked against the corresponding result artifacts before being treated as independently reproduced.

### Dataset Statistics

| Metric | Reported value |
|---|---:|
| SFT training examples | 20,000 |
| SFT validation examples | 2,000 |
| Preference training pairs | 20,000 |
| Preference validation pairs | 1,000 |
| Held-out evaluation posts | 300 |
| Mean SFT prompt length | 196.4 tokens |
| Maximum SFT prompt length | 448 tokens |
| Mean SFT summary length | 26.8 tokens |
| Maximum SFT summary length | 48 tokens |

### Training Summary

| Stage | Final training loss | Validation / reported metric | GPU time |
|---|---:|---|---:|
| SFT | 1.8540 | Validation loss: 1.9124 | 1,124.6 s |
| Reward model | 0.5812 | Pairwise accuracy: 63.60% | 1,032.6 s |
| PPO | 0.0814 | Final mean reward: +8.520 | 662.5 s |
| DPO | 0.6312 | Implicit reward accuracy: 60.50% | 1,811.7 s |

### Automatic Evaluation

Reported evaluation used 300 held-out posts and deterministic greedy decoding with `max_new_tokens=48`, `temperature=0.0`, and stopping at `<|endoftext|>`.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean tokens | KL to SFT | Mean reward $r_\phi$ |
|---|---:|---:|---:|---:|---:|---:|
| Base SmolLM2-135M | 18.68 | 4.77 | 15.47 | 48.0 | N/A | -1.396 |
| SFT | 32.04 | 11.93 | 25.42 | 30.4 | 0.00 | +0.352 |
| PPO | 23.59 | 5.29 | 17.54 | 48.0 | 1.82 | +0.513 |
| DPO | 31.19 | 10.35 | 22.88 | 45.9 | 0.78 | +2.655 |
| Human reference | 100.00* | 100.00* | 100.00* | 32.8 | — | +1.008 |

*The human-reference ROUGE values are a reference self-comparison, not a model score or a meaningful comparison against generated summaries.*

Reading the results: SFT achieved the strongest reported ROUGE scores among the learned policies. DPO achieved the highest reported mean reward-model score and performed strongly in the reported human preference exercise. PPO's average output length reached the 48-token generation limit; this may indicate a length-related reward incentive and should be interpreted alongside qualitative samples.

### Reward-Model Pairwise Comparisons

Each cell is the percentage of evaluation posts for which the row model received a higher reward-model score than the column model.

| Candidate | Base | SFT | PPO | DPO |
|---|---:|---:|---:|---:|
| Base | 50.00% | 12.00% | 11.33% | 0.67% |
| SFT | 87.00% | 50.00% | 44.67% | 3.33% |
| PPO | 88.67% | 55.33% | 50.00% | 4.67% |
| DPO | 99.33% | 96.33% | 95.33% | 50.00% |

These are reward-model comparisons, not human pairwise win rates. A learned reward model can encode biases or be exploited by a policy; a higher score alone does not establish better factuality, usefulness, or human preference.

## Human Evaluation

The project notes report a blinded comparison on 30 held-out posts. SFT, PPO, and DPO outputs were presented using randomized anonymous labels.

| Evaluator | DPO selected | PPO selected | SFT selected |
|---|---:|---:|---:|
| Member 1 | 24 (80.0%) | 5 (16.7%) | 1 (3.3%) |
| Member 2 | 20 (66.7%) | 4 (13.3%) | 6 (20.0%) |

Reported agreement between evaluators: 23/30 (76.7%).

The raw annotations are intended to be stored in `results/human_eval.csv`. For a transparent evaluation record, include the evaluator instructions, randomization procedure, tie/abstention handling, and confirmation that both evaluators rated the same examples. The sample is small and should not be presented as statistically conclusive.

## Implementation Correctness Checks

The following six invariants are reported as checked in the project notes. These outcomes are not independently verified by this README; retain executable tests and genuine run logs in the repository as supporting evidence.

| # | Invariant | Reported outcome |
|---:|---|---|
| 1 | SFT summary mask: decoding positions selected by `summary_mask == 1` recovers the target summary plus EOS, without prompt leakage. | Exact token match reported. |
| 2 | Frozen models: reference policy and reward-model parameters remain frozen during the relevant PPO/DPO stages. | `requires_grad=False` reported; tensor counts 272 and 273 for the PPO reference and reward model. |
| 3 | PPO initial KL: identical initial policy/reference weights produce approximately zero sampled-action divergence. | Initial mean KL reported as 0.000000 nats. |
| 4 | PPO first optimization epoch: before the first optimizer update, policy ratios are approximately 1 and clip fraction is 0. | Ratio reported in `[0.999999, 1.000001]`; clip fraction 0.00%. |
| 5 | DPO initial loss: when policy and reference are identical, the preference margin is zero and the loss is $\ln 2 \approx 0.693147$. | Initial loss reported as 0.693147. |
| 6 | Reward pooling: pool from the final valid token using `attention_mask`, including left-padded batches. | Attention-mask-based indexing reported. |

Recommended evidence: add runnable tests (for example, `tests/test_correctness.py`) that exercise actual batches and model parameters. Capture the real test command and unedited output; do not replace runtime checks with hard-coded expected values.

## Interactive Demo

A Streamlit interface is included at `app/app.py` for comparing SFT, PPO, and DPO generations and displaying corresponding reward-model scores, if the app and required checkpoints are present in the repository.

### Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app/app.py
```

If available, the interface screenshot is stored at `app/screenshot.png`.

> Reward-model scores shown in the interface should be labeled as learned proxy scores, not human ratings.

## Repository Structure

The following is the project layout described in the supplied project documentation. Update it if the committed repository differs.

```text
.
├── app/
│   ├── app.py
│   └── screenshot.png
├── results/
│   ├── analysis.md
│   ├── samples.md
│   ├── tables.md
│   ├── human_eval.csv
│   ├── sft/
│   ├── reward/
│   ├── ppo/
│   ├── dpo/
│   └── eval/
├── starter/
├── common.py
├── train_sft.py
├── train_reward.py
├── train_ppo.py
├── train_dpo.py
├── evaluate.py
├── modal_runner.py
├── requirements.txt
├── REPORT.md
└── README.md
```

## Reproducing the Project

### 1. Clone the repository

```bash
git clone https://github.com/Razaasherazi09/smollm2-rlhf-summarization.git
cd smollm2-rlhf-summarization
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the Python, CUDA, and PyTorch versions specified by the repository's environment configuration.

### 3. Configure cloud access

Authenticate with Modal before launching cloud workloads:

```bash
modal setup
```

Configure Hugging Face access if required. Store credentials using environment variables or platform secrets; never commit API tokens or other credentials.

### 4. Run training and evaluation

Inspect `modal_runner.py` for the exact supported stage commands. The project notes indicate a Modal orchestration entry point:

```bash
modal run modal_runner.py
```

Confirm whether this command runs all stages or requires stage-specific arguments. Run evaluation and tests using the entry points actually implemented in the repository. For example, if supported:

```bash
python evaluate.py
pytest -q
```

Save the generated metrics, summaries, and logs under `results/` so that reported values can be traced to the run that produced them.

### 5. Download published checkpoints

The project checkpoint repository is linked below. If it is public and contains the expected model artifacts, it can be downloaded with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="razaasherazi/smollm2-135m-rlhf-checkpoints",
    local_dir="checkpoints",
)
```

## Checkpoints and Deliverables

- Trained checkpoints: [Hugging Face — smollm2-135m-rlhf-checkpoints](https://huggingface.co/razaasherazi/smollm2-135m-rlhf-checkpoints?utm_source=chatgpt.com)
- Base model: [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M?utm_source=chatgpt.com)
- Source code: [GitHub repository](https://github.com/Razaasherazi09/smollm2-rlhf-summarization?utm_source=chatgpt.com)
- Academic report: `REPORT.md`
- Reward-model analysis: `results/analysis.md`
- Qualitative comparisons: `results/samples.md`
- Human evaluation records: `results/human_eval.csv`

## Limitations

- The reward model is a proxy trained from preference data; it can learn superficial patterns and may not generalize to human judgments.
- PPO's reported output-length saturation at the 48-token limit is consistent with a possible length-related reward incentive. Confirming reward exploitation requires examining rollout samples and reward behavior.
- DPO's higher reported reward-model score does not, by itself, prove superior factuality or human-perceived quality.
- ROUGE measures lexical overlap and does not fully capture semantic adequacy, factual consistency, or usefulness.
- The human evaluation sample is small, so its results are descriptive rather than statistically conclusive.
- Results depend on preprocessing, dataset splits, random seeds, decoding configuration, and implementation details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/?utm_source=chatgpt.com) and the SmolLM2 model team.
- [CarperAI](https://huggingface.co/CarperAI?utm_source=chatgpt.com) for the summarization and preference datasets.
- [Modal](https://modal.com/?utm_source=chatgpt.com) for serverless GPU infrastructure.
- Course instructors and human evaluators.

Author: Syed Muhammad Raza  
Course: Generative AI — Semester 7  
Project: Learning to Summarize from Human Feedback: End-to-End Alignment of SmolLM2-135M
