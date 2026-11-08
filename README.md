# SmolLM2-135M RLHF & DPO Summarization

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Compute-Modal%20%7C%20NVIDIA%20A10G-7654b8)](https://modal.com/)

An end-to-end preference-alignment project that adapts `HuggingFaceTB/SmolLM2-135M` for Reddit TL;DR summarization. It explores Supervised Fine-Tuning (SFT), Bradley–Terry Reward Modeling, Proximal Policy Optimization (PPO), and Direct Preference Optimization (DPO).

> **Reproducibility note:** Metrics and check outcomes in this README are project-reported values. Verify them against the committed code, raw evaluation files, and captured test logs before treating them as independently reproduced results. Reward-model scores are proxy measurements, not direct human judgments.

## Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Data and infrastructure](#data-and-infrastructure)
- [Methodology](#methodology)
- [Results](#results)
- [Human evaluation](#human-evaluation)
- [Correctness checks](#correctness-checks)
- [Demo](#demo)
- [Repository structure](#repository-structure)
- [Reproduction](#reproduction)
- [Limitations](#limitations)

## Overview

| Component | Details |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M` |
| Model size | Approximately 134.5M parameters |
| Task | Reddit post → concise TL;DR |
| Datasets | `CarperAI/openai_summarize_tldr`, `CarperAI/openai_summarize_comparisons` |
| Training stages | SFT → Reward Model → PPO; SFT → DPO |
| Reported compute | Modal serverless NVIDIA A10G, 24 GB VRAM |
| Author | Syed Muhammad Raza |
| Course | Generative AI, Semester 7 |

### Goals

1. Fine-tune a small decoder-only model for instruction-conditioned summarization.
2. Learn a reward function from chosen/rejected summary pairs.
3. Investigate online policy optimization with PPO and reference-policy KL regularization.
4. Optimize directly from preference pairs using DPO.
5. Compare lexical overlap, reward-model scores, and blinded human judgments.

## Pipeline

```text
Prompt-summary data
       |
       v
 Base SmolLM2-135M
       |
       v
   SFT policy ------------------------------+
       |                                    |
       v                                    v
 Reward model                         DPO training
       |                                    ^
       v                                    |
 PPO rollouts + KL penalty             Preference pairs
       |
       v
   PPO policy

SFT / PPO / DPO policies
       |
       v
Automatic metrics + human evaluation
```

### 1. Supervised Fine-Tuning (SFT)

The base model is trained on prompt-summary examples using causal language-model cross-entropy over summary tokens only. Prompt and padding positions are masked from the loss.

### 2. Bradley–Terry Reward Modeling

The reward model is initialized from the SFT transformer and extended with a scalar prediction head. For prompt \(x\), chosen response \(y_w\), and rejected response \(y_l\), it minimizes:

\[
\mathcal{L}_{RM} =
-\mathbb{E}\left[
\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
\right].
\]

### 3. Proximal Policy Optimization (PPO)

PPO uses a trainable policy and value function alongside frozen reference-policy and reward-model copies. A token-level KL penalty discourages excessive policy drift, while the reward model provides the sequence-level reward.

### 4. Direct Preference Optimization (DPO)

DPO trains directly on chosen/rejected responses relative to the SFT reference policy, without requiring a separate reward model during DPO optimization. The reported DPO temperature parameter was \(\beta=0.1\).

## Data and infrastructure

- `CarperAI/openai_summarize_tldr`: Reddit posts and TL;DR summaries.
- `CarperAI/openai_summarize_comparisons`: pairwise preference comparisons.

The experiment notes report 20,000 prompt-summary examples and 20,000 preference comparisons. For full reproducibility, record the dataset revision, filtering/preprocessing steps, exact split sizes, and random seeds.

Training was reported on Modal Labs using an NVIDIA A10G GPU (24 GB VRAM). The reported precision setup used BF16 autocast with FP32 master parameters.

## Results

### Automatic evaluation

Reported evaluation on 300 held-out posts using greedy decoding, `max_new_tokens=48`, temperature `0.0`, and stopping at `<|endoftext|>`.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean tokens | Mean reward \(r_\phi\) |
|---|---:|---:|---:|---:|---:|
| Base SmolLM2-135M | 18.68 | 4.77 | 15.47 | 48.0 | -1.396 |
| SFT | **32.04** | **11.93** | **25.42** | 30.4 | +0.352 |
| PPO | 23.59 | 5.29 | 17.54 | 48.0 | +0.513 |
| DPO | 31.19 | 10.35 | 22.88 | 45.9 | **+2.655** |
| Human reference | 100.00* | 100.00* | 100.00* | 32.8 | +1.008 |

\*The reference row's ROUGE scores are a self-comparison, not a model performance result.

### Pairwise reward-model win rates

Percentage of evaluation examples where the row candidate received a higher reward-model score than the column candidate.

| Candidate | Base | SFT | PPO | DPO |
|---|---:|---:|---:|---:|
| Base | 50.00% | 12.00% | 11.33% | 0.67% |
| SFT | 87.00% | 50.00% | 44.67% | 3.33% |
| PPO | 88.67% | 55.33% | 50.00% | 4.67% |
| DPO | 99.33% | 96.33% | 95.33% | 50.00% |

These are reward-model-based comparisons, not human pairwise win rates. A reward model can contain biases and may favor outputs that exploit its learned scoring patterns.

### Reported training configuration

| Setting | SFT | Reward model | PPO | DPO |
|---|---:|---:|---:|---:|
| Initialization | Base model | SFT | SFT policy + value head | SFT |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Learning rate | \(5\times10^{-5}\) | \(5\times10^{-5}\) | \(1\times10^{-5}\) | \(2\times10^{-6}\) |
| Batch / rollout size | 8 | 8 | 16 | 8 |
| Training | 1 epoch | 1 epoch | 200 iterations | 1 epoch |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 |
| Wall time | 1,124.6 s | 1,032.6 s | 662.5 s | 1,811.7 s |

PPO coefficients reported: \(\epsilon=0.2\), \(c_V=0.1\), \(c_H=0.01\), and \(\beta_{KL}=0.05\). DPO used \(\beta=0.1\).

## Human evaluation

The project notes report a blinded comparison over 30 held-out posts, with SFT, PPO, and DPO outputs shown under randomized anonymous labels.

| Evaluator | DPO | PPO | SFT |
|---|---:|---:|---:|
| Member 1 | 24 (80.0%) | 5 (16.7%) | 1 (3.3%) |
| Member 2 | 20 (66.7%) | 4 (13.3%) | 6 (20.0%) |

Reported inter-annotator agreement: **23/30 (76.7%)**.

Include the raw annotations at `results/human_eval.csv`. Also document evaluator instructions, randomization, tie/abstention handling, and whether both evaluators rated the same posts. This small sample should not be presented as statistically conclusive.

## Correctness checks

The rubric requests evidence for six implementation invariants. The table below records the outcomes supplied in the project notes. **These are reported outcomes, not independently verified test results.** Commit runnable tests and genuine terminal output from the final code before claiming the checks are verified.

| Check | Invariant | Reported outcome |
|---|---|---|
| 1. SFT summary mask | Decoding summary-mask positions recovers the target summary plus EOS, without prompt leakage. | Passed; exact token match reported |
| 2. Frozen models | PPO reference policy, PPO reward model, and DPO reference policy parameters have `requires_grad=False`. | Passed; reported tensor counts: 272, 273, 272 |
| 3. PPO initial KL | Identical initial policy/reference weights produce approximately zero sampled-action log-probability difference. | Passed; mean KL reported as 0.000000 nats |
| 4. PPO first epoch | Before the first optimizer update, policy ratios are approximately 1 and clip fraction is 0. | Passed; ratio reported in [0.999999, 1.000001], clip fraction 0% |
| 5. DPO initial loss | When policy equals reference, the preference margin is zero and loss is \(\ln 2 \approx 0.693147\). | Passed; empirical initial loss reported as 0.693147 |
| 6. Reward pooling | Pool from the last non-padding position using `attention_mask`, including left-padded batches. | Passed; attention-mask indexing reported |

### Suggested executable assertions

Adapt these examples to the actual model and variable names in the repository. Tests should inspect real model parameters and real batches; do not print hard-coded expected outputs.

```python
import math
import torch

assert all(not p.requires_grad for p in reference_policy.parameters())
assert all(not p.requires_grad for p in reward_model.parameters())

assert torch.allclose(
    initial_dpo_loss,
    torch.tensor(math.log(2.0), device=initial_dpo_loss.device),
    atol=1e-5,
)
```

For the pooling invariant, create a batch with different left-padding lengths and verify that each selected index is the final position for which `attention_mask == 1`. This avoids confusing EOS with padding when both share a token ID.

**Recommended:** add `tests/test_correctness.py`, run it against the submitted implementation, and paste the exact command and unedited output here.

## Demo

If `app/app.py` exists and the required checkpoints are available locally, launch the Streamlit interface:

```bash
streamlit run app/app.py
```

The interface is intended to compare SFT, PPO, and DPO generations and optionally display reward-model scores. Label these scores as model-generated proxy scores, not human ratings.

## Repository structure

Update this tree to match the files actually committed to your repository.

```text
.
├── app/
│   ├── app.py
│   └── screenshot.png
├── checkpoints/
│   ├── sft/
│   ├── reward/
│   ├── ppo/
│   └── dpo/
├── results/
│   ├── analysis.md
│   ├── samples.md
│   ├── human_eval.csv
│   ├── sft/
│   ├── reward/
│   ├── ppo/
│   ├── dpo/
│   └── eval/
├── tests/
│   └── test_correctness.py
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

## Reproduction

### 1. Clone

```bash
git clone https://github.com/Razaasherazi09/smollm2-rlhf-summarization.git
cd smollm2-rlhf-summarization
```

### 2. Install dependencies

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the Python, CUDA, and PyTorch versions specified by the repository's environment configuration.

### 3. Configure services

Authenticate with Modal for cloud training. Configure Hugging Face access if required by the project. Store credentials in environment variables or platform secrets; never commit tokens.

### 4. Run training and evaluation

Inspect `modal_runner.py` for the exact supported stage commands. A typical Modal entry point may be:

```bash
modal run modal_runner.py
```

Do not assume one command runs every stage unless the runner implements that behavior. Record the exact command and configuration used for each stage.

Run the evaluation and test entry points supported by the repository. For example:

```bash
python evaluate.py
pytest -q
```

These are examples; confirm the scripts and tests expose these commands before using them in a submission. Save metrics, generated summaries, and logs under `results/`.

## Checkpoints and project links

- **Base model:** [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- **Source repository:** [Razaasherazi09/smollm2-rlhf-summarization](https://github.com/Razaasherazi09/smollm2-rlhf-summarization)

Add a project checkpoint Hub URL only after confirming that the repository exists, is accessible, and contains the expected model files.

## Limitations and interpretation

- The reward model is a proxy for preference labels and can learn superficial artifacts.
- A high reward score does not prove that humans prefer an output.
- PPO's reported length saturation is consistent with possible reward exploitation; confirming it requires analysis of rollout samples and reward components.
- DPO's higher reward-model score does not by itself establish better factuality or human-perceived quality.
- ROUGE measures lexical overlap and does not fully measure semantic adequacy, factual consistency, or usefulness.
- The human evaluation set is small; report its size and procedure transparently.
- Results depend on preprocessing, data splits, random seeds, decoding settings, and implementation details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) and the SmolLM2 model team.
- [CarperAI](https://huggingface.co/CarperAI) for the summarization and preference datasets.
- [Modal](https://modal.com/) for serverless GPU infrastructure.
- Course instructors and human evaluators.

---

**Author:** Syed Muhammad Raza  
**Course:** Generative AI — Semester 7  
**Project:** Learning to Summarize from Human Feedback: End-to-End Alignment of SmolLM2-135M
