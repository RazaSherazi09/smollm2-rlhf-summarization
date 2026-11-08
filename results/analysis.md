# Task 7: RLHF & Preference Alignment Analysis

---

## 7.1 Reward Model Accuracy & Error Analysis

### Pairwise Accuracy
On the held-out validation set of 1,000 preference pairs, the Bradley–Terry Reward Model achieved a pairwise validation accuracy of **63.60%** (final training loss: 0.5812). Given that Reddit TL;DR human preferences carry an estimated 35%–40% inter-annotator disagreement margin, 63.60% aligns closely with the empirical Bayes optimal ceiling for a 135M parameter backbone.

### Failure Modes & Qualitative Analysis
Analysis of misclassified pairs ($r_\phi(x, y_w) \le r_\phi(x, y_l)$) from `results/reward/reward_misclassified_sample.json` revealed three primary error mechanisms:

1. **Verbosity and Grammatical Completeness Bias:** The linear scalar head assigns positive weight to longer sequences ending with clean terminal punctuation, occasionally penalizing concise human-preferred summaries.
2. **Context Degradation over Long Prompts:** Because the reward score is pooled exclusively from the final token embedding $h_T$, cross-attention signals from early context tokens decay over 300+ prompt tokens in a 135M model, allowing subtle factual hallucinations in rejected candidates to go unpenalized.
3. **Subjective Preference Margins:** Edge cases where both summaries are acceptable or neither is ideal.

### Three Concrete Misclassified Pairs

#### Misclassified Pair 1 (`score_diff: -0.9844`)
* **Prompt Length:** 400 tokens
* **Human Chosen Reward ($r_\phi(x, y_w)$):** +1.7344
* **Human Rejected Reward ($r_\phi(x, y_l)$):** +2.7188 ($\Delta r = -0.9844$)
* **Chosen Summary ($y_w$):** *"College credit hours cannot be canceled for past debt; need alternatives."* (11 tokens)
* **Rejected Summary ($y_l$):** *"Four years ago I took classes at PSU and accumulated debt, and now I am unable to receive financial aid because of the uncooperative records of my parents, which prevents me from graduating."* (35 tokens)
* **Why the Model Erred:** The rejected candidate is three times longer and sounds like a well-formed narrative with complete clauses. The reward head prioritized syntactic completeness and sentence length over the concise summary favored by the human annotator.

#### Misclassified Pair 2 (`score_diff: -0.3750`)
* **Prompt Length:** 321 tokens
* **Human Chosen Reward ($r_\phi(x, y_w)$):** +3.1719
* **Human Rejected Reward ($r_\phi(x, y_l)$):** +3.5469 ($\Delta r = -0.3750$)
* **Chosen Summary ($y_w$):** *"Crush broke our 6-month no-contact rule to text me happy birthday. Wondering if I should reply."* (16 tokens)
* **Rejected Summary ($y_l$):** *"Long-time crush messaged me happy birthday after we broke contact for forever, making me feel sad and confused about whether I should respond to her message or ignore it."* (29 tokens)
* **Why the Model Erred:** Both completions are factually accurate, but the rejected summary expands on the emotional dilemma with descriptive adjectives. The reward model rewarded this descriptive detail, giving it a slight edge (+0.375) over the tighter chosen text.

#### Misclassified Pair 3 (`score_diff: -1.0156`)
* **Prompt Length:** 284 tokens
* **Human Chosen Reward ($r_\phi(x, y_w)$):** +2.4062
* **Human Rejected Reward ($r_\phi(x, y_l)$):** +3.4218 ($\Delta r = -1.0156$)
* **Chosen Summary ($y_w$):** *"GF of 2 years and I are dealing with anxiety and lack of intimacy. Looking for advice to spice things up."* (22 tokens)
* **Rejected Summary ($y_l$):** *"Girlfriend is on birth control pills which causes extreme pain and anxiety, ruining our relationship completely. Need urgent advice."* (20 tokens)
* **Why the Model Erred:** The rejected summary introduces a strong claim (*"ruining our relationship completely"*) that was exaggerated relative to the prompt context. The reward model's scalar head reacted to the high-urgency keywords (*"extreme pain"*, *"urgent advice"*) rather than catching the factual distortion.

---

## 7.2 PPO Dynamics: Reward, KL, Summary Length, and Human Evaluation

### Trajectory of Metrics (Refer to `results/ppo/fig3_ppo_curves.png`)
* **Mean $r_\phi$ Score:** Rose consistently across training. Starting from negative baseline values, rollout batch mean rewards climbed to **+8.520** at Iteration 200. On the 300 held-out evaluation test posts, PPO achieved a mean reward of **+0.513** (vs. SFT at +0.352).
* **Mean KL Divergence:** Grew steadily from **0.0000 nats** at Iteration 1 to **1.82 nats** at Iteration 200, remaining bounded without exploding thanks to the token-level KL penalty ($\beta_{\text{KL}} = 0.05$).
* **Summary Length:** Expanded rapidly during rollouts, shifting from an initial mean of ~33.9 tokens up to the **48.0 token ceiling** (the generation limit), where it saturated for the remainder of training.

### Is a Higher Reward-Model Score the Same as a Better Summary?
**No.** While PPO achieved higher scalar rewards than SFT (+0.513 vs. +0.352), qualitative inspection and human evaluation demonstrate classic **reward exploitation (reward hacking)**:
1. **Length Saturation:** PPO learned that maximizing token output up to the 48-token truncation threshold generated higher rewards, often padding summaries with trailing descriptive clauses or repeated phrases.
2. **Lexical Degradation:** PPO's ROUGE-L dropped sharply to **17.54** (compared to SFT's **25.42**), reflecting a divergence from natural human summarization style.
3. **Human Evaluation Findings:** In our 30-sample blinded human evaluation (Table 4), human evaluators picked PPO only **13.3% to 16.7%** of the time, compared to **66.7% to 80.0%** for DPO. Evaluators noted that PPO summaries frequently ended abruptly due to hitting the 48-token limit or exhibited repetitive phrasing. Thus, an elevated proxy reward score does not reliably indicate higher human-perceived quality.

---

## 7.3 Role of $\beta_{\text{KL}}$ & Analysis of $\beta_{\text{KL}} = 0$ (Eq. 41)

In reinforcement learning from human feedback, the objective (Eq. 41) balances reward maximization against reference policy divergence:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) \right] - \beta_{\text{KL}} \mathbb{D}_{\text{KL}}\left(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)\right)$$

### Role of $\beta_{\text{KL}}$
$\beta_{\text{KL}}$ serves as an entropy and trust-region anchor:
* It keeps the policy $\pi_\theta$ within the valid linguistic distribution learned during SFT ($\pi_{\text{ref}}$).
* It prevents the policy from collapsing its generative entropy into degenerate, repetitive loops that game the scalar head $r_\phi$.

### Expected Outcome with $\beta_{\text{KL}} = 0$
If $\beta_{\text{KL}} = 0$:
1. **Unconstrained Reward Exploitation:** The policy has no incentive to stay close to natural language. It will exploit blind spots and spurious correlations in the reward model's scalar linear head $W_v$.
2. **Language Degeneration & Mode Collapse:** The policy would likely collapse into repetitive phrases, punctuation loops, or nonsensical keyword stuffing that trigger high positive activations in the final layer embedding $h_T$.
3. **Catastrophic Forgetting:** The model would lose its general instruction-following and grammatical coherence, resulting in near-zero ROUGE scores despite recording high proxy reward values.

---

## 7.4 PPO Clip Fraction and Policy Update Dynamics

The PPO clipped surrogate objective uses the importance sampling ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}, \qquad L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

with $\epsilon = 0.2$.

### Observed Clip Fraction Behavior
* In the first epoch ($k=1$) of every rollout iteration, $r_t(\theta) \equiv 1.0$ exactly, producing a clip fraction of **0.00%**.
* Across the subsequent $K=4$ optimization epochs, the clip fraction remained bounded within **2.5% to 13.6%**, averaging approximately **6% to 8%** throughout training.

### What It Indicates About Update Sizes
A clip fraction consistently between 5% and 15% indicates **healthy, stable policy updates within the trust region**:
* If the clip fraction were consistently 0%, the learning rate or advantage estimates would be too small to produce policy movement.
* If the clip fraction spiked above 30%–40%, the policy would be taking excessively large gradient steps, destroying previously learned representations.
* The steady 6%–8% clip rate confirms that updates were bounded without destabilizing the actor network.

---

## 7.5 DPO Mathematical Formulation (Eqs. 55–58) & Implicit Reward Accuracy

Direct Preference Optimization bypasses explicit reward modeling and reinforcement learning by directly solving the KL-constrained preference problem.

### Where Did the Reward Go? (Derivation of Eqs. 55–58)
Starting from the KL-regularized objective (Eq. 55):

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r(x, y)] - \beta \mathbb{D}_{\text{KL}}(\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

The closed-form optimal policy solution (Eq. 56) is:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function.

Taking the logarithm and rearranging yields the ground-truth reward in terms of the optimal policy (Eq. 57):

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting this expression for $r(x, y)$ into the Bradley–Terry preference model $p(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$ (Eq. 58):

$$r(x, y_w) - r(x, y_l) = \left[ \beta \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right]$$

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

The reward function is reparameterized as the log-ratio of the policy relative to the reference: **$r(x, y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$**.

### Why $Z(x)$ Does Not Need to Be Computed
Because $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ depends strictly on prompt $x$ and sums over all completions $y$, it is constant with respect to $y_w$ and $y_l$. When computing the preference difference $r(x, y_w) - r(x, y_l)$, the $\beta \log Z(x)$ terms cancel out algebraically:

$$\beta \log Z(x) - \beta \log Z(x) = 0$$

This eliminates the intractable partition function from the loss.

### Validation Accuracy: Reward Model vs. DPO Implicit Reward
Evaluated on the identical 1,000 holdout validation preference pairs:
* **Reward Model Pairwise Accuracy:** **63.60%**
* **DPO Implicit Reward Accuracy:** **60.50%**

DPO's implicit reward accuracy is within 3.1% of the standalone reward model, confirming that policy log-probabilities under DPO learn pairwise human preferences without training a separate scalar head.

---

## 7.6 System Resources, Wall-Clock Time & Method Selection

### Models in GPU Memory
* **PPO requires 4 models concurrently in memory:**
  1. Trainable Policy ($\pi_\theta$)
  2. Trainable Value Function / Critic ($V_\psi$)
  3. Frozen Reference Policy ($\pi_{\text{ref}}$)
  4. Frozen Reward Model ($r_\phi$)
* **DPO requires 2 models concurrently in memory:**
  1. Trainable Policy ($\pi_\theta$)
  2. Frozen Reference Policy ($\pi_{\text{ref}}$)

### Wall-Clock Training Times (NVIDIA A10G on Modal Labs)
* **Stage 1 (SFT):** 1,124.6s (~18.7 min)
* **Stage 2 (Reward Model):** 1,032.6s (~17.2 min)
* **Stage 3 (PPO):** 662.5s (~11.0 min) — *200 iterations (3,200 prompts, 4 rollout epochs)*
* **Stage 4 (DPO):** 1,811.7s (~30.2 min) — *1 full epoch (2,500 steps over 20,000 preference pairs)*

*Total Pipeline Execution:*
* Full RLHF Pipeline (SFT + RM + PPO): **2,819.7s (~47.0 min)**
* Full DPO Pipeline (SFT + DPO): **2,936.3s (~48.9 min)**

### Method Selection for Future Projects
For a new production or research project, **DPO is the superior choice**:
1. **Architectural Simplicity:** Halves GPU memory requirements (2 models instead of 4) and eliminates the need for actor-critic synchronization, generalized advantage estimation (GAE), and rollout buffer management.
2. **Optimization Stability:** DPO optimizes a stable, supervised classification-style loss function. It avoids PPO's high rollout variance, value function drift, and reward hacking (such as the length saturation observed in Stage 3).
3. **Superior Empirical Alignment:** DPO achieved a **+2.655 mean reward** and a **96.33% win rate over SFT**, whereas PPO achieved only +0.513 and 55.33%. Human annotators selected DPO over 4× more often than PPO in blinded evaluation.

---

# Section 4: Benchmark Tables

## 4.1 Table 1 – Data
| Metric | Value |
| :--- | :---: |
| SFT train / validation pairs | 20,000 / 2,000 |
| Preference train / validation pairs | 20,000 / 1,000 |
| Evaluation posts | 300 |
| Mean / max prompt length, SFT train (tokens) | 196.4 / 448 |
| Mean / max summary length, SFT train (tokens) | 26.8 / 48 |

---

## 4.2 Table 2 – Training
| Stage | Final Train Loss | Validation Metric | Time / GPU |
| :--- | :---: | :---: | :---: |
| **SFT** | 1.8540 | val. loss: 1.9124 | 1,124.6s / NVIDIA A10G |
| **Reward Model** | 0.5812 | pairwise acc.: 63.60% | 1,032.6s / NVIDIA A10G |
| **PPO** | 0.0814 | final mean $r_\phi$: +8.520 | 662.5s / NVIDIA A10G |
| **DPO** | 0.6312 | implicit reward acc.: 60.50% | 1,811.7s / NVIDIA A10G |

---

## 4.3 Table 3 – Final Comparison (300 Evaluation Posts)
| Policy | Mean $r_\phi$ | Win Rate vs. SFT | KL to SFT | Length | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SFT** | +0.352 | — | 0.00 | 30.4 | 25.42 |
| **PPO** | +0.513 | 55.33% | 1.82 | 48.0 | 17.54 |
| **DPO** | +2.655 | 96.33% | 0.78 | 45.9 | 22.88 |

---

## 4.4 Table 4 – Human Evaluation (30 Posts)
| Evaluator | SFT Picked | PPO Picked | DPO Picked |
| :--- | :---: | :---: | :---: |
| **Member 1** | 1 (3.3%) | 5 (16.7%) | 24 (80.0%) |
| **Member 2** | 6 (20.0%) | 4 (13.3%) | 20 (66.7%) |
| **Agreement Between Members** | **23 / 30** | **(76.7%)** | — |
