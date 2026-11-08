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
| Stage | Final train loss | Validation metric | Time / GPU |
| :--- | :---: | :---: | :---: |
| **SFT** | 1.8540 | val. loss: 1.9124 | 1,124.6s / NVIDIA A10G |
| **Reward model** | 0.5812 | pairwise acc.: 63.60% | 1,032.6s / NVIDIA A10G |
| **PPO** | 0.0814 | final mean $r_\phi$: +8.520 | 662.5s / NVIDIA A10G |
| **DPO** | 0.6312 | implicit reward acc.: 60.50% | 1,811.7s / NVIDIA A10G |

---

## 4.3 Table 3 – Final Comparison (300 evaluation posts)
| Policy | Mean $r_\phi$ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SFT** | +0.352 | — | 0.00 | 30.4 | 25.42 |
| **PPO** | +0.513 | 55.33% | 1.82 | 48.0 | 17.54 |
| **DPO** | +2.655 | 96.33% | 0.78 | 45.9 | 22.88 |

---

## 4.4 Table 4 – Human Evaluation (30 posts)
| Evaluator | SFT picked | PPO picked | DPO picked |
| :--- | :---: | :---: | :---: |
| **Member 1** | 1 | 5 | 24 |
| **Member 2** | 6 | 4 | 20 |
| **Agreement between members** | \multicolumn{3}{c|}{**23 / 30 (76.7%)**} |
