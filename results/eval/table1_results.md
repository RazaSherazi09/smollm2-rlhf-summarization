# Table 1: Automatic Evaluation on 300 Held-Out Test Posts

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean Length (Tokens) | Mean Reward $r_\phi$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base | 18.68 | 4.77 | 15.47 | 48.0 | -1.396 |
| SFT | 32.04 | 11.93 | 25.42 | 30.4 | +0.352 |
| PPO | 23.59 | 5.29 | 17.54 | 48.0 | +0.513 |
| DPO | 31.19 | 10.35 | 22.88 | 45.9 | +2.655 |
| Human Reference | 100.00 | 100.00 | 100.00 | 32.8 | +1.008 |


# Table 2: Pairwise Win Rates (% Row beats Column under $r_\phi$)

| vs. | Base | SFT | PPO | DPO |
| :--- | :---: | :---: | :---: | :---: |
| **Base** | 50.0% | 12.0% | 11.3% | 0.7% |
| **SFT** | 87.0% | 50.0% | 44.7% | 3.3% |
| **PPO** | 88.7% | 55.3% | 50.0% | 4.7% |
| **DPO** | 99.3% | 96.3% | 95.0% | 50.0% |
