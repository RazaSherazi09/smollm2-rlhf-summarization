import os
import modal

app = modal.App("smollm2-rlhf-runner")
volume = modal.Volume.from_name("rlhf-storage-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "rouge-score",
        "matplotlib",
        "numpy",
        "tqdm",
    )
    .env({
        "HF_HOME": "/vol/hf_cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_python_source("starter")
    .add_local_file("common.py", "/root/common.py")
    .add_local_file("train_sft.py", "/root/train_sft.py")
    .add_local_file("train_reward.py", "/root/train_reward.py")
    .add_local_file("train_ppo.py", "/root/train_ppo.py")
    .add_local_file("train_dpo.py", "/root/train_dpo.py")
    .add_local_file("evaluate.py", "/root/evaluate.py")
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume},
    timeout=2400,
)
def run_evaluation():
    from evaluate import run_full_evaluation

    print("Launching Task 6 Evaluation on Modal A10G GPU...")
    table1, win_rates = run_full_evaluation()

    # Commit evaluation tables, metrics, and JSONL to persistent volume
    volume.commit()
    print("Evaluation deliverables committed to /vol.")
    return table1, win_rates

@app.local_entrypoint()
def main():
    table1, win_rates = run_evaluation.remote()
    print("\n=================== TASK 6: TABLE 1 RESULTS ===================")
    for row in table1:
        print(f"{row['Model']:<16} | R1: {row['ROUGE-1']:<5} | R2: {row['ROUGE-2']:<5} | RL: {row['ROUGE-L']:<5} | Len: {row['Mean Length']:<4} | RM: {row['Mean Reward']:+.3f}")
    
    print("\n================= TASK 6: WIN RATE MATRIX (%) =================")
    for m, opponents in win_rates.items():
        print(f"{m:<6} vs SFT: {opponents.get('SFT', '-'):>5}% | vs Base: {opponents.get('Base', '-'):>5}% | vs DPO: {opponents.get('DPO', '-'):>5}%")