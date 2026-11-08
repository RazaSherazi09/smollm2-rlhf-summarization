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
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume},
    timeout=2400,
)
def run_stage4_dpo():
    from train_dpo import train_dpo

    print("Launching Stage 4: DPO on Modal A10G GPU...")
    metrics = train_dpo()

    # Commit checkpoints, Figure 4, and metrics to persistent volume
    volume.commit()
    print("DPO checkpoint, plots, and metrics committed to /vol.")
    return metrics

@app.local_entrypoint()
def main():
    metrics = run_stage4_dpo.remote()
    print("\n--- DPO Training Finished Successfully ---")
    print(f"Final Train Loss:        {metrics['final_train_loss']:.4f}")
    print(f"Final Val Loss:          {metrics['final_val_loss']:.4f}")
    print(f"Implicit Reward Acc:     {metrics['final_implicit_accuracy'] * 100:.2f}%")
    print(f"Duration:                {metrics['total_time_seconds']}s")
    print(f"GPU:                     {metrics['gpu']}")