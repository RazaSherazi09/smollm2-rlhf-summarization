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
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": volume},
    timeout=2400,
)
def run_stage3_ppo():
    from train_ppo import train_ppo

    print("Launching Stage 3: PPO on Modal A10G GPU...")
    metrics = train_ppo()

    # Commit checkpoints, Figure 3, and metrics to persistent volume
    volume.commit()
    print("PPO checkpoint, plots, and metrics committed to /vol.")
    return metrics

@app.local_entrypoint()
def main():
    metrics = run_stage3_ppo.remote()
    print("\n--- PPO Training Finished Successfully ---")
    print(f"Final Mean Reward:    {metrics['final_mean_reward']:+.4f}")
    print(f"Final Mean KL:        {metrics['final_mean_kl']:.4f}")
    print(f"Final Mean Length:    {metrics['final_mean_length']:.1f} tokens")
    print(f"Final Clip Fraction:  {metrics['final_clip_fraction'] * 100:.2f}%")
    print(f"Duration:             {metrics['total_time_seconds']}s")
    print(f"GPU:                  {metrics['gpu']}")