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
)

@app.function(
    image=image,
    gpu="L4",
    volumes={"/vol": volume},
    timeout=2400,
)
def run_stage1_sft():
    from train_sft import train_sft

    print("Launching Stage 1 SFT on Modal L4 GPU...")
    metrics = train_sft()

    # Commit saved checkpoint, plots, and json results to persistent volume
    volume.commit()
    print("SFT artifacts, checkpoints, and figures committed to /vol.")
    return metrics

@app.local_entrypoint()
def main():
    metrics = run_stage1_sft.remote()
    print("\n--- SFT Training Finished Successfully ---")
    print(f"Final Train Loss: {metrics['final_train_loss']:.4f}")
    print(f"Final Val Loss:   {metrics['final_val_loss']:.4f}")
    print(f"Duration:         {metrics['total_time_seconds']}s")
    print(f"GPU:              {metrics['gpu']}")