import os
import time
import json
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from starter.models import load_tokenizer
from starter.batching import collate_sft
from common import token_logprobs, get_summary_mask, generate_summaries


def compute_sft_loss(model, input_ids, attention_mask, prompt_lens):
    """
    Computes mean negative log-likelihood over summary tokens only.
    Prompt tokens and padding contribute zero.
    """
    # token_logprobs returns log pi(x_t | x_{<t}) for t >= 1 with shape (B, L - 1)
    logprobs = token_logprobs(model, input_ids, attention_mask)

    # summary_mask has shape (B, L); position t >= 1 aligns with logprobs[:, t - 1]
    summary_mask = get_summary_mask(input_ids, prompt_lens, attention_mask)
    target_mask = summary_mask[:, 1:].float()

    sum_nll = -(logprobs * target_mask).sum()
    total_tokens = target_mask.sum().clamp(min=1.0)
    return sum_nll / total_tokens


@torch.no_grad()
def evaluate_val_loss(model, val_loader, device):
    """Computes mean NLL loss over the 500 validation examples."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        loss = compute_sft_loss(model, input_ids, attention_mask, prompt_lens)
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def sample_eval_posts(model, tokenizer, eval_posts, device):
    """Generates greedy summaries for 5 evaluation posts (max 48 new tokens)."""
    model.eval()
    samples = []
    sample_posts = eval_posts[:5]

    for item in sample_posts:
        prompt_text = item["prompt"]
        human_ref = item["summary"]

        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        _, gen_tokens = generate_summaries(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=48,
            do_sample=False,
        )
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        samples.append({
            "prompt_tail": prompt_text[-250:],
            "human_reference": human_ref.strip(),
            "generated_summary": gen_text,
        })
    return samples


def train_sft(
    data_path="/vol/data/processed_splits.pt",
    checkpoint_dir="/vol/checkpoints/sft",
    results_dir="/vol/results",
    batch_size=16,
    lr=5e-5,
    grad_clip=1.0,
    eval_every=250,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Starting Stage 1: SFT on device: {device} ({gpu_name})")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    tok = load_tokenizer()
    splits = torch.load(data_path, weights_only=False)
    sft_train = splits["sft_train"]
    sft_val = splits["sft_val"]
    eval_posts = splits["eval_posts"]

    pad = tok.pad_token_id
    train_loader = DataLoader(
        sft_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_sft, pad_id=pad),
    )
    val_loader = DataLoader(
        sft_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_sft, pad_id=pad),
    )

    # 1. Load base model with SDPA attention for lower activation memory
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        attn_implementation="sdpa",
    ).to(device)

    # 2. Generate initial summaries before SFT (Task 2.5)
    print("Generating summaries for 5 evaluation posts BEFORE SFT...")
    samples_before = sample_eval_posts(model, tok, eval_posts, device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Setup AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader)
    print(f"Training 1 epoch: {len(sft_train)} pairs, batch size {batch_size} -> {total_steps} steps")

    train_steps_log = []
    train_loss_log = []
    val_steps_log = []
    val_loss_log = []

    running_loss = 0.0
    start_time = time.time()

    model.train()
    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        optimizer.zero_grad()
        loss = compute_sft_loss(model, input_ids, attention_mask, prompt_lens)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if step % eval_every == 0 or step == total_steps:
            steps_in_window = eval_every if (step % eval_every == 0 and step != total_steps) else ((step - 1) % eval_every + 1)
            avg_train_loss = running_loss / steps_in_window
            val_loss = evaluate_val_loss(model, val_loader, device)

            train_steps_log.append(step)
            train_loss_log.append(avg_train_loss)
            val_steps_log.append(step)
            val_loss_log.append(val_loss)

            elapsed = time.time() - start_time
            print(f"Step [{step:4d}/{total_steps}] | Train NLL: {avg_train_loss:.4f} | Val NLL: {val_loss:.4f} | Time: {elapsed:.1f}s")
            running_loss = 0.0
            model.train()

    total_duration = time.time() - start_time
    print(f"SFT Training completed in {total_duration:.1f}s")

    # 4. Save final checkpoint and tokenizer
    print(f"Saving SFT model and tokenizer to {checkpoint_dir}...")
    model.save_pretrained(checkpoint_dir)
    tok.save_pretrained(checkpoint_dir)

    # 5. Generate summaries after SFT (Task 2.5)
    print("Generating summaries for 5 evaluation posts AFTER SFT...")
    samples_after = sample_eval_posts(model, tok, eval_posts, device)

    # 6. Save qualitative samples to results
    qualitative_results = []
    for i in range(5):
        qualitative_results.append({
            "index": i + 1,
            "prompt_tail": samples_before[i]["prompt_tail"],
            "human_reference": samples_before[i]["human_reference"],
            "summary_before_sft": samples_before[i]["generated_summary"],
            "summary_after_sft": samples_after[i]["generated_summary"],
        })

    samples_json_path = os.path.join(results_dir, "sft_samples.json")
    with open(samples_json_path, "w") as f:
        json.dump(qualitative_results, f, indent=2)

    samples_md_path = os.path.join(results_dir, "sft_samples.md")
    with open(samples_md_path, "w") as f:
        f.write("# Task 2.5: Qualitative Summaries Before and After SFT\n\n")
        for item in qualitative_results:
            f.write(f"### Sample {item['index']}\n")
            f.write(f"**Prompt (Tail):** ...{item['prompt_tail']}\n\n")
            f.write(f"**Human Reference:** {item['human_reference']}\n\n")
            f.write(f"**Before SFT (Base Model):** {item['summary_before_sft']}\n\n")
            f.write(f"**After SFT (Stage 1):** {item['summary_after_sft']}\n\n")
            f.write("---\n")

    # 7. Plot Figure 1 (SFT Loss Curves)
    fig_path = os.path.join(results_dir, "fig1_sft_loss.png")
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps_log, train_loss_log, marker="o", label="Training Loss (NLL)")
    plt.plot(val_steps_log, val_loss_log, marker="s", label="Validation Loss (NLL)")
    plt.title("Figure 1: SFT Loss over Training Steps")
    plt.xlabel("Steps")
    plt.ylabel("Negative Log-Likelihood")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved Figure 1 to {fig_path}")

    # 8. Record Table 2 Training Metrics
    metrics = {
        "stage": "SFT",
        "final_train_loss": float(train_loss_log[-1]),
        "final_val_loss": float(val_loss_log[-1]),
        "total_time_seconds": round(total_duration, 2),
        "gpu": gpu_name,
        "steps_log": train_steps_log,
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
    }
    metrics_path = os.path.join(results_dir, "sft_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved SFT metrics to {metrics_path}")

    return metrics