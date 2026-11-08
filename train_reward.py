import os
import time
import json
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from starter.models import load_tokenizer
from starter.batching import collate_pref


class RewardModel(nn.Module):
    """
    Reward model using the transformer body from the SFT checkpoint with a scalar linear head.
    Master weights are maintained in float32 to prevent AdamW update underflow,
    while execution runs in bfloat16 via torch.autocast.
    """
    def __init__(self, sft_path):
        super().__init__()
        base_causal_lm = AutoModelForCausalLM.from_pretrained(
            sft_path,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        self.transformer = base_causal_lm.model
        hidden_size = base_causal_lm.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1, bias=False, dtype=torch.float32)

    def forward(self, input_ids, attention_mask):
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # shape: (B, L, H)

            # Identify last real token using attention_mask, never pad_token_id (Section 3.3)
            last_indices = (attention_mask.sum(dim=1) - 1).clamp(min=0)
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            last_hidden = hidden_states[batch_idx, last_indices]

            rewards = self.v_head(last_hidden).squeeze(-1)
            
        return rewards.float()


def compute_reward_loss(r_w, r_l):
    """Bradley-Terry loss: -log(sigmoid(r_w - r_l))"""
    diff = r_w - r_l
    return -F.logsigmoid(diff).mean()


@torch.no_grad()
def evaluate_reward_model(model, val_loader, device, collect_errors=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    misclassified = []

    for batch in val_loader:
        ch_ids = batch["chosen_ids"].to(device)
        ch_mask = batch["chosen_mask"].to(device)
        rj_ids = batch["rejected_ids"].to(device)
        rj_mask = batch["rejected_mask"].to(device)

        r_w = model(ch_ids, ch_mask)
        r_l = model(rj_ids, rj_mask)

        loss = compute_reward_loss(r_w, r_l)
        total_loss += loss.item() * len(r_w)

        correct = (r_w > r_l)
        total_correct += correct.sum().item()
        total_pairs += len(r_w)

        if collect_errors and len(misclassified) < 10:
            for i in range(len(correct)):
                if not correct[i]:
                    misclassified.append({
                        "prompt_tokens": batch["prompt_lens"][i].item(),
                        "r_chosen": round(r_w[i].item(), 4),
                        "r_rejected": round(r_l[i].item(), 4),
                        "score_diff": round((r_w[i] - r_l[i]).item(), 4),
                    })

    avg_loss = total_loss / max(total_pairs, 1)
    acc = total_correct / max(total_pairs, 1)
    return avg_loss, acc, misclassified


def train_reward(
    data_path="/vol/data/processed_splits.pt",
    sft_checkpoint="/vol/checkpoints/sft",
    checkpoint_dir="/vol/checkpoints/reward",
    results_dir="/vol/results",
    batch_size=8,
    lr=5e-5,
    grad_clip=1.0,
    eval_every=250,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Starting Stage 2: Reward Model on device: {device} ({gpu_name})")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    tok = load_tokenizer()
    splits = torch.load(data_path, weights_only=False)
    pref_train = splits["pref_train"]
    pref_val = splits["pref_val"]

    pad = tok.pad_token_id
    train_loader = DataLoader(
        pref_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_pref, pad_id=pad),
    )
    val_loader = DataLoader(
        pref_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_pref, pad_id=pad),
    )

    model = RewardModel(sft_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader)
    print(f"Training 1 epoch: {len(pref_train)} preference pairs, batch size {batch_size} -> {total_steps} steps")

    train_steps_log = []
    train_loss_log = []
    val_steps_log = []
    val_loss_log = []
    val_acc_log = []

    running_loss = 0.0
    start_time = time.time()

    model.train()
    for step, batch in enumerate(train_loader, start=1):
        ch_ids = batch["chosen_ids"].to(device)
        ch_mask = batch["chosen_mask"].to(device)
        rj_ids = batch["rejected_ids"].to(device)
        rj_mask = batch["rejected_mask"].to(device)

        optimizer.zero_grad()
        r_w = model(ch_ids, ch_mask)
        r_l = model(rj_ids, rj_mask)

        loss = compute_reward_loss(r_w, r_l)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if step % eval_every == 0 or step == total_steps:
            steps_in_window = eval_every if (step % eval_every == 0 and step != total_steps) else ((step - 1) % eval_every + 1)
            avg_train_loss = running_loss / steps_in_window
            val_loss, val_acc, _ = evaluate_reward_model(model, val_loader, device)

            train_steps_log.append(step)
            train_loss_log.append(avg_train_loss)
            val_steps_log.append(step)
            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)

            elapsed = time.time() - start_time
            print(f"Step [{step:4d}/{total_steps}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pairwise Acc: {val_acc * 100:.2f}% | Elapsed: {elapsed:.1f}s")
            running_loss = 0.0
            model.train()

    total_duration = time.time() - start_time
    print(f"Reward Model Training completed in {total_duration:.1f}s")

    # Final evaluation and collect misclassified pairs for Task 7.1
    final_val_loss, final_val_acc, misclassified_pairs = evaluate_reward_model(
        model, val_loader, device, collect_errors=True
    )

    # Save reward model checkpoint
    print(f"Saving Reward Model weights to {checkpoint_dir}...")
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "reward_model.pt"))
    tok.save_pretrained(checkpoint_dir)

    # Plot Figure 2: Reward Model Training Loss and Validation Accuracy
    fig_path = os.path.join(results_dir, "fig2_reward_loss_acc.png")
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss (Bradley-Terry)", color=color)
    ax1.plot(train_steps_log, train_loss_log, marker="o", color=color, label="Train Loss")
    ax1.plot(val_steps_log, val_loss_log, marker="s", color="tab:cyan", linestyle="--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Validation Pairwise Accuracy", color=color)
    ax2.plot(val_steps_log, val_acc_log, marker="^", color=color, label="Val Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Figure 2: Reward Model Loss and Pairwise Accuracy")
    fig.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved Figure 2 to {fig_path}")

    # Save metrics and misclassified samples for Table 2 and Task 7.1
    metrics = {
        "stage": "Reward Model",
        "final_train_loss": float(train_loss_log[-1]),
        "final_val_loss": float(final_val_loss),
        "final_pairwise_accuracy": float(final_val_acc),
        "total_time_seconds": round(total_duration, 2),
        "gpu": gpu_name,
        "steps_log": train_steps_log,
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
        "val_acc_log": val_acc_log,
    }
    metrics_path = os.path.join(results_dir, "reward_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    misclassified_path = os.path.join(results_dir, "reward_misclassified_sample.json")
    with open(misclassified_path, "w") as f:
        json.dump(misclassified_pairs[:5], f, indent=2)

    return metrics