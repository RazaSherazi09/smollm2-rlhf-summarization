import os
import time
import json
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from starter.models import load_tokenizer
from starter.batching import collate_pref


def compute_summary_logprobs(model, input_ids, attention_mask, prompt_lens, amp_dtype):
    """
    Computes log π(y | x) = sum of token log-probabilities over summary tokens only (Task 5.2).
    Uses selected_logits - logsumexp for memory efficiency.
    """
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (B, L, V)

    # Shift so logits predict next token
    shift_logits = logits[:, :-1, :]  # (B, L-1, V)
    shift_tokens = input_ids[:, 1:]   # (B, L-1)

    # Selected log-probs: selected_logits - logsumexp
    sel_logits = shift_logits.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(shift_logits.float(), dim=-1)
    token_logprobs = sel_logits.float() - lse  # (B, L-1)

    # Build binary mask isolating summary tokens
    B, L_minus_1 = shift_tokens.shape
    summary_mask = torch.zeros((B, L_minus_1), dtype=torch.float32, device=input_ids.device)

    total_lens = attention_mask.sum(dim=1)  # Non-padding token count
    for i in range(B):
        p_len = prompt_lens[i].item() if torch.is_tensor(prompt_lens[i]) else prompt_lens[i]
        tot_len = total_lens[i].item() if torch.is_tensor(total_lens[i]) else total_lens[i]
        # In shift_tokens (offset by 1), summary tokens span [p_len - 1, tot_len - 1)
        start_idx = max(0, p_len - 1)
        end_idx = max(0, tot_len - 1)
        if end_idx > start_idx:
            summary_mask[i, start_idx:end_idx] = 1.0

    # Sum of log-probabilities over summary tokens only
    seq_logprobs = (token_logprobs * summary_mask).sum(dim=-1)  # shape: (B,)
    return seq_logprobs


def compute_dpo_loss(
    pi_chosen_logps,
    pi_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """
    DPO Loss (Eq. 60):
    L_DPO(θ) = - E [ log σ( β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)) ) ]
    """
    pi_logratios = pi_chosen_logps - pi_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    implicit_logits = beta * (pi_logratios - ref_logratios)

    loss = -F.logsigmoid(implicit_logits).mean()

    with torch.no_grad():
        # Implicit reward accuracy: fraction where the expression inside σ(·) is positive (Task 5.5)
        accuracy = (implicit_logits > 0).float().mean().item()

    return loss, accuracy, implicit_logits


@torch.no_grad()
def evaluate_dpo(policy, ref_policy, val_loader, device, amp_dtype, beta=0.1):
    policy.eval()
    total_loss = 0.0
    total_correct = 0
    total_pairs = 0

    for batch in val_loader:
        ch_ids = batch["chosen_ids"].to(device)
        ch_mask = batch["chosen_mask"].to(device)
        rj_ids = batch["rejected_ids"].to(device)
        rj_mask = batch["rejected_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)
        B = ch_ids.shape[0]

        pi_chosen_logps = compute_summary_logprobs(policy, ch_ids, ch_mask, prompt_lens, amp_dtype)
        pi_rejected_logps = compute_summary_logprobs(policy, rj_ids, rj_mask, prompt_lens, amp_dtype)
        ref_chosen_logps = compute_summary_logprobs(ref_policy, ch_ids, ch_mask, prompt_lens, amp_dtype)
        ref_rejected_logps = compute_summary_logprobs(ref_policy, rj_ids, rj_mask, prompt_lens, amp_dtype)

        loss, _, logits = compute_dpo_loss(
            pi_chosen_logps,
            pi_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )

        total_loss += loss.item() * B
        total_correct += (logits > 0).sum().item()
        total_pairs += B

    avg_loss = total_loss / max(total_pairs, 1)
    avg_acc = total_correct / max(total_pairs, 1)
    return avg_loss, avg_acc


def train_dpo(
    data_path="/vol/data/processed_splits.pt",
    sft_checkpoint="/vol/checkpoints/sft",
    checkpoint_dir="/vol/checkpoints/dpo",
    results_dir="/vol/results",
    batch_size=8,
    lr=2e-6,
    beta=0.1,
    grad_clip=1.0,
    eval_every=250,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Starting Stage 4: DPO on device: {device} ({gpu_name})")

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

    # 5.1 Load SFT checkpoint twice: trainable policy π_θ and frozen reference π_ref
    print("Loading trainable Policy π_θ from SFT checkpoint...")
    policy = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)

    print("Loading frozen Reference π_ref from SFT checkpoint...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    ref_policy.eval()
    ref_policy.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    total_steps = len(train_loader)
    print(f"Training 1 epoch: {len(pref_train)} preference pairs, batch size {batch_size} -> {total_steps} steps")

    train_steps_log = []
    train_loss_log = []
    val_steps_log = []
    val_loss_log = []
    val_acc_log = []

    running_loss = 0.0
    start_time = time.time()

    policy.train()
    for step, batch in enumerate(train_loader, start=1):
        ch_ids = batch["chosen_ids"].to(device)
        ch_mask = batch["chosen_mask"].to(device)
        rj_ids = batch["rejected_ids"].to(device)
        rj_mask = batch["rejected_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        # Policy forward passes (with gradients)
        pi_chosen_logps = compute_summary_logprobs(policy, ch_ids, ch_mask, prompt_lens, amp_dtype)
        pi_rejected_logps = compute_summary_logprobs(policy, rj_ids, rj_mask, prompt_lens, amp_dtype)

        # Reference forward passes (frozen, without gradients)
        with torch.no_grad():
            ref_chosen_logps = compute_summary_logprobs(ref_policy, ch_ids, ch_mask, prompt_lens, amp_dtype)
            ref_rejected_logps = compute_summary_logprobs(ref_policy, rj_ids, rj_mask, prompt_lens, amp_dtype)

        loss, _, _ = compute_dpo_loss(
            pi_chosen_logps,
            pi_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()

        # 5.5 Every 250 steps log loss and implicit reward accuracy on 1,000 val pairs
        if step % eval_every == 0 or step == total_steps:
            steps_in_window = eval_every if (step % eval_every == 0 and step != total_steps) else ((step - 1) % eval_every + 1)
            avg_train_loss = running_loss / steps_in_window
            val_loss, val_acc = evaluate_dpo(policy, ref_policy, val_loader, device, amp_dtype, beta=beta)

            train_steps_log.append(step)
            train_loss_log.append(avg_train_loss)
            val_steps_log.append(step)
            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)

            elapsed = time.time() - start_time
            print(
                f"Step [{step:4d}/{total_steps}] | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Implicit Acc: {val_acc * 100:.2f}% | "
                f"Elapsed: {elapsed:.1f}s"
            )
            running_loss = 0.0
            policy.train()

    total_duration = time.time() - start_time
    print(f"DPO Training completed in {total_duration:.1f}s")

    # Save fine-tuned DPO policy checkpoint (Task 5.5)
    print(f"Saving DPO policy to {checkpoint_dir}...")
    policy.save_pretrained(checkpoint_dir)
    tok.save_pretrained(checkpoint_dir)

    # Plot Figure 4: DPO Training Loss and Validation Implicit Reward Accuracy
    fig_path = os.path.join(results_dir, "fig4_dpo_loss_acc.png")
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss (DPO)", color=color)
    ax1.plot(train_steps_log, train_loss_log, marker="o", color=color, label="Train Loss")
    ax1.plot(val_steps_log, val_loss_log, marker="s", color="tab:cyan", linestyle="--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Validation Implicit Accuracy", color=color)
    ax2.plot(val_steps_log, val_acc_log, marker="^", color=color, label="Val Implicit Acc")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Figure 4: DPO Loss and Implicit Reward Accuracy")
    fig.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved Figure 4 to {fig_path}")

    # Save Table 2 Metrics and full iteration logs
    metrics = {
        "stage": "DPO",
        "final_train_loss": float(train_loss_log[-1]),
        "final_val_loss": float(val_loss_log[-1]),
        "final_implicit_accuracy": float(val_acc_log[-1]),
        "total_time_seconds": round(total_duration, 2),
        "gpu": gpu_name,
        "steps_log": train_steps_log,
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
        "val_acc_log": val_acc_log,
    }
    metrics_path = os.path.join(results_dir, "dpo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved DPO metrics to {metrics_path}")

    return metrics