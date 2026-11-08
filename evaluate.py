import os
import json
import gc
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from rouge_score import rouge_scorer

from starter.models import load_tokenizer
from train_reward import RewardModel


def clean_summary_text(decoded_text, eos_token="<|endoftext|>"):
    """Truncates generated text at the first EOS token and strips excess whitespace."""
    if eos_token in decoded_text:
        decoded_text = decoded_text.split(eos_token)[0]
    return decoded_text.strip()


def extract_prompt_ids(item, tokenizer):
    """Extracts prompt token IDs from a test split item."""
    if isinstance(item, dict):
        if "prompt_ids" in item:
            p = item["prompt_ids"]
            return torch.tensor(p, dtype=torch.long) if not torch.is_tensor(p) else p
        if "prompt_tokens" in item:
            p = item["prompt_tokens"]
            return torch.tensor(p, dtype=torch.long) if not torch.is_tensor(p) else p
        if "input_ids" in item and "prompt_len" in item:
            p = item["input_ids"][:item["prompt_len"]]
            return torch.tensor(p, dtype=torch.long) if not torch.is_tensor(p) else p
        if "prompt" in item:
            ids = tokenizer.encode(item["prompt"], truncation=True, max_length=448)
            return torch.tensor(ids, dtype=torch.long)
        if "post" in item:
            p_text = item["post"].strip() + "\nTL;DR:"
            ids = tokenizer.encode(p_text, truncation=True, max_length=448)
            return torch.tensor(ids, dtype=torch.long)
    raise ValueError(f"Unable to extract prompt from item: {type(item)}")


def extract_reference_text(item, tokenizer):
    """Extracts human reference summary string from a test split item."""
    if isinstance(item, dict):
        if "reference" in item and isinstance(item["reference"], str):
            return item["reference"]
        if "summary" in item and isinstance(item["summary"], str):
            return item["summary"]
        if "reference_text" in item and isinstance(item["reference_text"], str):
            return item["reference_text"]
        if "summary_ids" in item:
            return tokenizer.decode(item["summary_ids"], skip_special_tokens=True)
        if "summary_tokens" in item:
            return tokenizer.decode(item["summary_tokens"], skip_special_tokens=True)
        if "reference_ids" in item:
            return tokenizer.decode(item["reference_ids"], skip_special_tokens=True)
        if "label_ids" in item:
            return tokenizer.decode(item["label_ids"], skip_special_tokens=True)
    return ""


@torch.no_grad()
def generate_summaries(model, tokenizer, test_posts, device, max_new_tokens=48):
    """Generates greedy summaries for all test prompts."""
    model.eval()
    generated_summaries = []

    for item in tqdm(test_posts, desc="Generating summaries"):
        p_ids = extract_prompt_ids(item, tokenizer)
        input_ids = p_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        prompt_len = input_ids.shape[1]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_tokens = outputs[0, prompt_len:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        clean_text = clean_summary_text(decoded, eos_token="<|endoftext|>")

        # Measure token length up to the first EOS token
        eos_indices = (gen_tokens == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if len(eos_indices) > 0:
            token_len = eos_indices[0].item() + 1
        else:
            token_len = len(gen_tokens)

        generated_summaries.append({
            "text": clean_text,
            "token_length": token_len,
        })

    return generated_summaries


@torch.no_grad()
def score_summaries_with_reward_model(reward_model, tokenizer, test_posts, summaries, device, batch_size=16):
    """Computes r_ϕ(x, y) scores for prompt + summary sequences."""
    reward_model.eval()
    scores = []
    pad = tokenizer.pad_token_id

    for i in range(0, len(test_posts), batch_size):
        batch_posts = test_posts[i:i + batch_size]
        batch_sums = summaries[i:i + batch_size]
        seq_tensors = []

        for p_item, s_item in zip(batch_posts, batch_sums):
            p_ids = extract_prompt_ids(p_item, tokenizer)
            s_text = " " + s_item["text"].strip() + "<|endoftext|>"
            s_ids = torch.tensor(tokenizer.encode(s_text, add_special_tokens=False), dtype=torch.long)
            seq_tensors.append(torch.cat([p_ids.cpu(), s_ids]))

        max_len = max(len(t) for t in seq_tensors)
        B = len(seq_tensors)
        input_ids = torch.full((B, max_len), pad, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)

        for j in range(B):
            l = len(seq_tensors[j])
            input_ids[j, :l] = seq_tensors[j]
            attention_mask[j, :l] = 1

        batch_scores = reward_model(input_ids, attention_mask)
        scores.extend(batch_scores.cpu().tolist())

    return scores


def run_full_evaluation(
    data_path="/vol/data/processed_splits.pt",
    base_model_name="HuggingFaceTB/SmolLM2-135M",
    sft_checkpoint="/vol/checkpoints/sft",
    reward_checkpoint="/vol/checkpoints/reward",
    ppo_checkpoint="/vol/checkpoints/ppo",
    dpo_checkpoint="/vol/checkpoints/dpo",
    results_dir="/vol/results/eval",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Task 6 Comprehensive Evaluation on {device}...")
    os.makedirs(results_dir, exist_ok=True)

    tok = load_tokenizer()
    splits = torch.load(data_path, weights_only=False)
    print(f"Available keys in {data_path}: {list(splits.keys())}")

    # Resolve test split key
    test_key = None
    for candidate in ["sft_test", "test_posts", "test", "eval_test", "eval_posts", "eval"]:
        if candidate in splits and len(splits[candidate]) > 0:
            test_key = candidate
            break

    if test_key is None:
        for k in splits.keys():
            if "test" in k.lower() and len(splits[k]) > 0:
                test_key = k
                break

    if test_key is None:
        raise KeyError(f"Could not locate 300 test posts in splits! Available keys: {list(splits.keys())}")

    test_posts = splits[test_key]
    print(f"Loaded {len(test_posts)} held-out evaluation test posts from key '{test_key}'.")

    # Extract human references
    references = [clean_summary_text(extract_reference_text(item, tok)) for item in test_posts]

    models_to_evaluate = [
        ("Base", base_model_name),
        ("SFT", sft_checkpoint),
        ("PPO", ppo_checkpoint),
        ("DPO", dpo_checkpoint),
    ]

    all_generations = {}

    for name, path in models_to_evaluate:
        print(f"\n--- Generating summaries for: {name} ({path}) ---")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).to(device)

        gens = generate_summaries(model, tok, test_posts, device)
        all_generations[name] = gens

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Score with Reward Model
    print(f"\n--- Scoring all candidate summaries with Reward Model ---")
    reward_model = RewardModel(sft_checkpoint).to(device)
    rm_weights = os.path.join(reward_checkpoint, "reward_model.pt")
    reward_model.load_state_dict(torch.load(rm_weights, weights_only=True))

    all_rewards = {}
    for name in all_generations:
        all_rewards[name] = score_summaries_with_reward_model(
            reward_model, tok, test_posts, all_generations[name], device
        )

    # Score Human References
    ref_summaries = [{"text": r} for r in references]
    human_rewards = score_summaries_with_reward_model(
        reward_model, tok, test_posts, ref_summaries, device
    )
    all_rewards["Human Reference"] = human_rewards

    del reward_model
    gc.collect()
    torch.cuda.empty_cache()

    # Compute ROUGE scores
    print("\n--- Computing ROUGE metrics against human references ---")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    table1_data = []

    for name in ["Base", "SFT", "PPO", "DPO"]:
        r1_list, r2_list, rl_list = [], [], []
        gens = all_generations[name]

        for gen_item, ref_text in zip(gens, references):
            scores = scorer.score(ref_text, gen_item["text"])
            r1_list.append(scores["rouge1"].fmeasure * 100)
            r2_list.append(scores["rouge2"].fmeasure * 100)
            rl_list.append(scores["rougeL"].fmeasure * 100)

        mean_r1 = sum(r1_list) / len(r1_list)
        mean_r2 = sum(r2_list) / len(r2_list)
        mean_rl = sum(rl_list) / len(rl_list)
        mean_len = sum(g["token_length"] for g in gens) / len(gens)
        mean_rm = sum(all_rewards[name]) / len(all_rewards[name])

        table1_data.append({
            "Model": name,
            "ROUGE-1": round(mean_r1, 2),
            "ROUGE-2": round(mean_r2, 2),
            "ROUGE-L": round(mean_rl, 2),
            "Mean Length": round(mean_len, 1),
            "Mean Reward": round(mean_rm, 3),
        })

    # Human Reference baseline metrics
    ref_lens = [len(tok.encode(r, add_special_tokens=False)) for r in references]
    table1_data.append({
        "Model": "Human Reference",
        "ROUGE-1": 100.0,
        "ROUGE-2": 100.0,
        "ROUGE-L": 100.0,
        "Mean Length": round(sum(ref_lens) / max(len(ref_lens), 1), 1),
        "Mean Reward": round(sum(human_rewards) / max(len(human_rewards), 1), 3),
    })

    # Compute Pairwise Win Rates (% where r_phi(row) > r_phi(col))
    model_keys = ["Base", "SFT", "PPO", "DPO"]
    win_rate_matrix = {m1: {} for m1 in model_keys}
    N = len(test_posts)

    for m1 in model_keys:
        for m2 in model_keys:
            if m1 == m2:
                win_rate_matrix[m1][m2] = 50.0
            else:
                wins = sum(1 for i in range(N) if all_rewards[m1][i] > all_rewards[m2][i])
                win_rate_matrix[m1][m2] = round((wins / N) * 100, 2)

    # Save Markdown Table 1 and Table 2
    md_table_path = os.path.join(results_dir, "table1_results.md")
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# Table 1: Automatic Evaluation on 300 Held-Out Test Posts\n\n")
        f.write("| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean Length (Tokens) | Mean Reward $r_\\phi$ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for row in table1_data:
            f.write(f"| {row['Model']} | {row['ROUGE-1']:.2f} | {row['ROUGE-2']:.2f} | {row['ROUGE-L']:.2f} | {row['Mean Length']:.1f} | {row['Mean Reward']:+.3f} |\n")

        f.write("\n\n# Table 2: Pairwise Win Rates (% Row beats Column under $r_\\phi$)\n\n")
        f.write("| vs. | Base | SFT | PPO | DPO |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for m1 in model_keys:
            f.write(f"| **{m1}** | {win_rate_matrix[m1]['Base']:.1f}% | {win_rate_matrix[m1]['SFT']:.1f}% | {win_rate_matrix[m1]['PPO']:.1f}% | {win_rate_matrix[m1]['DPO']:.1f}% |\n")

    print(f"Saved Markdown report to {md_table_path}")

    # Save JSON summary metrics
    summary_path = os.path.join(results_dir, "eval_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "table1": table1_data,
            "win_rate_matrix": win_rate_matrix,
        }, f, indent=2)

    # Save full JSONL dataset
    jsonl_path = os.path.join(results_dir, "eval_summaries.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(N):
            p_ids = extract_prompt_ids(test_posts[i], tok)
            prompt_full = tok.decode(p_ids, skip_special_tokens=True)
            entry = {
                "test_index": i + 1,
                "prompt_tail": prompt_full[-300:],
                "human_reference": references[i],
                "base_summary": all_generations["Base"][i]["text"],
                "base_reward": round(all_rewards["Base"][i], 4),
                "sft_summary": all_generations["SFT"][i]["text"],
                "sft_reward": round(all_rewards["SFT"][i], 4),
                "ppo_summary": all_generations["PPO"][i]["text"],
                "ppo_reward": round(all_rewards["PPO"][i], 4),
                "dpo_summary": all_generations["DPO"][i]["text"],
                "dpo_reward": round(all_rewards["DPO"][i], 4),
            }
            f.write(json.dumps(entry) + "\n")

    # Save Qualitative Comparisons (First 5 samples formatted in Markdown)
    qual_path = os.path.join(results_dir, "qualitative_comparisons.md")
    with open(qual_path, "w", encoding="utf-8") as f:
        f.write("# Task 6 Qualitative Comparisons: Base vs. SFT vs. PPO vs. DPO\n\n")
        for i in range(min(5, N)):
            p_ids = extract_prompt_ids(test_posts[i], tok)
            p_tail = tok.decode(p_ids, skip_special_tokens=True)[-250:]
            f.write(f"### Sample {i + 1}\n")
            f.write(f"**Prompt (Tail):** ...{p_tail}\n\n")
            f.write(f"**Human Reference:** {references[i]} *(Reward: {all_rewards['Human Reference'][i]:+.3f})*\n\n")
            f.write(f"- **Base Model:** {all_generations['Base'][i]['text']} *(Reward: {all_rewards['Base'][i]:+.3f})*\n")
            f.write(f"- **SFT (Stage 1):** {all_generations['SFT'][i]['text']} *(Reward: {all_rewards['SFT'][i]:+.3f})*\n")
            f.write(f"- **PPO (Stage 3):** {all_generations['PPO'][i]['text']} *(Reward: {all_rewards['PPO'][i]:+.3f})*\n")
            f.write(f"- **DPO (Stage 4):** {all_generations['DPO'][i]['text']} *(Reward: {all_rewards['DPO'][i]:+.3f})*\n\n")
            f.write("---\n\n")

    print(f"Saved qualitative comparisons to {qual_path}")
    return table1_data, win_rate_matrix