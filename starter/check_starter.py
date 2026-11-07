from functools import partial
import numpy as np
from torch.utils.data import DataLoader

try:
    from starter.models import load_tokenizer
    from starter.data import load_sft_splits, load_preference_splits, load_eval_posts
    from starter.batching import collate_sft, collate_pref, collate_prompts
except ImportError:
    from models import load_tokenizer
    from data import load_sft_splits, load_preference_splits, load_eval_posts
    from batching import collate_sft, collate_pref, collate_prompts

def run_checks():
    tok = load_tokenizer()
    print("Loading SFT splits...")
    sft_train, sft_val = load_sft_splits(tok)
    print("Loading preference splits...")
    pref_train, pref_val = load_preference_splits(tok)
    print("Loading eval posts...")
    eval_posts = load_eval_posts(tok)

    print("\n--- Task 1.1 / Table 1: Dataset Sizes ---")
    print(f"SFT train / val pairs:        {len(sft_train)} / {len(sft_val)}")
    print(f"Preference train / val pairs: {len(pref_train)} / {len(pref_val)}")
    print(f"Evaluation posts:             {len(eval_posts)}")

    # Length statistics for SFT train (Task 1.2 & Table 1)
    prompt_lens = [len(x["prompt_ids"]) for x in sft_train]
    summary_lens = [len(x["summary_ids"]) for x in sft_train]

    print("\n--- Task 1.2: Token Length Statistics (SFT Train) ---")
    print(f"Prompt length  -> Mean: {np.mean(prompt_lens):.2f}, Max: {np.max(prompt_lens)}")
    print(f"Summary length -> Mean: {np.mean(summary_lens):.2f}, Max: {np.max(summary_lens)}")

    ex = sft_train[0]
    print("\n--- One SFT Example ---")
    print("PROMPT (tail):", ex["prompt"][-250:])
    print("SUMMARY:      ", repr(ex["summary"]))

    pad = tok.pad_token_id
    b_sft = next(iter(DataLoader(sft_train, 8, shuffle=True, collate_fn=partial(collate_sft, pad_id=pad))))
    print("\n--- Batching Sanity Checks ---")
    print("SFT batch shape:    ", tuple(b_sft["input_ids"].shape), "Prompt lens:", b_sft["prompt_lens"].tolist())

    b_pref = next(iter(DataLoader(pref_train, 8, shuffle=True, collate_fn=partial(collate_pref, pad_id=pad))))
    print("Pref batch shapes:  ", tuple(b_pref["chosen_ids"].shape), tuple(b_pref["rejected_ids"].shape))

    b_eval = next(iter(DataLoader(eval_posts, 8, collate_fn=partial(collate_prompts, pad_id=pad))))
    print("Prompt batch shape: ", tuple(b_eval["input_ids"].shape), "(left-padded)")

    return {
        "sft_train": sft_train,
        "sft_val": sft_val,
        "pref_train": pref_train,
        "pref_val": pref_val,
        "eval_posts": eval_posts,
    }

if __name__ == "__main__":
    run_checks()