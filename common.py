import torch

def token_logprobs(model, input_ids, attention_mask):
    """
    Computes log pi(x_t | x_{<t}) for every position t >= 1.
    Memory-efficient implementation: uses selected_logits - logsumexp(shift_logits)
    to avoid materializing a massive (B, L - 1, vocab_size) tensor.
    
    Args:
        model: Causal language model
        input_ids: Tensor of shape (B, L)
        attention_mask: Tensor of shape (B, L)
        
    Returns:
        token_lp: Tensor of shape (B, L - 1)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (B, L, vocab_size)

    # Position t - 1 predicts token at position t
    shift_logits = logits[:, :-1, :]  # shape: (B, L - 1, vocab_size)
    shift_labels = input_ids[:, 1:].unsqueeze(-1)  # shape: (B, L - 1, 1)

    # Gather logit of the actual target token: shape (B, L - 1)
    selected_logits = torch.gather(shift_logits, dim=-1, index=shift_labels).squeeze(-1)
    # Log-sum-exp along vocabulary dimension: shape (B, L - 1)
    lse = torch.logsumexp(shift_logits, dim=-1)

    token_lp = selected_logits - lse
    return token_lp


def get_summary_mask(input_ids, prompt_lens, attention_mask):
    """
    Builds a binary mask of shape (B, L) that is 1 exactly on summary tokens
    (excluding prompt and right-padding).
    """
    B, L = input_ids.shape
    device = input_ids.device
    col_indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    prompt_lens_2d = prompt_lens.unsqueeze(1).to(device)

    mask = (col_indices >= prompt_lens_2d) & (attention_mask == 1)
    return mask.long()


@torch.no_grad()
def generate_summaries(
    model,
    tokenizer,
    prompt_ids,
    attention_mask,
    max_new_tokens=48,
    do_sample=False,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    """
    Generates summary completions from left-padded prompts.
    """
    model.eval()
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })

    outputs = model.generate(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    prompt_len = prompt_ids.shape[1]
    generated_tokens = outputs[:, prompt_len:]
    return outputs, generated_tokens