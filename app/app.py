import os
import gc
import html
import torch
import torch.nn as nn
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmolLM2 | Summary Studio",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
HF_REPO_ID = "razaasherazi/smollm2-135m-rlhf-checkpoints"

MAX_INPUT_LENGTH = 448
MAX_NEW_TOKENS = 48
MAX_REWARD_LENGTH = 512

# Streamlit Community Cloud is CPU-only for this deployment.
DEVICE = "cpu"
MODEL_DTYPE = torch.float32


# ============================================================
# EDITORIAL / BRUTALIST UI
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --paper:#f5f1e7; --ink:#17151d; --pink:#f45d91;
        --yellow:#ffd83d; --blue:#a8d9f5; --green:#a9e878;
        --muted:#716d68; --white:#fffdf8;
    }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stMainBlockContainer"] { max-width:1320px; padding:1.3rem 2.4rem 4rem; }
    #MainMenu, footer { visibility:hidden; }
    h1,h2,h3,p,label,span,div { font-family:'DM Sans',sans-serif; }

    .brandbar { display:flex; align-items:center; justify-content:space-between; padding:4px 0 26px; border-bottom:2px solid var(--ink); margin-bottom:40px; }
    .brand-left { display:flex; align-items:center; gap:10px; }
    .brand-mark { width:30px; height:30px; display:flex; align-items:center; justify-content:center; background:var(--pink); border:2px solid var(--ink); box-shadow:3px 3px 0 var(--ink); font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:900; }
    .brand-name { font-family:'DM Mono',monospace; font-size:11px; letter-spacing:1.3px; }
    .online-pill { border:2px solid var(--ink); background:var(--green); box-shadow:3px 3px 0 var(--ink); padding:7px 12px; font-family:'DM Mono',monospace; font-size:9px; letter-spacing:1px; text-transform:uppercase; }
    .online-dot { display:inline-block; width:7px; height:7px; background:#277b43; border-radius:50%; margin-right:7px; }

    .hero-kicker,.eyebrow { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1.5px; text-transform:uppercase; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(3.6rem,7.4vw,6.6rem); font-weight:700; line-height:.86; letter-spacing:-.075em; text-transform:uppercase; color:var(--ink); margin:22px 0 24px; }
    .hero-title .pink,.studio-title .pink { color:var(--pink); }
    .hero-copy { max-width:440px; color:#48444a; font-size:15px; line-height:1.65; }
    .hero-art { position:relative; min-height:300px; display:flex; align-items:center; justify-content:center; padding:25px; }
    .code-window { width:100%; max-width:470px; background:var(--ink); color:#f8f3e9; border:2px solid var(--ink); box-shadow:10px 10px 0 #9275ef; transform:rotate(-2deg); }
    .code-top { padding:12px 16px; border-bottom:1px solid #494550; color:#c4bacd; font-family:'DM Mono',monospace; font-size:10px; }
    .code-body { padding:25px 22px 27px; font-family:'DM Mono',monospace; font-size:clamp(12px,1.4vw,16px); line-height:2; }
    .code-purple { color:#9c88ff; } .code-pink { color:#ff6f9e; } .code-green { color:#a9e878; } .code-yellow { color:#ffd83d; }
    .sticker { position:absolute; z-index:2; width:82px; height:82px; border-radius:50%; display:flex; align-items:center; justify-content:center; text-align:center; transform:rotate(-12deg); background:var(--yellow); border:2px solid var(--ink); box-shadow:4px 4px 0 var(--ink); font-family:'DM Mono',monospace; font-size:10px; line-height:1.25; top:8px; left:2%; }
    .mini-note { position:absolute; right:2%; bottom:0; padding:12px 10px; background:var(--blue); border:2px solid var(--ink); box-shadow:4px 4px 0 var(--ink); transform:rotate(7deg); font-family:'DM Mono',monospace; font-size:10px; line-height:1.35; }

    .feature-strip { display:grid; grid-template-columns:repeat(3,1fr); border-top:2px solid var(--ink); border-bottom:2px solid var(--ink); margin:38px 0 58px; }
    .feature { display:flex; align-items:center; gap:14px; min-height:64px; padding:12px 15px; border-right:2px solid var(--ink); font-family:'DM Mono',monospace; font-size:11px; line-height:1.4; }
    .feature:last-child { border-right:0; } .feature-num { color:var(--pink); font-size:16px; font-weight:700; }

    .studio-grid { display:grid; grid-template-columns:.85fr 1.15fr; gap:7%; align-items:start; margin:0 0 24px; }
    .studio-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.7rem,5vw,4.6rem); line-height:.9; letter-spacing:-.07em; font-weight:700; text-transform:uppercase; margin:18px 0 20px; }
    .studio-description { max-width:350px; color:#5e5955; font-size:14px; line-height:1.7; }
    .tip { margin-top:28px; display:flex; gap:10px; align-items:flex-start; max-width:340px; font-family:'DM Mono',monospace; font-size:10px; line-height:1.6; color:#55504d; }
    .tip-mark { flex:0 0 20px; width:20px; height:20px; display:flex; align-items:center; justify-content:center; border:1.5px solid var(--ink); border-radius:50%; background:var(--yellow); font-weight:700; }

    [data-testid="stVerticalBlockBorderWrapper"] { background:var(--white); border:2px solid var(--ink); border-radius:0; box-shadow:5px 5px 0 var(--ink); }
    .form-panel-top { display:flex; justify-content:space-between; padding-bottom:14px; margin-bottom:16px; border-bottom:1.5px solid var(--ink); font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1px; }
    .required { color:#b94d72; }
    .stTextArea label,.stTextInput label { color:var(--ink)!important; font-size:12px!important; font-weight:700!important; }
    .stTextArea textarea,.stTextInput input { background:var(--paper)!important; color:var(--ink)!important; border:1.5px solid var(--ink)!important; border-radius:0!important; font-size:13px!important; box-shadow:none!important; }
    .stTextArea textarea:focus,.stTextInput input:focus { border:2px solid var(--pink)!important; box-shadow:3px 3px 0 var(--pink)!important; }
    .stButton > button { background:var(--pink)!important; color:var(--ink)!important; border:2px solid var(--ink)!important; border-radius:0!important; box-shadow:4px 4px 0 var(--ink)!important; font-family:'DM Sans',sans-serif!important; font-weight:900!important; letter-spacing:.5px; min-height:48px; }
    .stButton > button:hover { transform:translate(2px,2px); box-shadow:2px 2px 0 var(--ink)!important; color:var(--ink)!important; }
    .stProgress > div > div > div > div { background:var(--pink); }

    .results-heading { display:flex; justify-content:space-between; align-items:end; gap:20px; border-bottom:2px solid var(--ink); padding-bottom:14px; margin:42px 0 22px; }
    .results-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2rem,4vw,3.4rem); font-weight:700; letter-spacing:-.06em; text-transform:uppercase; line-height:1; }
    .results-tag { font-family:'DM Mono',monospace; font-size:10px; padding:7px 9px; border:1.5px solid var(--ink); background:var(--blue); white-space:nowrap; }
    .result-card { background:var(--white); border:2px solid var(--ink); box-shadow:5px 5px 0 var(--ink); padding:19px; min-height:300px; margin-bottom:8px; }
    .result-top { display:flex; justify-content:space-between; align-items:center; gap:8px; border-bottom:1.5px solid var(--ink); padding-bottom:13px; margin-bottom:15px; }
    .result-model { font-family:'Space Grotesk',sans-serif; font-size:23px; font-weight:700; letter-spacing:-1px; }
    .result-stage { font-family:'DM Mono',monospace; font-size:9px; padding:5px 7px; border:1px solid var(--ink); background:var(--yellow); }
    .result-method { font-family:'DM Mono',monospace; font-size:9px; color:#77716c; text-transform:uppercase; margin-bottom:12px; }
    .result-summary { color:#343039; font-size:13px; line-height:1.7; min-height:115px; overflow-wrap:anywhere; }
    .result-scoreline { border-top:1.5px solid var(--ink); margin-top:17px; padding-top:13px; display:flex; justify-content:space-between; align-items:end; }
    .result-score-label { font-family:'DM Mono',monospace; font-size:9px; color:#77716c; text-transform:uppercase; }
    .result-score { font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; letter-spacing:-1px; }
    .result-delta { font-family:'DM Mono',monospace; font-size:10px; margin-top:5px; }
    .delta-positive { color:#26774d; } .delta-negative { color:#b53d5c; }
    .empty-result { background:#fffdf8; border:2px dashed #8c857d; padding:30px; text-align:center; font-family:'DM Mono',monospace; color:#77716c; font-size:11px; line-height:1.8; }
    .score-note { background:#fff3c2; border:1.5px solid var(--ink); padding:12px 15px; margin-top:20px; font-size:11px; line-height:1.6; color:#514a3c; }
    .footer { margin-top:70px; padding-top:16px; border-top:2px solid var(--ink); display:flex; justify-content:space-between; gap:15px; color:#716d68; font-family:'DM Mono',monospace; font-size:9px; line-height:1.6; letter-spacing:.4px; }

    @media(max-width:850px) {
        [data-testid="stMainBlockContainer"] { padding:1rem 1.1rem 3rem; }
        .studio-grid { grid-template-columns:1fr; gap:20px; }
        .hero-art { min-height:250px; margin-top:15px; }
        .feature-strip { grid-template-columns:1fr; }
        .feature { border-right:0; border-bottom:1.5px solid var(--ink); }
        .feature:last-child { border-bottom:0; }
        .footer { flex-direction:column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHECKPOINTS AND MODEL LOADING
# ============================================================

def resolve_model_path(stage):
    """Return a local checkpoint directory or download that stage."""
    local_path = os.path.join("checkpoints", stage)
    if os.path.isdir(local_path) and os.listdir(local_path):
        return local_path

    downloaded_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=f"{stage}/*",
    )
    stage_path = os.path.join(downloaded_dir, stage)

    if not os.path.isdir(stage_path):
        raise FileNotFoundError(
            f"Checkpoint folder '{stage}' was not found in {HF_REPO_ID}."
        )

    return stage_path


def load_causal_model(stage):
    """Load only one generation model at a time."""
    checkpoint_path = resolve_model_path(stage)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True,
    )

    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def release_model(model):
    """Release model references and ask Python to reclaim memory."""
    if model is not None:
        del model
    gc.collect()


# ============================================================
# REWARD MODEL
# ============================================================

class RewardModel(nn.Module):
    """SFT transformer backbone plus a scalar reward head."""

    def __init__(self, base_causal_lm):
        super().__init__()
        self.transformer = base_causal_lm.model
        self.v_head = nn.Linear(
            base_causal_lm.config.hidden_size,
            1,
            bias=False,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_indices = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        batch_indices = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )

        last_hidden = outputs.last_hidden_state[
            batch_indices,
            last_indices,
        ]

        return self.v_head(last_hidden).squeeze(-1)


def load_reward_model():
    """
    Load the reward model only after all three summaries have been generated.
    Its backbone is independent of the SFT generation model.
    """
    sft_path = resolve_model_path("sft")
    reward_path = resolve_model_path("reward")

    backbone = AutoModelForCausalLM.from_pretrained(
        sft_path,
        dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True,
    )

    reward_model = RewardModel(backbone)
    reward_model.to(DEVICE)
    reward_model.eval()

    del backbone
    gc.collect()

    weights_file = os.path.join(reward_path, "reward_model.pt")
    if not os.path.isfile(weights_file):
        raise FileNotFoundError(
            f"Reward checkpoint not found: {weights_file}"
        )

    # mmap avoids eagerly copying the entire checkpoint into RAM where
    # supported by the installed PyTorch version.
    try:
        state_dict = torch.load(
            weights_file,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state_dict = torch.load(
            weights_file,
            map_location="cpu",
            weights_only=True,
        )

    reward_model.load_state_dict(state_dict, strict=True)

    del state_dict
    gc.collect()

    return reward_model


# ============================================================
# INFERENCE
# ============================================================

def generate_summary(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0, prompt_length:]

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    return text.strip()


def compute_reward_score(reward_model, tokenizer, prompt, summary):
    full_text = prompt.strip() + " " + summary.strip() + "<|endoftext|>"

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_REWARD_LENGTH,
    )

    with torch.inference_mode():
        score = reward_model(
            inputs["input_ids"],
            inputs["attention_mask"],
        ).item()

    return round(score, 3)


def generate_with_one_model(stage, tokenizer, prompt, progress, start, end):
    """
    Load -> generate -> delete.
    At no point are SFT, PPO and DPO generation models resident together.
    """
    model = None
    try:
        progress.progress(start, text=f"Loading {stage.upper()} checkpoint...")
        model = load_causal_model(stage)

        progress.progress(
            min(start + 5, end),
            text=f"Generating {stage.upper()} summary...",
        )
        summary = generate_summary(model, tokenizer, prompt)

        return summary
    finally:
        if model is not None:
            del model
        gc.collect()


def render_result_card(name, stage, method, summary, score, baseline=None):
    if baseline is None:
        delta_html = '<div class="result-delta">BASELINE MODEL</div>'
    else:
        delta = round(score - baseline, 3)
        css_class = "delta-positive" if delta >= 0 else "delta-negative"
        sign = "+" if delta >= 0 else ""
        delta_html = (
            f'<div class="result-delta {css_class}">'
            f'{sign}{delta:.3f} VS SFT BASELINE</div>'
        )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-top">
                <div class="result-model">{html.escape(name)}</div>
                <div class="result-stage">{html.escape(stage)}</div>
            </div>
            <div class="result-method">{html.escape(method)}</div>
            <div class="result-summary">{html.escape(summary) or "No summary was generated."}</div>
            <div class="result-scoreline">
                <div>
                    <div class="result-score-label">REWARD MODEL SCORE</div>
                    <div class="result-score">{score:+.3f}</div>
                    {delta_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BRAND BAR AND HERO
# ============================================================

st.markdown(
    """
    <div class="brandbar">
        <div class="brand-left">
            <div class="brand-mark">S</div>
            <div class="brand-name">SMOLLM2 / SUMMARY STUDIO</div>
        </div>
        <div class="online-pill"><span class="online-dot"></span> MODEL INTERFACE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.05, 1], gap="large")

with hero_left:
    st.markdown(
        """
        <div class="hero-kicker">TEXT → SUMMARY / ALIGNMENT LAB</div>
        <div class="hero-title">Make your<br>words<br><span class="pink">matter.</span></div>
        <div class="hero-copy">
            Explore how human-preference alignment changes the way a
            language model summarizes Reddit discussions. Compare
            supervised fine-tuning, PPO, and DPO — side by side.
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        """
        <div class="hero-art">
            <div class="sticker">LESS<br>NOISE.<br>MORE<br>MEANING.</div>
            <div class="code-window">
                <div class="code-top">alignment_pipeline.py / preview</div>
                <div class="code-body">
                    <span class="code-purple">INPUT</span> = <span class="code-green">reddit_post</span><br>
                    <span class="code-purple">SFT</span> → <span class="code-pink">PPO</span> → <span class="code-yellow">DPO</span><br><br>
                    <span class="code-purple">GENERATE</span>(post)<br>
                    <span class="code-purple">SCORE</span>(summary, reward_model)<br>
                    <span class="code-green">COMPARE</span>(all_outputs)
                </div>
            </div>
            <div class="mini-note">01<br>GENERATE<br>02<br>COMPARE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="feature-strip">
        <div class="feature"><span class="feature-num">01</span> Natural-language<br>Reddit input</div>
        <div class="feature"><span class="feature-num">02</span> Three alignment<br>approaches</div>
        <div class="feature"><span class="feature-num">03</span> Reward-scored<br>summaries</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOKENIZER
# ============================================================

try:
    tokenizer = load_tokenizer()
except Exception as error:
    st.error("The tokenizer could not be initialized.")
    with st.expander("Technical details"):
        st.exception(error)
    st.stop()


# ============================================================
# QUERY STUDIO
# ============================================================

st.markdown(
    """
    <div class="studio-grid">
        <div>
            <div class="eyebrow">QUERY STUDIO / 01</div>
            <div class="studio-title">What do<br>you want<br><span class="pink">summarized?</span></div>
            <div class="studio-description">
                Paste a Reddit post. Each model generates a summary in turn.
                The trained reward model then scores all three outputs.
            </div>
            <div class="tip">
                <span class="tip-mark">i</span>
                <span>Tip: Include the full post and its relevant context.
                The application appends the TL;DR prompt automatically.</span>
            </div>
        </div>
        <div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

default_sample = (
    "I have been working at my current software company for two years "
    "as a junior developer. I recently received a job offer from another "
    "firm that pays 25% more salary, but requires full in-office attendance "
    "with a 45-minute commute each way. My current job is 100% remote, "
    "and I value my flexibility and work-life balance greatly. "
    "I have to make a final decision by Monday."
)

with st.container(border=True):
    st.markdown(
        """
        <div class="form-panel-top">
            <span>INPUT_01 / REDDIT POST</span>
            <span class="required">REQUIRED</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_post = st.text_area(
        "Your Reddit post",
        value=default_sample,
        height=170,
        placeholder="Paste the Reddit post here...",
    )

    st.caption(f"INPUT LENGTH: {len(user_post):,} CHARACTERS")

    generate_button = st.button(
        "GENERATE SUMMARIES  →",
        type="primary",
        use_container_width=True,
        disabled=not user_post.strip(),
    )


# ============================================================
# RUN SEQUENTIAL INFERENCE
# ============================================================

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if generate_button:
    prompt = user_post.strip() + "\nTL;DR:"
    progress = st.progress(0, text="Preparing inference...")

    try:
        # Stage 1: SFT
        sft_summary = generate_with_one_model(
            "sft", tokenizer, prompt, progress, 0, 25
        )

        # Stage 2: PPO
        ppo_summary = generate_with_one_model(
            "ppo", tokenizer, prompt, progress, 25, 50
        )

        # Stage 3: DPO
        dpo_summary = generate_with_one_model(
            "dpo", tokenizer, prompt, progress, 50, 75
        )

        # Stage 4: load reward model only after generation models are gone.
        progress.progress(78, text="Loading reward model...")
        reward_model = load_reward_model()

        try:
            progress.progress(88, text="Scoring generated summaries...")

            sft_score = compute_reward_score(
                reward_model, tokenizer, prompt, sft_summary
            )
            ppo_score = compute_reward_score(
                reward_model, tokenizer, prompt, ppo_summary
            )
            dpo_score = compute_reward_score(
                reward_model, tokenizer, prompt, dpo_summary
            )
        finally:
            del reward_model
            gc.collect()

        st.session_state.comparison_results = {
            "sft_summary": sft_summary,
            "ppo_summary": ppo_summary,
            "dpo_summary": dpo_summary,
            "sft_score": sft_score,
            "ppo_score": ppo_score,
            "dpo_score": dpo_score,
        }

        progress.progress(100, text="Comparison complete.")

    except Exception as error:
        gc.collect()
        st.error(
            "Inference failed. Check the checkpoint files and available "
            "memory, then inspect the technical details below."
        )
        with st.expander("Technical details"):
            st.exception(error)


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.comparison_results

st.markdown(
    """
    <div class="results-heading">
        <div>
            <div class="eyebrow">OUTPUT / MODEL EVALUATION</div>
            <div class="results-title">The results.</div>
        </div>
        <div class="results-tag">SFT / PPO / DPO</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if results is None:
    st.markdown(
        """
        <div class="empty-result">
            NO OUTPUTS YET<br><br>
            Submit a Reddit post above to generate three summaries
            and compare their reward-model scores.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        render_result_card(
            "SFT", "STAGE 01", "SUPERVISED FINE-TUNING",
            results["sft_summary"], results["sft_score"]
        )

    with col2:
        render_result_card(
            "PPO", "STAGE 03", "PROXIMAL POLICY OPTIMIZATION",
            results["ppo_summary"], results["ppo_score"],
            results["sft_score"]
        )

    with col3:
        render_result_card(
            "DPO", "STAGE 04", "DIRECT PREFERENCE OPTIMIZATION",
            results["dpo_summary"], results["dpo_score"],
            results["sft_score"]
        )

    st.markdown(
        """
        <div class="score-note">
            <strong>HOW TO READ THESE SCORES</strong><br>
            Scores are scalar outputs from the trained reward model.
            A higher score indicates that the reward model assigns a
            higher preference score to that summary. It is not a direct
            human rating or a guarantee of factual accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div>SMOLLM2-135M / RLHF & DPO SUMMARIZATION</div>
        <div>GENERATIVE AI — SEMESTER 07</div>
        <div>SFT · REWARD MODELING · PPO · DPO</div>
    </div>
    """,
    unsafe_allow_html=True,
)
