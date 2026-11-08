
import os
import html

import streamlit as st
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmolLM2 | Summarization Arena",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
HF_REPO_ID = "razaasherazi/smollm2-135m-rlhf-checkpoints"

MAX_INPUT_LENGTH = 448
MAX_GENERATION_TOKENS = 48
MAX_REWARD_LENGTH = 512


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --background: #0b1120;
        --surface: #111b2e;
        --surface-light: #162238;
        --border: #26354d;
        --text: #e8eef8;
        --muted: #94a3b8;
        --accent: #7c9cff;
        --accent-soft: rgba(124, 156, 255, 0.12);
        --success: #50c9a5;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1450px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.5px;
    }

    p, label, span {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header */

    .top-label {
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .main-heading {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.1rem);
        font-weight: 800;
        letter-spacing: -1.8px;
        line-height: 1.15;
        color: #f1f5ff;
        margin-bottom: 14px;
    }

    .main-description {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        max-width: 850px;
        margin-bottom: 24px;
    }

    .header-divider {
        border: none;
        height: 1px;
        background: var(--border);
        margin: 30px 0;
    }

    /* Section headings */

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #e8eef8;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 0.9rem;
        color: var(--muted);
        margin-bottom: 20px;
    }

    /* Input */

    [data-testid="stTextArea"] textarea {
        background: #101a2c !important;
        color: #e8eef8 !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        padding: 16px !important;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    [data-testid="stTextArea"] label {
        color: #dce6f5 !important;
        font-weight: 600 !important;
    }

    /* Buttons */

    .stButton > button {
        background: #7898ff !important;
        color: #081121 !important;
        border: none !important;
        border-radius: 9px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        min-height: 48px !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #94adff !important;
        transform: translateY(-1px);
        box-shadow: 0 5px 18px rgba(100, 130, 255, 0.2);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px var(--accent) !important;
    }

    /* Model cards */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
    }

    .model-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 12px;
    }

    .model-name {
        font-family: 'Manrope', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #f1f5ff;
    }

    .model-stage {
        font-size: 0.7rem;
        color: var(--accent);
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        background: var(--accent-soft);
        border: 1px solid rgba(124, 156, 255, 0.22);
        padding: 5px 9px;
        border-radius: 20px;
        white-space: nowrap;
    }

    .model-subtitle {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 18px;
    }

    .summary-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--muted);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .summary-text {
        font-size: 0.94rem;
        line-height: 1.75;
        color: #dce6f5;
        min-height: 125px;
        overflow-wrap: anywhere;
    }

    .score-divider {
        height: 1px;
        background: var(--border);
        margin: 22px 0 16px 0;
    }

    .score-label {
        font-size: 0.76rem;
        color: var(--muted);
        margin-bottom: 5px;
    }

    .score-value {
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #f1f5ff;
        letter-spacing: -0.5px;
    }

    .score-note {
        font-size: 0.75rem;
        color: var(--muted);
        margin-top: 5px;
    }

    .positive-delta {
        color: var(--success);
        font-size: 0.8rem;
        font-weight: 700;
    }

    .negative-delta {
        color: #f08b8b;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* Empty state */

    .empty-state {
        background: var(--surface);
        border: 1px dashed #34445e;
        border-radius: 13px;
        padding: 38px 20px;
        text-align: center;
        margin-top: 12px;
    }

    .empty-title {
        font-family: 'Manrope', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #e8eef8;
        margin-bottom: 8px;
    }

    .empty-description {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background: var(--surface-light);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 13px 15px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    /* Info and alerts */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Footer */

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 48px;
        padding-top: 20px;
        color: #74839b;
        font-size: 0.78rem;
        line-height: 1.7;
        text-align: center;
    }

    /* Hide Streamlit branding */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REWARD MODEL
# ============================================================

class RewardModel(nn.Module):
    """Reward model using an SFT backbone and a scalar value head."""

    def __init__(self, base_causal_lm):
        super().__init__()

        self.transformer = base_causal_lm.model

        hidden_size = base_causal_lm.config.hidden_size

        self.v_head = nn.Linear(
            hidden_size,
            1,
            bias=False,
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_indices = (
            attention_mask.sum(dim=1) - 1
        ).clamp(min=0)

        batch_idx = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )

        last_hidden = outputs.last_hidden_state[
            batch_idx,
            last_indices,
        ]

        return self.v_head(last_hidden).squeeze(-1)


# ============================================================
# MODEL CHECKPOINT MANAGEMENT
# ============================================================

def resolve_model_path(stage_subfolder):
    """Resolve a checkpoint locally or download it from Hugging Face."""

    local_path = os.path.join(
        "checkpoints",
        stage_subfolder,
    )

    if os.path.exists(local_path) and os.listdir(local_path):
        return local_path

    downloaded_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=f"{stage_subfolder}/*",
    )

    resolved_path = os.path.join(
        downloaded_dir,
        stage_subfolder,
    )

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {resolved_path}"
        )

    return resolved_path


# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_all_models():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # SFT MODEL
    # --------------------------------------------------------

    sft_path = resolve_model_path("sft")

    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=dtype,
    ).to(device).eval()

    # --------------------------------------------------------
    # PPO MODEL
    # --------------------------------------------------------

    ppo_path = resolve_model_path("ppo")

    ppo_model = AutoModelForCausalLM.from_pretrained(
        ppo_path,
        torch_dtype=dtype,
    ).to(device).eval()

    # --------------------------------------------------------
    # DPO MODEL
    # --------------------------------------------------------

    dpo_path = resolve_model_path("dpo")

    dpo_model = AutoModelForCausalLM.from_pretrained(
        dpo_path,
        torch_dtype=dtype,
    ).to(device).eval()

    # --------------------------------------------------------
    # REWARD MODEL
    # Load an independent SFT backbone.
    # This prevents reward checkpoint loading from
    # modifying the actual SFT generation model.
    # --------------------------------------------------------

    reward_backbone = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=dtype,
    )

    reward_model = RewardModel(
        reward_backbone
    ).to(device).eval()

    reward_path = resolve_model_path("reward")

    weights_file = os.path.join(
        reward_path,
        "reward_model.pt",
    )

    if not os.path.exists(weights_file):
        raise FileNotFoundError(
            f"Reward model weights not found: {weights_file}"
        )

    reward_state = torch.load(
        weights_file,
        map_location=device,
        weights_only=True,
    )

    reward_model.load_state_dict(
        reward_state,
        strict=True,
    )

    return (
        tokenizer,
        sft_model,
        ppo_model,
        dpo_model,
        reward_model,
        device,
    )


# ============================================================
# TEXT GENERATION
# ============================================================

def generate_summary(
    model,
    tokenizer,
    prompt_text,
    device,
    max_new_tokens=MAX_GENERATION_TOKENS,
):

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0, prompt_len:]

    raw_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False,
    )

    if "<|endoftext|>" in raw_text:
        raw_text = raw_text.split("<|endoftext|>")[0]

    return raw_text.strip()


# ============================================================
# REWARD SCORING
# ============================================================

def compute_reward_score(
    reward_model,
    tokenizer,
    prompt_text,
    summary_text,
    device,
):

    full_text = (
        prompt_text.strip()
        + " "
        + summary_text.strip()
        + "<|endoftext|>"
    )

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_REWARD_LENGTH,
    ).to(device)

    with torch.no_grad():

        score = reward_model(
            inputs["input_ids"],
            inputs["attention_mask"],
        ).item()

    return round(score, 3)


# ============================================================
# MODEL RESULT CARD
# ============================================================

def render_model_card(
    model_name,
    stage,
    description,
    summary,
    score,
    baseline_score=None,
):

    with st.container(border=True):

        st.markdown(
            f"""
            <div class="model-card-header">
                <div class="model-name">{model_name}</div>
                <div class="model-stage">{stage}</div>
            </div>

            <div class="model-subtitle">
                {description}
            </div>

            <div class="summary-label">
                Generated summary
            </div>

            <div class="summary-text">
                {html.escape(summary)}
            </div>

            <div class="score-divider"></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="score-label">Reward model score</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="score-value">{score:+.3f}</div>',
            unsafe_allow_html=True,
        )

        if baseline_score is not None:

            difference = round(
                score - baseline_score,
                3,
            )

            if difference >= 0:

                st.markdown(
                    f"""
                    <div class="positive-delta">
                        +{difference:.3f} compared with SFT
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="negative-delta">
                        {difference:.3f} compared with SFT
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.markdown(
                """
                <div class="score-note">
                    Reference baseline
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="top-label">
        Generative AI Research Project
    </div>

    <div class="main-heading">
        SmolLM2-135M<br>
        Summarization Arena
    </div>

    <div class="main-description">
        An interactive comparison of supervised fine-tuning,
        reinforcement learning from human feedback, and direct
        preference optimization for Reddit summarization.
        Evaluate generated summaries using a trained reward model.
    </div>

    <hr class="header-divider">
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    with st.spinner("Initializing models and loading checkpoints..."):

        (
            tokenizer,
            sft_model,
            ppo_model,
            dpo_model,
            reward_model,
            device,
        ) = load_all_models()

except Exception as error:

    st.error(
        "The models could not be loaded. Please verify your "
        "Hugging Face repository, checkpoint structure, and dependencies."
    )

    with st.expander("Technical error details"):
        st.exception(error)

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.metric("Base architecture", "SmolLM2-135M")

with status_col2:
    st.metric("Optimization methods", "3")

with status_col3:
    st.metric("Execution device", device.upper())


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        Input Reddit Post
    </div>

    <div class="section-description">
        Provide a Reddit post to generate and compare summaries
        across all three fine-tuning approaches.
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

    user_post = st.text_area(
        "Reddit post content",
        value=default_sample,
        height=190,
        placeholder="Paste the Reddit post you want to summarize...",
        help="The application automatically appends the TL;DR prompt.",
        label_visibility="visible",
    )

    input_col1, input_col2 = st.columns([3, 1])

    with input_col1:

        st.caption(
            "Prompt format: Reddit post content + TL;DR:"
        )

    with input_col2:

        st.caption(
            f"Input length: {len(user_post):,} characters"
        )

    generate_button = st.button(
        "Generate and Compare Summaries",
        type="primary",
        use_container_width=True,
        disabled=not user_post.strip(),
    )


# ============================================================
# GENERATION AND EVALUATION
# ============================================================

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None


if generate_button:

    formatted_prompt = user_post.strip() + "\nTL;DR:"

    try:

        with st.spinner(
            "Generating summaries and calculating reward scores..."
        ):

            # Generate summaries
            sft_summary = generate_summary(
                sft_model,
                tokenizer,
                formatted_prompt,
                device,
            )

            ppo_summary = generate_summary(
                ppo_model,
                tokenizer,
                formatted_prompt,
                device,
            )

            dpo_summary = generate_summary(
                dpo_model,
                tokenizer,
                formatted_prompt,
                device,
            )

            # Calculate reward scores
            sft_score = compute_reward_score(
                reward_model,
                tokenizer,
                formatted_prompt,
                sft_summary,
                device,
            )

            ppo_score = compute_reward_score(
                reward_model,
                tokenizer,
                formatted_prompt,
                ppo_summary,
                device,
            )

            dpo_score = compute_reward_score(
                reward_model,
                tokenizer,
                formatted_prompt,
                dpo_summary,
                device,
            )

        st.session_state.comparison_results = {
            "sft_summary": sft_summary,
            "ppo_summary": ppo_summary,
            "dpo_summary": dpo_summary,
            "sft_score": sft_score,
            "ppo_score": ppo_score,
            "dpo_score": dpo_score,
        }

    except Exception as error:

        st.error("An error occurred during generation or evaluation.")

        with st.expander("Technical error details"):
            st.exception(error)


# ============================================================
# RESULTS SECTION
# ============================================================

results = st.session_state.comparison_results


if results is not None:

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-heading">
            Model Comparison
        </div>

        <div class="section-description">
            Compare the generated summaries and their reward-model
            scores. PPO and DPO score differences are calculated
            relative to the SFT baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:

        render_model_card(
            model_name="SFT",
            stage="Stage 01",
            description="Supervised Fine-Tuning",
            summary=results["sft_summary"],
            score=results["sft_score"],
        )

    with col2:

        render_model_card(
            model_name="PPO",
            stage="Stage 03",
            description="Proximal Policy Optimization",
            summary=results["ppo_summary"],
            score=results["ppo_score"],
            baseline_score=results["sft_score"],
        )

    with col3:

        render_model_card(
            model_name="DPO",
            stage="Stage 04",
            description="Direct Preference Optimization",
            summary=results["dpo_summary"],
            score=results["dpo_score"],
            baseline_score=results["sft_score"],
        )

    # --------------------------------------------------------
    # SCORE OVERVIEW
    # --------------------------------------------------------

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-heading">
            Reward Score Overview
        </div>

        <div class="section-description">
            Scalar scores assigned by the trained reward model
            to each generated summary.
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:
        st.metric(
            "SFT reward",
            f'{results["sft_score"]:+.3f}',
        )

    with score_col2:
        st.metric(
            "PPO reward",
            f'{results["ppo_score"]:+.3f}',
            delta=round(
                results["ppo_score"] - results["sft_score"],
                3,
            ),
        )

    with score_col3:
        st.metric(
            "DPO reward",
            f'{results["dpo_score"]:+.3f}',
            delta=round(
                results["dpo_score"] - results["sft_score"],
                3,
            ),
        )

    st.caption(
        "Interpretation: Reward scores are produced by a learned reward "
        "model. They are not direct human ratings or guaranteed measures "
        "of summary quality."
    )

else:

    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">
                Ready to evaluate
            </div>

            <div class="empty-description">
                Enter a Reddit post above and select
                <strong>Generate and Compare Summaries</strong>.
                <br>
                The SFT, PPO, and DPO models will generate their
                summaries and receive reward-model scores.
            </div>
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
        <strong>SmolLM2-135M: RLHF & DPO for Reddit Summarization</strong>
        <br>
        Supervised Fine-Tuning · Reward Modeling · PPO · DPO
        <br><br>
        Generative AI — Semester 7
        <br>
        Research and implementation project
    </div>
    """,
    unsafe_allow_html=True,
)
