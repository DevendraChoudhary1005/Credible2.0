"""
app.py  -  Credible | Hybrid ML + Groq Fake News Detection
========================================================
How it works:
- ML Ensemble handles high-confidence predictions (fast, free)
- Groq API handles low-confidence / short text (smart, accurate)
"""

import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ensemble_model   import load_or_train, predict, preprocess
from groq_verifier import configure_groq, groq_fact_check

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credible - Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Confidence threshold — below this Groq takes over ──────────────────────
ML_CONFIDENCE_THRESHOLD = 0.80

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0d0f14; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #131720 50%, #0d0f14 100%);
    border: 1px solid #1e2535;
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 80% 60% at 70% 40%, rgba(99,102,241,0.12) 0%, transparent 70%),
                radial-gradient(ellipse 60% 40% at 20% 80%, rgba(236,72,153,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title { font-size: 3.2rem; font-weight: 700; color: #f1f5f9; line-height: 1.1; margin: 0 0 0.8rem; }
.hero-title span { color: #818cf8; }
.hero-sub { color: #64748b; font-size: 1.05rem; max-width: 580px; line-height: 1.6; margin: 0; }
.model-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.8rem; }
.pill {
    background: rgba(30,37,53,0.9);
    border: 1px solid #1e2535;
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    color: #94a3b8;
}
.pill .dot { color: #34d399; margin-right: 0.4rem; }
.pill .dot-groq { color: #fbbf24; margin-right: 0.4rem; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: #475569;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

  /* Verdict */
.verdict-real {
    background: linear-gradient(135deg, rgba(5,150,105,0.15) 0%, rgba(16,185,129,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px; padding: 2rem; text-align: center;
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(248,113,113,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px; padding: 2rem; text-align: center;
}
.verdict-unverifiable {
    background: linear-gradient(135deg, rgba(250,204,21,0.15) 0%, rgba(251,191,36,0.08) 100%);
    border: 1px solid rgba(250,204,21,0.4);
    border-radius: 16px; padding: 2rem; text-align: center;
}
.verdict-icon  { font-size: 3rem; margin-bottom: 0.4rem; }
.verdict-word  { font-size: 2.2rem; font-weight: 700; margin: 0.2rem 0; }
.verdict-real  .verdict-word { color: #34d399; }
.verdict-fake  .verdict-word { color: #f87171; }
.verdict-unverifiable .verdict-word { color: #fbbf24; }
.verdict-conf  { font-size: 1rem; color: #94a3b8; }

  /* Engine badge */
.engine-ml {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    margin-top: 0.8rem;
}
.engine-groq {
    display: inline-block;
    background: rgba(251,191,36,0.15);
    border: 1px solid rgba(251,191,36,0.4);
    color: #fbbf24;
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    margin-top: 0.8rem;
}

  /* Reasoning box */
.reasoning-box {
    background: #131720;
    border: 1px solid #1e2535;
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.reasoning-title { font-size: 0.75rem; color: #6366f1; font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.reasoning-text  { color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; }

  /* Red flags */
.red-flag {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #f87171;
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    margin: 0.2rem;
}

  /* Sources */
.source-chip {
    display: inline-block;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    color: #34d399;
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    margin: 0.2rem;
    font-family: 'Space Mono', monospace;
}

  /* Metric cards */
.metric-card {
    background: #131720;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #475569; font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; margin: 0.3rem 0; }

  /* Model rows */
.model-row {
    background: #131720;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.model-name  { font-weight: 600; color: #e2e8f0; font-size: 0.95rem; min-width: 185px; }
.badge-real  { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #34d399; border-radius: 100px; padding: 0.2rem 0.8rem; font-size: 0.8rem; font-weight: 600; }
.badge-fake  { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: #f87171; border-radius: 100px; padding: 0.2rem 0.8rem; font-size: 0.8rem; font-weight: 600; }
.conf-text   { color: #94a3b8; font-size: 0.85rem; min-width: 55px; text-align: right; }
.weight-badge {
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
    color: #818cf8; border-radius: 100px; padding: 0.15rem 0.6rem;
    font-size: 0.72rem; font-family: 'Space Mono', monospace; min-width: 60px; text-align: center;
}

  /* Streamlit overrides */
.stTextArea textarea {
    background: #0d0f14 !important; border: 1px solid #1e2535 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important; font-size: 0.95rem !important;
}
.stTextArea textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important; }
.stButton button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 0.7rem 2rem !important; width: 100%;
}
.stButton button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important; }
div[data-testid="stFileUploader"] { background: #131720 !important; border: 1px dashed #2d3748 !important; border-radius: 12px !important; }
.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #34d399) !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #64748b !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { color: #818cf8 !important; border-bottom-color: #6366f1 !important; }
section[data-testid="stSidebar"] { background: #0d0f14 !important; border-right: 1px solid #1e2535; }
  section[data-testid="stSidebar"] * { color: #94a3b8; }
div[data-testid="stExpander"] { background: #131720; border: 1px solid #1e2535; border-radius: 12px; }
.stTextInput input {
    background: #0d0f14 !important; border: 1px solid #1e2535 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important;  
            }
</style>
""", unsafe_allow_html=True)


# ── Model Info ────────────────────────────────────────────────────────────────
MODEL_INFO = {
    "Logistic Regression": {"icon": "📈", "desc": "Linear classifier on TF-IDF bigrams",           "color": "#818cf8"},
    "Random Forest":       {"icon": "🌲", "desc": "200 decision trees on TF-IDF features",          "color": "#34d399"},
    "Naive Bayes":         {"icon": "🧮", "desc": "Probabilistic word frequency model",             "color": "#fbbf24"},
    "Gradient Boosting":   {"icon": "🚀", "desc": "300 boosted shallow trees on TF-IDF",            "color": "#f472b6"},
    "Linear SVC":          {"icon": "✂️",  "desc": "Support Vector Machine with calibration",        "color": "#38bdf8"},
}


# ── Auto-load Groq API key from .env ─────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from the same folder automatically

if "groq_configured" not in st.session_state:
    _api_key = os.getenv("GROQ_API_KEY", "")
    if _api_key and _api_key != "your_groq_api_key_here":
        try:
            configure_groq(_api_key)
            st.session_state.groq_configured = True
        except Exception:
            st.session_state.groq_configured = False
    else:
        st.session_state.groq_configured = False


# ── Load ML Model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    with st.spinner("Loading ML ensemble model..."):
        return load_or_train()


# ── Hybrid Analysis ───────────────────────────────────────────────────────────
def hybrid_analyse(text: str, payload: dict, use_groq: bool) -> dict:
    """
    Run ML first. If confidence is low OR text is short OR Groq is forced,
    escalate to Groq API for fact checking.
    """
    # Always run ML first
    ml_result = predict(text, payload)
    word_count = len(text.split())

    # Decide whether to escalate to Groq
    escalate = (
        use_groq and st.session_state.groq_configured and (
            ml_result["confidence"] < ML_CONFIDENCE_THRESHOLD or  # low confidence
            word_count < 20                                         # short text
        )
    )

    if escalate:
        with st.spinner("🤖 ML confidence low — escalating to Groq (Llama 3.3 70B)..."):
            groq_result = groq_fact_check(text)

        return {
            "engine":        "Groq",
            "label":         groq_result["label"],
            "confidence":    groq_result["confidence"],
            "reasoning":     groq_result["reasoning"],
            "red_flags":     groq_result["red_flags"],
            "sources":       groq_result["sources"],
            "ml_result":     ml_result,
            "escalated":     True,
            "word_count":    word_count,
        }
    else:
        return {
            "engine":        "ML",
            "label":         ml_result["label"],
            "confidence":    ml_result["confidence"],
            "fake_prob":     ml_result["fake_prob"],
            "real_prob":     ml_result["real_prob"],
            "individual":    ml_result["individual"],
            "votes":         ml_result["votes"],
            "ml_result":     ml_result,
            "escalated":     False,
            "word_count":    word_count,
        }

# ── Charts ────────────────────────────────────────────────────────────────────
def gauge_chart(fake_prob: float, real_prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(real_prob * 100, 1),
        number={"suffix": "% Real", "font": {"size": 22, "color": "#e2e8f0", "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#1e2535", "tickfont": {"color": "#475569", "size": 11}},
            "bar":  {"color": "#34d399" if real_prob >= 0.5 else "#f87171", "thickness": 0.3},
            "bgcolor": "#131720", "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "rgba(239,68,68,0.12)"},
                {"range": [40, 60],  "color": "rgba(250,204,21,0.10)"},
                {"range": [60, 100], "color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {"line": {"color": "#6366f1", "width": 3}, "thickness": 0.8, "value": 50},
        },
        title={"text": "Confidence", "font": {"size": 13, "color": "#64748b", "family": "Space Mono"}},
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def bar_chart(individual: dict):
    names  = list(individual.keys())
    vals   = [round(res["proba"][1] * 100, 1) for res in individual.values()]
    colors = [MODEL_INFO[n]["color"] for n in names]

    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12, family="Space Mono"),
    ))
    fig.add_vline(x=50, line_dash="dot", line_color="#6366f1", line_width=1.5,
                annotation_text="Decision boundary", annotation_font_color="#6366f1", annotation_font_size=11)
    fig.update_layout(
        height=300, margin=dict(l=10, r=70, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False,
                tickfont=dict(color="#475569"), title="P(Real) %", title_font=dict(color="#475569", size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(color="#cbd5e1", family="Space Grotesk", size=12)),
        bargap=0.35,
    )
    return fig


def vote_donut(votes: dict):
    total = votes["Fake"] + votes["Real"]
    fig = go.Figure(go.Pie(
        labels=["Fake", "Real"], values=[votes["Fake"], votes["Real"]],
        hole=0.68, marker=dict(colors=["#f87171", "#34d399"], line=dict(color="#0d0f14", width=3)),
        textinfo="label+percent", textfont=dict(color="#e2e8f0", size=13, family="Space Grotesk"), showlegend=False,
    ))
    fig.add_annotation(text=f"{votes['Real']}/{total}<br>Real", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="#e2e8f0", family="Space Grotesk"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Credible")
    st.markdown("---")

    # ── Groq Status (loaded from .env) ───────────────────────────
    st.markdown("### 🤖 Groq AI Status")
    if st.session_state.groq_configured:
        st.markdown(
            "<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);"
            "border-radius:8px;padding:0.5rem 0.8rem;font-size:0.8rem;color:#34d399;margin-top:0.5rem'>"
            "✅ Groq Connected (via .env)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);"
            "border-radius:8px;padding:0.5rem 0.8rem;font-size:0.8rem;color:#f87171;margin-top:0.5rem'>"
            "❌ Groq Not Connected</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<small style='color:#475569'>Add GROQ_API_KEY to your .env file</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Threshold slider ──────────────────────────────────────────
    st.markdown("### ⚙️ Hybrid Settings")
    threshold = st.slider(
        "Escalate to Groq below:",
        min_value=0.50,
        max_value=0.95,
        value=ML_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="If ML confidence is below this value, Groq fact-checks the text",
    )
    ML_CONFIDENCE_THRESHOLD = threshold
    st.markdown(
        f"<small style='color:#475569'>ML handles >{threshold:.0%} confidence<br>"
        f"Groq handles <{threshold:.0%} confidence</small>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── ML Model info ─────────────────────────────────────────────
    st.markdown("### 🧠 ML Models")
    for name, info in MODEL_INFO.items():
        st.markdown(
            f"**{info['icon']} {name}**  \n"
            f"<small style='color:#475569'>{info['desc']}</small>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown("---")

    payload = get_model()

    with st.expander("📊 ML Performance"):
        acc = payload.get("accuracy", 0)
        st.metric("Ensemble Accuracy", f"{acc:.2%}")
        model_accs    = payload.get("model_accuracies", {})
        model_weights = payload.get("weights", {})
        if model_accs:
            for name, a in model_accs.items():
                w    = model_weights.get(name, 0)
                icon = MODEL_INFO[name]["icon"]
                st.markdown(
                    f"{icon} **{name}**  \n"
                    f"<small style='color:#475569'>Acc: `{a:.2%}` Weight: `{w:.2f}`</small>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        "<small style='color:#334155'>Credible v2.0 · ML + Groq Hybrid</small>",
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
<div class="hero-badge">⚡ HYBRID · 5 ML MODELS + GROQ AI (LLAMA 3.3 70B)</div>
<h1 class="hero-title">Cred<span>ible</span></h1>
<p class="hero-sub">
    Dual-engine fake news detection. Our ML ensemble handles high-confidence
    predictions instantly. When confidence is low, <strong>Groq AI</strong>
    searches the web in real-time to fact-check against live sources.
</p>
<div class="model-pills">
    <span class="pill"><span class="dot">●</span>Logistic Regression</span>
    <span class="pill"><span class="dot">●</span>Random Forest</span>
    <span class="pill"><span class="dot">●</span>Naive Bayes</span>
    <span class="pill"><span class="dot">●</span>Gradient Boosting</span>
    <span class="pill"><span class="dot">●</span>Linear SVC</span>
    <span class="pill"><span class="dot-groq">●</span>Groq (Llama 3.3 70B)</span>
</div>
</div>
""", unsafe_allow_html=True)


# ── Input Tabs ────────────────────────────────────────────────────────────────
tab_text, tab_file, tab_csv = st.tabs([
    "✍️  Text / Headline",
    "📄  Upload File (.txt)",
    "📊  Batch CSV",
])

with tab_text:
    st.markdown('<div class="section-label">Paste your article, headline or claim below</div>',
                unsafe_allow_html=True)
    text_input = st.text_area("", height=200,
                            placeholder="e.g.  Modi is India's Prime Minister  OR  paste a full article...",
                            label_visibility="collapsed")
    use_groq = st.checkbox(
        "Enable Groq escalation (uses API for low-confidence predictions)",
        value=st.session_state.groq_configured,
        disabled=not st.session_state.groq_configured,
    )
    btn_text = st.button("🔍  Analyse Text", key="btn_text")

with tab_file:
    st.markdown('<div class="section-label">Upload a plain-text file (.txt)</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["txt"], label_visibility="collapsed")
    btn_file = st.button("🔍  Analyse File", key="btn_file")

with tab_csv:
    st.markdown('<div class="section-label">Upload a CSV with a <code>text</code> column</div>',
                unsafe_allow_html=True)
    uploaded_csv = st.file_uploader("", type=["csv"], label_visibility="collapsed", key="csv_up")
    btn_csv = st.button("🚀  Run Batch Analysis", key="btn_csv")


# ── Active Input ──────────────────────────────────────────────────────────────
active_text = None
batch_mode  = False

if btn_text and text_input.strip():
    active_text = text_input.strip()
elif btn_file and uploaded_file:
    active_text = uploaded_file.read().decode("utf-8", errors="ignore").strip()
elif btn_csv and uploaded_csv:
    batch_mode = True


# ── Single Prediction ─────────────────────────────────────────────────────────
if active_text:
    with st.spinner("🧠  Running ML ensemble..."):
        time.sleep(0.3)
        result = hybrid_analyse(active_text, payload, use_groq)

    st.markdown("---")
    st.markdown("## 🏁 Verdict")

    # ── Engine badge ──────────────────────────────────────────────
    if result["engine"] == "Groq":
        st.markdown(
            '<div style="margin-bottom:1rem">'
            '<span class="engine-groq">🤖 Decided by: Groq AI (Llama 3.3 70B)</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="margin-bottom:1rem">'
            '<span class="engine-ml">⚡ Decided by: ML Ensemble (fast path)</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    col_verdict, col_gauge = st.columns([1, 1])

    with col_verdict:
        label = result["label"]
        conf  = result["confidence"]

        if label == "Real":
            cls_div = "verdict-real"
            icon    = "✅"
            word    = "GENUINE"
        elif label == "Fake":
            cls_div = "verdict-fake"
            icon    = "🚫"
            word    = "FAKE"
        else:
            cls_div = "verdict-unverifiable"
            icon    = "⚠️"
            word    = "UNVERIFIABLE"

        st.markdown(f"""
        <div class="{cls_div}">
        <div class="verdict-icon">{icon}</div>
        <div class="verdict-word">{word}</div>
        <div class="verdict-conf">Confidence: <strong>{conf:.1%}</strong></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics row
        if result["engine"] == "ML":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Fake Prob</div>
                <div class="metric-value" style="color:#f87171">{result['fake_prob']:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Real Prob</div>
                <div class="metric-value" style="color:#34d399">{result['real_prob']:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                votes = result["votes"]
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Votes</div>
                <div class="metric-value">{votes['Real']}R/{votes['Fake']}F</div>
                </div>""", unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Words</div>
                <div class="metric-value">{result['word_count']}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                ml_conf = result["ml_result"]["confidence"]
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">ML Confidence</div>
                <div class="metric-value" style="color:#f87171">{ml_conf:.1%}</div>
                </div>""", unsafe_allow_html=True)

    with col_gauge:
        if result["engine"] == "ML":
            st.plotly_chart(
                gauge_chart(result["fake_prob"], result["real_prob"]),
                use_container_width=True,
            )
        else:
            fake_p = 1 - result["confidence"] if result["label"] == "Real" else result["confidence"]
            real_p = result["confidence"] if result["label"] == "Real" else 1 - result["confidence"]
            st.plotly_chart(gauge_chart(fake_p, real_p), use_container_width=True)

    # ── Groq reasoning ────────────────────────────────────────────
    if result["engine"] == "Groq":
        st.markdown("---")
        st.markdown("## 🤖 Groq Analysis")

        if result.get("reasoning"):
            st.markdown(f"""
            <div class="reasoning-box">
            <div class="reasoning-title">Reasoning</div>
            <div class="reasoning-text">{result['reasoning']}</div>
            </div>""", unsafe_allow_html=True)

        if result.get("red_flags"):
            st.markdown("**Red Flags Detected:**")
            flags_html = "".join([f'<span class="red-flag">⚠️ {f}</span>' for f in result["red_flags"]])
            st.markdown(flags_html, unsafe_allow_html=True)

        if result.get("sources"):
            st.markdown("<br>**Sources Checked:**", unsafe_allow_html=True)
            sources_html = "".join([f'<span class="source-chip">🔗 {s}</span>' for s in result["sources"]])
            st.markdown(sources_html, unsafe_allow_html=True)

        # Also show what ML said
        st.markdown("---")
        st.markdown("### 📊 ML Ensemble also said...")
        ml = result["ml_result"]
        ml_word = "GENUINE" if ml["label"] == "Real" else "FAKE"
        st.markdown(
            f"ML predicted **{ml_word}** with **{ml['confidence']:.1%}** confidence "
            f"(below {ML_CONFIDENCE_THRESHOLD:.0%} threshold → escalated to Groq)",
        )

    # ── ML breakdown ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 ML Model Breakdown")

    ml_data = result["ml_result"]
    col_bars, col_donut = st.columns([3, 2])

    with col_bars:
        st.markdown("**Individual Model Confidences (P(Real) %)**")
        st.plotly_chart(bar_chart(ml_data["individual"]), use_container_width=True)

    with col_donut:
        st.markdown("**Vote Distribution**")
        st.plotly_chart(vote_donut(ml_data["votes"]), use_container_width=True)

    st.markdown("### 🔬 Per-Model Details")
    model_weights = payload.get("weights", {})
    for name, res in ml_data["individual"].items():
        info      = MODEL_INFO[name]
        badge_cls = "badge-real" if res["label"] == "Real" else "badge-fake"
        badge_txt = "✅ Real"   if res["label"] == "Real" else "🚫 Fake"
        bar_w     = int(res["confidence"] * 100)
        bar_col   = "#34d399" if res["label"] == "Real" else "#f87171"
        weight    = model_weights.get(name, 0)

        st.markdown(f"""
        <div class="model-row">
        <span class="model-name">{info['icon']} {name}</span>
        <span class="{badge_cls}">{badge_txt}</span>
        <div style="flex:1; background:#1e2535; border-radius:100px; height:8px; margin:0 0.5rem;">
            <div style="width:{bar_w}%; background:{bar_col}; height:8px; border-radius:100px;"></div>
        </div>
        <span class="conf-text">{res['confidence']:.1%}</span>
        <span class="weight-badge">w={weight:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔤 Preprocessed Token View"):
        st.code(preprocess(active_text), language=None)


# ── Batch Prediction ──────────────────────────────────────────────────────────
elif batch_mode and uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    if "text" not in df.columns:
        st.error("CSV must contain a column named **`text`**.")
    else:
        st.markdown("---")
        st.markdown(f"### 🚀 Batch Analysis — {len(df)} rows")
        progress = st.progress(0)
        results  = []

        for i, row in enumerate(df["text"].astype(str)):
            r = predict(row, payload)
            results.append({
                "text":       row[:120] + ("..." if len(row) > 120 else ""),
                "verdict":    r["label"],
                "confidence": f"{r['confidence']:.1%}",
                "fake_prob":  f"{r['fake_prob']:.1%}",
                "real_prob":  f"{r['real_prob']:.1%}",
                "votes":      f"{r['votes']['Real']}R / {r['votes']['Fake']}F",
                "engine":     "ML",
            })
            progress.progress((i + 1) / len(df))

        result_df = pd.DataFrame(results)
        fake_cnt  = sum(1 for r in results if r["verdict"] == "Fake")
        real_cnt  = len(results) - fake_cnt

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Total</div>
            <div class="metric-value">{len(results)}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Genuine</div>
            <div class="metric-value" style="color:#34d399">{real_cnt}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Fake</div>
            <div class="metric-value" style="color:#f87171">{fake_cnt}</div></div>""", unsafe_allow_html=True)

        fig_dist = px.pie(names=["Genuine", "Fake"], values=[real_cnt, fake_cnt],
                        color_discrete_sequence=["#34d399", "#f87171"], hole=0.5)
        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            legend_font_color="#94a3b8", height=260,
                            margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig_dist, use_container_width=True)

        def colour_verdict(val):
            return "color: #34d399" if val == "Real" else "color: #f87171"

        styled = result_df.style.map(colour_verdict, subset=["verdict"])
        st.dataframe(styled, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv_bytes, "credible_results.csv", "text/csv")


# ── Empty state ───────────────────────────────────────────────────────────────
else:
    if not (btn_text or btn_file or btn_csv):
        groq_status = "✅ Connected" if st.session_state.groq_configured else "❌ Not connected (enter key in sidebar)"
        st.markdown(f"""
        <div style="text-align:center; padding:3rem 1rem; color:#334155;">
        <div style="font-size:3rem; margin-bottom:1rem;">🛡️</div>
        <p style="font-size:1.1rem;">Enter text, upload a file, or upload a CSV and hit <strong>Analyse</strong>.</p>
        <p style="font-size:0.85rem; color:#475569; margin-top:0.5rem;">Groq: {groq_status}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No input detected — please enter some text or upload a file before clicking Analyse.")
