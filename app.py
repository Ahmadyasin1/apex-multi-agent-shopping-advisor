"""
Intelligent Shopping Advisor — Elite Edition
Ultra-luxury Streamlit UI with dark gold premium aesthetic.
"""
import streamlit as st
from graph import create_workflow
from state import AgentState
from agents.learning import get_user_context

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="APEX — Intelligent Shopping Advisor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "APEX Shopping Advisor v2.0 — Elite AI Edition by Ahmad Yasin"}
)

# ─── Luxury CSS Injection ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Global Reset & Theme ── */
html, body, [class*="css"] {
    background-color: #080810 !important;
    color: #E8E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Remove Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Hero Header ── */
.apex-hero {
    background: linear-gradient(135deg, #0D0D1A 0%, #1A0A2E 40%, #0D1A0D 100%);
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.apex-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #D4AF37, #FFD700, #D4AF37, transparent);
}
.apex-hero::after {
    content: '◆';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    color: rgba(212,175,55,0.04);
    font-family: sans-serif;
}
.apex-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #D4AF37, #FFD700, #B8860B);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.15 !important;
}
.apex-subtitle {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    color: rgba(200,200,220,0.65) !important;
    font-weight: 300 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.apex-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(212,175,55,0.15), rgba(212,175,55,0.08));
    border: 1px solid rgba(212,175,55,0.35);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #D4AF37;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C0C18 0%, #080810 100%) !important;
    border-right: 1px solid rgba(212,175,55,0.15) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}

/* ── Sidebar Section Header ── */
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: #D4AF37;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.8rem 0 0.4rem 0;
    border-bottom: 1px solid rgba(212,175,55,0.15);
    margin-bottom: 0.8rem;
}

/* ── Agent Status Pill ── */
.agent-pill {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.55rem 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    font-weight: 500;
    transition: all 0.2s ease;
}
.agent-pill.idle {
    background: rgba(255,255,255,0.03);
    color: rgba(200,200,220,0.4);
    border: 1px solid rgba(255,255,255,0.06);
}
.agent-pill.active {
    background: linear-gradient(135deg, rgba(212,175,55,0.15), rgba(212,175,55,0.05));
    color: #D4AF37;
    border: 1px solid rgba(212,175,55,0.35);
    box-shadow: 0 0 12px rgba(212,175,55,0.1);
}
.agent-pill.done {
    background: rgba(34,197,94,0.08);
    color: rgba(134,239,172,0.85);
    border: 1px solid rgba(34,197,94,0.2);
}
.agent-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-idle   { background: rgba(255,255,255,0.15); }
.dot-active { background: #D4AF37; box-shadow: 0 0 6px #D4AF37; animation: pulse 1s infinite; }
.dot-done   { background: #22C55E; }
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(1.3); }
}

/* ── Chat Messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    margin-bottom: 2rem;
}
.msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, rgba(212,175,55,0.18), rgba(212,175,55,0.08));
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.4rem;
    max-width: 72%;
    color: #F0E8D0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-ai {
    align-self: flex-start;
    background: linear-gradient(135deg, rgba(26,26,46,0.95), rgba(18,18,36,0.95));
    border: 1px solid rgba(212,175,55,0.15);
    border-left: 3px solid #D4AF37;
    border-radius: 4px 18px 18px 18px;
    padding: 1.4rem 1.8rem;
    max-width: 90%;
    color: #DDD8F0;
    font-size: 0.92rem;
    line-height: 1.75;
}
.msg-label-user {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: rgba(212,175,55,0.7);
    text-transform: uppercase;
    margin-bottom: 0.35rem;
    text-align: right;
}
.msg-label-ai {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: rgba(180,160,220,0.6);
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

/* ── Product Score Card ── */
.product-card {
    background: linear-gradient(135deg, rgba(20,20,38,0.98), rgba(14,14,28,0.98));
    border: 1px solid rgba(212,175,55,0.15);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.product-card:hover {
    border-color: rgba(212,175,55,0.45);
    box-shadow: 0 8px 32px rgba(212,175,55,0.08);
    transform: translateY(-1px);
}
.product-card.pareto::before {
    content: '✦ OPTIMAL';
    position: absolute;
    top: 0.9rem; right: 1rem;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #D4AF37;
    background: rgba(212,175,55,0.12);
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 50px;
    padding: 0.2rem 0.6rem;
}
.product-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #F0E8D0;
    margin-bottom: 0.25rem;
}
.product-meta {
    font-size: 0.78rem;
    color: rgba(200,200,220,0.5);
    margin-bottom: 0.9rem;
}
.product-price {
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #D4AF37;
}
.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.3rem;
}
.score-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(200,200,220,0.5);
    width: 80px;
    flex-shrink: 0;
}
.score-bar-bg {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #D4AF37, #FFD700);
}
.score-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #D4AF37;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Metric Cards Row ── */
.metric-card {
    background: linear-gradient(135deg, rgba(20,20,38,0.9), rgba(14,14,28,0.9));
    border: 1px solid rgba(212,175,55,0.15);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #D4AF37;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(200,200,220,0.45);
    margin-top: 0.3rem;
}

/* ── Chat input overrides ── */
[data-testid="stChatInput"] textarea {
    background: rgba(20,20,40,0.9) !important;
    border: 1px solid rgba(212,175,55,0.3) !important;
    border-radius: 14px !important;
    color: #E8E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(212,175,55,0.6) !important;
    box-shadow: 0 0 20px rgba(212,175,55,0.08) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(16,16,32,0.8) !important;
    border: 1px solid rgba(212,175,55,0.12) !important;
    border-radius: 14px !important;
}

/* ── Streamlit widgets ── */
.stSelectbox label, .stSlider label, .stMultiselect label {
    color: rgba(200,200,220,0.7) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #D4AF37, #FFD700) !important;
}

/* ── Divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,175,55,0.4), transparent);
    margin: 1.5rem 0;
}

/* ── Info box ── */
.info-box {
    background: linear-gradient(135deg, rgba(212,175,55,0.06), rgba(212,175,55,0.02));
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-size: 0.83rem;
    color: rgba(212,175,55,0.8);
    line-height: 1.6;
}

/* ── Query chip ── */
.query-chip {
    display: inline-block;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: rgba(212,175,55,0.7);
    margin: 0.2rem;
    cursor: pointer;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212,175,55,0.45); }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_states" not in st.session_state:
    st.session_state.agent_states = {}
if "last_final_state" not in st.session_state:
    st.session_state.last_final_state = None
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">💎</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:600;
                    background: linear-gradient(135deg,#D4AF37,#FFD700);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            APEX Advisor
        </div>
        <div style="font-size:0.65rem; letter-spacing:0.2em; color:rgba(212,175,55,0.5);
                    text-transform:uppercase; margin-top:0.2rem;">
            Elite AI Edition
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent pipeline status
    st.markdown('<div class="sidebar-section">⚡ AI Agent Pipeline</div>', unsafe_allow_html=True)

    agent_pipeline = [
        ("🔍", "Supervisor",       "Complexity Routing"),
        ("🧠", "Preference",       "Intent Extraction"),
        ("📡", "Retrieval",        "Semantic Search"),
        ("✨", "Enrichment",       "Data Enhancement"),
        ("⚖️", "Comparison",       "Pareto Analysis"),
        ("🎯", "Critique",         "Quality Audit"),
        ("🏆", "Recommendation",   "Luxury Advisory"),
        ("💾", "Learning",         "Memory Sync"),
    ]

    active_agents = st.session_state.get("active_agents", set())
    done_agents = st.session_state.get("done_agents", set())

    for icon, name, desc in agent_pipeline:
        if name in active_agents:
            cls, dot_cls = "active", "dot-active"
        elif name in done_agents:
            cls, dot_cls = "done", "dot-done"
        else:
            cls, dot_cls = "idle", "dot-idle"

        st.markdown(f"""
        <div class="agent-pill {cls}">
            <div class="agent-dot {dot_cls}"></div>
            <div>
                <div style="font-weight:600;">{icon} {name}</div>
                <div style="font-size:0.68rem; opacity:0.6;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Learned user context
    st.markdown('<div class="sidebar-section">🧬 Learned Profile</div>', unsafe_allow_html=True)
    try:
        ctx = get_user_context()
        st.markdown(f"""
        <div class="info-box">
            Sessions: <strong>{ctx['total_sessions']}</strong><br>
            Top Category: <strong>{ctx['most_searched_category'] or 'N/A'}</strong><br>
            Recent: <em>{(lambda s: s if len(s) <= 40 else s[0:40])( str((ctx['recent_queries'] or ['—'])[-1]) )}…</em>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="info-box">No profile yet — start searching!</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Quick stats
    st.markdown('<div class="sidebar-section">📊 Session Stats</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.query_count}</div>
            <div class="metric-label">Queries</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{msgs}</div>
            <div class="metric-label">Reports</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_final_state = None
        st.session_state.query_count = 0
        st.session_state.active_agents = set()
        st.session_state.done_agents = set()
        st.rerun()

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="apex-hero">
    <div class="apex-badge">✦ Powered by Multi-Agent AI &nbsp;·&nbsp; LangGraph Orchestration</div>
    <div class="apex-title">Intelligent Shopping Advisor</div>
    <div class="apex-subtitle">Your Elite AI Concierge for Exceptional Product Decisions</div>
</div>
""", unsafe_allow_html=True)

# ─── Suggested Queries ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <div style="font-size:0.72rem; font-weight:600; letter-spacing:0.15em; text-transform:uppercase;
                    color:rgba(212,175,55,0.6); margin-bottom:0.7rem;">✦ Popular Searches</div>
        <div>
            <span class="query-chip">Best laptop under $1500</span>
            <span class="query-chip">Gaming phone with best display</span>
            <span class="query-chip">Noise cancelling headphones</span>
            <span class="query-chip">Camera for professional photography</span>
            <span class="query-chip">MacBook Pro vs Dell XPS</span>
            <span class="query-chip">Budget smartwatch for fitness</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Chat History ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:flex-end; margin-bottom:1rem;">
            <div class="msg-label-user">You</div>
            <div class="msg-user">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:flex-start; margin-bottom:1rem;">
            <div class="msg-label-ai">💎 APEX Advisor</div>
            <div class="msg-ai">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Chat Input ───────────────────────────────────────────────────────────────
user_query = st.chat_input("Describe what you're looking for — budget, category, use case, preferences…")

if user_query:
    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.active_agents = set()
    st.session_state.done_agents = set()

    # Show user message immediately
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:flex-end; margin-bottom:1rem;">
        <div class="msg-label-user">You</div>
        <div class="msg-user">{user_query}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Processing UI ──
    status_container = st.empty()
    progress_bar = st.progress(0)

    initial_state: AgentState = {
        "query": user_query,
        "complexity": "simple",
        "selected_llm": "local",
        "category": "",
        "budget": {"min": None, "max": None},
        "intent_preferences": [],
        "priority_vector": {"price": 0.2, "performance": 0.2, "brand": 0.2, "durability": 0.2, "innovation": 0.2},
        "retrieved_products": [],
        "enriched_products": [],
        "scored_products": [],
        "critique_passed": False,
        "critique_feedback": "",
        "revision_count": 0,
        "final_recommendation": "",
        "error": None,
        "history": []
    }

    agent_order = ["supervisor", "preference", "retrieval", "enrichment", "comparison", "critique", "recommendation", "learning"]
    agent_display = {
        "supervisor": "Supervisor", "preference": "Preference",
        "retrieval": "Retrieval", "enrichment": "Enrichment",
        "comparison": "Comparison", "critique": "Critique",
        "recommendation": "Recommendation", "learning": "Learning"
    }

    try:
        workflow = create_workflow()
        final_step = None
        step_count: int = 0
        total_steps: int = len(agent_order)

        for event in workflow.stream(initial_state):
            for node_name, state_val in event.items():
                final_step = state_val
                display_name = agent_display.get(node_name, node_name.capitalize())

                # Update active/done agent tracking
                st.session_state.done_agents = st.session_state.get("done_agents", set())
                st.session_state.done_agents.add(display_name)
                st.session_state.active_agents = {display_name}

                step_count = step_count + 1  # type: ignore[operator]
                progress = min(int(step_count / total_steps * 100), 95)
                progress_bar.progress(progress)

                status_container.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(212,175,55,0.1),rgba(212,175,55,0.04));
                            border:1px solid rgba(212,175,55,0.25); border-radius:12px;
                            padding:0.8rem 1.2rem; font-size:0.85rem; color:rgba(212,175,55,0.9);">
                    <span style="animation:pulse 1s infinite; display:inline-block;">⚡</span>
                    &nbsp; <strong>{display_name} Agent</strong> is processing your query…
                </div>
                """, unsafe_allow_html=True)

        progress_bar.progress(100)
        status_container.empty()
        progress_bar.empty()

        st.session_state.active_agents = set()
        st.session_state.last_final_state = final_step

        if final_step:
            final_rec = final_step.get("final_recommendation", "")

            if not final_rec or len(final_rec) < 30:
                final_rec = "⚠️ The advisor could not generate a recommendation. Please check your API configuration or try rephrasing your query."

            # ── Display AI Response ──
            st.markdown(f"""
            <div style="display:flex; flex-direction:column; align-items:flex-start; margin-bottom:1.5rem;">
                <div class="msg-label-ai">💎 APEX Advisor</div>
                <div class="msg-ai">{final_rec}</div>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": final_rec})

            # ── Analytics Dashboard ──
            scored = final_step.get("scored_products", [])
            if scored:
                st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

                with st.expander("📊  Premium Analytics Dashboard — AI Reasoning & Product Scores", expanded=False):
                    # Meta row
                    col1, col2, col3, col4 = st.columns(4)
                    meta_items = [
                        ("Complexity", final_step.get("complexity", "—").capitalize()),
                        ("Model Tier", final_step.get("selected_llm", "—").capitalize()),
                        ("Products Found", str(len(scored))),
                        ("Category", final_step.get("category", "—").capitalize()),
                    ]
                    for col, (label, val) in zip([col1, col2, col3, col4], meta_items):
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value" style="font-size:1.2rem;">{val}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

                    # Extracted preferences
                    prefs = final_step.get("intent_preferences", [])
                    if prefs:
                        st.markdown("""
                        <div style="font-size:0.75rem; font-weight:700; letter-spacing:0.12em;
                                    text-transform:uppercase; color:rgba(212,175,55,0.7); margin-bottom:0.6rem;">
                            🧠 Extracted Intent Preferences
                        </div>
                        """, unsafe_allow_html=True)
                        chips = "".join(f'<span class="query-chip">{p}</span>' for p in prefs)
                        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
                        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

                    # Priority vector
                    pv = final_step.get("priority_vector", {})
                    if pv:
                        st.markdown("""
                        <div style="font-size:0.75rem; font-weight:700; letter-spacing:0.12em;
                                    text-transform:uppercase; color:rgba(212,175,55,0.7); margin-bottom:0.8rem;">
                            ⚖️ AI Priority Vector
                        </div>
                        """, unsafe_allow_html=True)
                        pv_cols = st.columns(len(pv))
                        for pcol, (k, v) in zip(pv_cols, pv.items()):
                            with pcol:
                                st.markdown(f"""
                                <div style="text-align:center;">
                                    <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
                                                font-weight:700; color:#D4AF37;">{v:.2f}</div>
                                    <div style="font-size:0.65rem; text-transform:uppercase;
                                                letter-spacing:0.1em; color:rgba(200,200,220,0.4); margin-top:0.2rem;">{k}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

                    # Product score cards
                    st.markdown("""
                    <div style="font-size:0.75rem; font-weight:700; letter-spacing:0.12em;
                                text-transform:uppercase; color:rgba(212,175,55,0.7); margin-bottom:0.8rem;">
                        🏆 Ranked Products — Pareto-Optimal Analysis
                    </div>
                    """, unsafe_allow_html=True)

                    top_n = [p for i, p in enumerate(scored) if i < 5]
                    for rank, prod in enumerate(top_n):
                        pareto_cls = "pareto" if prod.get("is_pareto_optimal") else ""
                        criteria = prod.get("criteria_scores", {})
                        overall = prod.get("overall_score", 0)

                        score_bars = ""
                        for dim, score in criteria.items():
                            pct = min(100, int(score * 10))
                            score_bars += f"""
                            <div class="score-bar-wrap">
                                <div class="score-label">{dim}</div>
                                <div class="score-bar-bg">
                                    <div class="score-bar-fill" style="width:{pct}%;"></div>
                                </div>
                                <div class="score-val">{score:.1f}</div>
                            </div>"""

                        pareto_badge = '<span style="color:#D4AF37; font-size:0.7rem;">✦ PARETO OPTIMAL</span>' if prod.get("is_pareto_optimal") else ""
                        tier_badge = f'<span style="background:rgba(255,255,255,0.05); border-radius:50px; padding:0.15rem 0.5rem; font-size:0.65rem; color:rgba(200,200,220,0.5);">{prod.get("value_tier","")}</span>'

                        st.markdown(f"""
                        <div class="product-card {pareto_cls}">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.6rem;">
                                <div>
                                    <div style="font-size:0.68rem; color:rgba(212,175,55,0.5);
                                                font-weight:700; letter-spacing:0.1em; margin-bottom:0.3rem;">
                                        RANK #{rank+1} &nbsp; {pareto_badge}
                                    </div>
                                    <div class="product-name">{prod.get('name','')}</div>
                                    <div class="product-meta">{prod.get('brand','')} &nbsp;·&nbsp; {prod.get('category','').capitalize()} &nbsp;·&nbsp; ⭐ {prod.get('rating','')}/5 &nbsp; {tier_badge}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div class="product-price">${prod.get('price','')}</div>
                                    <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
                                                color:#D4AF37; margin-top:0.3rem;">
                                        {overall:.2f}<span style="font-size:0.6rem; color:rgba(212,175,55,0.5);">/10</span>
                                    </div>
                                </div>
                            </div>
                            {score_bars}
                        </div>
                        """, unsafe_allow_html=True)

                    # Critique status
                    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
                    critique_ok = final_step.get("critique_passed", True)
                    critique_icon = "✅" if critique_ok else "⚠️"
                    critique_label = "Quality Audit Passed" if critique_ok else "Revision Applied"
                    feedback_text = final_step.get("critique_feedback", "")
                    st.markdown(f"""
                    <div class="info-box">
                        {critique_icon} <strong>{critique_label}</strong>
                        {"<br><em>" + feedback_text + "</em>" if feedback_text and not critique_ok else ""}
                        &nbsp; · &nbsp; Revision Cycles: <strong>{final_step.get("revision_count", 0)}</strong>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.error("The workflow did not produce a final state. Check agent logs.")

    except Exception as e:
        progress_bar.empty()
        status_container.empty()
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);
                    border-radius:12px; padding:1rem 1.4rem; color:rgba(252,165,165,0.9); font-size:0.9rem;">
            ⚠️ <strong>Processing Error</strong><br>{str(e)}
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:3rem 0 1rem 0;">
    <div class="gold-divider"></div>
    <div style="font-size:0.68rem; letter-spacing:0.2em; text-transform:uppercase;
                color:rgba(212,175,55,0.3); padding-top:1rem;">
        APEX Intelligent Shopping Advisor &nbsp;·&nbsp; Elite Edition &nbsp;·&nbsp;
        Multi-Agent LangGraph Architecture &nbsp;·&nbsp; Ahmad Yasin
    </div>
</div>
""", unsafe_allow_html=True)
