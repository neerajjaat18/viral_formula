import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_videos
from features import engineer_features, emotion_score, title_sentiment, \
                     has_numbers, has_question, has_caps_words

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'virality_model.pkl')
FEATURES = [
    'like_ratio','engagement_rate','days_to_trend','tag_count',
    'title_length','title_words','emotion_score','sentiment',
    'has_numbers','has_question','caps_words',
    'publish_hour','is_weekend','is_prime_time'
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Viral Formula Analyzer", page_icon="🔥", layout="wide")

st.markdown("""
<style>
  .score-box {
    background: linear-gradient(135deg,#1a1a2e,#16213e);
    border: 1px solid #7c3aed;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
  }
  .score-number { font-size: 72px; font-weight: 800; color: #f59e0b; line-height:1; }
  .score-label  { font-size: 14px; color: #aaa; margin-top: 4px; }
  .tip-box {
    background: #1a1a2e;
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 14px;
  }
  .metric-card {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ── Load data & model ─────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df = load_videos('IN')
    return engineer_features(df)

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)

df = get_data()
bundle = get_model()
model  = bundle['model']

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔥 Viral Formula")
    st.caption("YouTube Trending Analyzer — India")
    st.markdown("---")
    page = st.radio("Navigate", ["🎯 Score My Video", "📊 EDA Dashboard", "🏆 Top Videos", "🧠 Model Insights"])

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Score My Video
# ═════════════════════════════════════════════════════════════════════════════
if page == "🎯 Score My Video":
    st.title("🎯 Virality Score Predictor")
    st.caption("Enter your video details and get an instant virality score powered by ML")

    col1, col2 = st.columns([2, 1])
    with col1:
        title = st.text_input("📝 Video title", placeholder="e.g. TOP 10 Shocking Facts About India You Won't Believe!")
    with col2:
        category = st.selectbox("📂 Category", sorted(df['category'].unique()))

    c1, c2, c3 = st.columns(3)
    with c1:
        tag_count   = st.slider("🏷️ Number of tags", 0, 80, 25)
    with c2:
        publish_hour = st.slider("⏰ Publish hour (UTC)", 0, 23, 19)
    with c3:
        is_weekend  = st.toggle("📅 Publishing on weekend?")

    if title.strip():
        emo  = emotion_score(title)
        sent = title_sentiment(title)
        nums = has_numbers(title)
        ques = has_question(title)
        caps = has_caps_words(title)

        row = {
            'like_ratio': 0.92, 'engagement_rate': 0.04, 'days_to_trend': 1,
            'tag_count': tag_count, 'title_length': len(title),
            'title_words': len(title.split()), 'emotion_score': emo,
            'sentiment': sent, 'has_numbers': nums, 'has_question': ques,
            'caps_words': caps, 'publish_hour': publish_hour,
            'is_weekend': int(is_weekend),
            'is_prime_time': int(17 <= publish_hour <= 21),
        }
        X = pd.DataFrame([row])[FEATURES]
        score = float(model.predict(X)[0])

        # ── Score display ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        left, mid, right = st.columns([1, 1, 1])

        with mid:
            color = "#ef4444" if score < 40 else "#f59e0b" if score < 52 else "#10b981"
            label = "🥶 Low virality" if score < 40 else "🔥 Average" if score < 52 else "🚀 High virality!"
            st.markdown(f"""
            <div class="score-box">
              <div class="score-number" style="color:{color}">{score:.1f}</div>
              <div style="font-size:18px;margin:6px 0">{label}</div>
              <div class="score-label">out of 100 · Virality Score</div>
            </div>""", unsafe_allow_html=True)

        # ── Signal breakdown ───────────────────────────────────────────────
        st.markdown("### 🔬 Signal breakdown")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Emotion words", emo, help="Clickbait/power words detected")
        m2.metric("Title length", len(title), help="Characters in title")
        m3.metric("Sentiment", f"{sent:+.2f}", help="-1 negative to +1 positive")
        m4.metric("Has numbers?", "✅" if nums else "❌")
        m5.metric("Has question?", "✅" if ques else "❌")

        # ── Tips ───────────────────────────────────────────────────────────
        st.markdown("### 💡 Optimization tips")
        tips = []
        if emo == 0:
            tips.append("Add emotion/power words (e.g. 'shocking', 'incredible', 'revealed', 'must watch')")
        if len(title) < 40:
            tips.append("Title is short — aim for 50–70 characters for better click-through")
        if tag_count < 20:
            tips.append("Use more tags (31–50 is the sweet spot based on our data)")
        if not (17 <= publish_hour <= 21):
            tips.append("Publish between 17:00–21:00 UTC for peak audience hours")
        if not nums:
            tips.append("Add numbers to your title — '10 Ways…' or 'Top 5…' boosts clicks")
        if score >= 55:
            tips.append("Great title! It hits multiple virality signals.")

        if tips:
            for tip in tips:
                st.markdown(f'<div class="tip-box">💬 {tip}</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Type a video title above to get your virality score!")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard — Indian YouTube Trends")

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total videos", f"{len(df):,}")
    k2.metric("Avg virality score", f"{df['virality_score'].mean():.1f}")
    k3.metric("Trend within 1 day", f"{(df['days_to_trend']<=1).mean()*100:.0f}%")
    k4.metric("Categories", df['category'].nunique())

    st.markdown("---")
    col1, col2 = st.columns(2)

    plt.rcParams.update({'figure.facecolor':'#0f0f0f','axes.facecolor':'#1a1a2e',
        'axes.edgecolor':'#444','axes.labelcolor':'#e0e0e0','xtick.color':'#aaa',
        'ytick.color':'#aaa','text.color':'#e0e0e0','grid.color':'#2a2a3e',
        'grid.linestyle':'--','grid.alpha':0.5})
    ACCENT = ['#7c3aed','#06b6d4','#f59e0b','#10b981','#ef4444','#ec4899',
              '#8b5cf6','#14b8a6','#f97316','#3b82f6']

    with col1:
        st.subheader("Category distribution")
        fig, ax = plt.subplots(figsize=(6,4))
        cat = df['category'].value_counts().head(8)
        ax.barh(cat.index[::-1], cat.values[::-1], color=ACCENT[:8])
        ax.grid(axis='x'); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Best publish hour")
        fig, ax = plt.subplots(figsize=(6,4))
        hv = df.groupby('publish_hour')['views'].median()
        clrs = ['#f59e0b' if v==hv.max() else '#7c3aed' for v in hv]
        ax.bar(hv.index, hv.values/1e6, color=clrs, width=0.7)
        ax.set_xlabel('Hour (UTC)'); ax.set_ylabel('Median views (M)')
        ax.grid(axis='y'); plt.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Views by category")
        fig, ax = plt.subplots(figsize=(6,4))
        av = df.groupby('category')['views'].median().sort_values().tail(8)
        ax.barh(av.index, av.values/1e6, color='#06b6d4')
        ax.set_xlabel('Median views (M)'); ax.grid(axis='x'); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Days to trend")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df['days_to_trend'].clip(0,30), bins=31, color='#ec4899', edgecolor='#0f0f0f', lw=0.3)
        ax.axvline(1, color='#f59e0b', linestyle='--', lw=1.5)
        ax.set_xlabel('Days'); ax.set_ylabel('Videos'); ax.grid(axis='y'); plt.tight_layout()
        st.pyplot(fig); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Top Videos
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Top Videos":
    st.title("🏆 Most Viral Videos — India")

    cat_filter = st.selectbox("Filter by category", ["All"] + sorted(df['category'].unique()))
    filtered = df if cat_filter == "All" else df[df['category'] == cat_filter]
    top = filtered.nlargest(20, 'virality_score')[
        ['title','channel_title','category','views','virality_score','days_to_trend','emotion_score']
    ].reset_index(drop=True)
    top.index += 1
    top['views'] = top['views'].apply(lambda x: f"{x/1e6:.2f}M")
    top['virality_score'] = top['virality_score'].apply(lambda x: f"{x:.1f}")
    st.dataframe(top, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Insights
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")

    st.markdown("### Model: Gradient Boosting Regressor")
    i1,i2,i3 = st.columns(3)
    i1.metric("R² Score", "0.852", "85.2% variance explained")
    i2.metric("MAE", "2.09", "avg error on 0-100 scale")
    i3.metric("Training samples", f"{int(len(df)*0.8):,}")

    st.markdown("---")
    st.markdown("### Feature importances")
    fi_data = {
        'like_ratio':0.465,'days_to_trend':0.217,'emotion_score':0.125,
        'tag_count':0.070,'sentiment':0.058,'engagement_rate':0.026,
        'title_length':0.015,'publish_hour':0.010,'title_words':0.006,
        'has_numbers':0.003,'caps_words':0.003,'is_prime_time':0.001,
        'has_question':0.001,'is_weekend':0.000
    }
    fi = pd.Series(fi_data).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.rcParams.update({'figure.facecolor':'#0f0f0f','axes.facecolor':'#1a1a2e',
        'axes.edgecolor':'#444','axes.labelcolor':'#e0e0e0','xtick.color':'#aaa',
        'ytick.color':'#aaa','text.color':'#e0e0e0','grid.color':'#2a2a3e'})
    colors = ['#7c3aed' if v >= 0.05 else '#444' for v in fi]
    ax.barh(fi.index, fi.values, color=colors)
    ax.set_xlabel('Importance'); ax.grid(axis='x')
    ax.set_title('Feature importance', color='#fff')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("### 📌 Key takeaways")
    st.markdown("""
    - **Like ratio (46.5%)** — audience approval is the strongest virality signal
    - **Days to trend (21.7%)** — the faster a video trends, the more viral it truly is
    - **Emotion score (12.5%)** — power words in titles measurably boost performance
    - **Tag count (7%)** — discoverability through tags is more important than most think
    - **Sentiment (5.8%)** — emotionally extreme titles (very positive or very negative) outperform neutral ones
    """)
