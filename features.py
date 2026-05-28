import pandas as pd
import numpy as np
from textblob import TextBlob
import re, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_videos

# ── Emotion / clickbait words ────────────────────────────────────────────────
EMOTION_WORDS = [
    'shocking','amazing','unbelievable','viral','exclusive','breaking',
    'secret','revealed','never','first','best','worst','biggest','crazy',
    'must watch','incredible','official','full','new','top','real','truth',
    'funny','cute','fail','win','epic','insane','mind blowing','omg'
]

def title_sentiment(title):
    try:
        return TextBlob(str(title)).sentiment.polarity
    except:
        return 0.0

def emotion_score(title):
    t = str(title).lower()
    return sum(1 for w in EMOTION_WORDS if w in t)

def has_numbers(title):
    return int(bool(re.search(r'\d', str(title))))

def has_question(title):
    return int('?' in str(title))

def has_caps_words(title):
    words = str(title).split()
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return min(caps, 5)  # cap at 5

def engineer_features(df):
    print("⚙️  Engineering features...")

    # Text features
    df['sentiment']      = df['title'].apply(title_sentiment)
    df['emotion_score']  = df['title'].apply(emotion_score)
    df['has_numbers']    = df['title'].apply(has_numbers)
    df['has_question']   = df['title'].apply(has_question)
    df['caps_words']     = df['title'].apply(has_caps_words)
    df['title_length']   = df['title'].str.len()
    df['title_words']    = df['title'].str.split().str.len()

    # Time features
    df['is_weekend']     = df['publish_day'].isin(['Saturday','Sunday']).astype(int)
    df['is_prime_time']  = df['publish_hour'].between(17, 21).astype(int)

    # Engagement features (already in loader, keep)
    # like_ratio, engagement_rate, tag_count, days_to_trend

    # ── Virality Score (0–100) ───────────────────────────────────────────────
    # Normalise each signal to 0-1 then weight & sum → scale to 0-100
    def norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    score = (
        norm(np.log1p(df['views']))        * 0.30 +   # views (log)
        norm(df['like_ratio'])             * 0.20 +   # quality signal
        norm(df['engagement_rate'])        * 0.15 +   # engagement depth
        norm(1 / (df['days_to_trend'] + 1))* 0.15 +  # speed of trending
        norm(df['emotion_score'])          * 0.10 +   # clickbait power
        norm(df['tag_count'].clip(0, 50))  * 0.05 +   # discoverability
        norm(df['sentiment'].abs())        * 0.05     # emotional polarity
    )
    df['virality_score'] = (score * 100).round(1)

    print(f"✅ Features engineered. Score range: {df['virality_score'].min():.1f} – {df['virality_score'].max():.1f}")
    return df

if __name__ == '__main__':
    df = load_videos('IN')
    df = engineer_features(df)

    print("\n📊 Virality Score distribution:")
    print(df['virality_score'].describe().round(2))

    print("\n🏆 Top 10 most viral videos:")
    top = df.nlargest(10, 'virality_score')[['title','category','views','virality_score']]
    for _, row in top.iterrows():
        print(f"  [{row['virality_score']:5.1f}] {row['views']/1e6:.1f}M views — {row['title'][:60]}")

    print("\n💀 Bottom 5 least viral:")
    bot = df.nsmallest(5, 'virality_score')[['title','virality_score','views']]
    for _, row in bot.iterrows():
        print(f"  [{row['virality_score']:5.1f}] {row['views']:,} views — {row['title'][:60]}")

def plot_features(df):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'visuals')
    plt.rcParams.update({
        'figure.facecolor':'#0f0f0f','axes.facecolor':'#1a1a2e',
        'axes.edgecolor':'#444','axes.labelcolor':'#e0e0e0',
        'xtick.color':'#aaa','ytick.color':'#aaa','text.color':'#e0e0e0',
        'grid.color':'#2a2a3e','grid.linestyle':'--','grid.alpha':0.5,
    })

    # Chart 9: Virality score distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['virality_score'], bins=50, color='#7c3aed', edgecolor='#0f0f0f', lw=0.3)
    ax.axvline(df['virality_score'].mean(), color='#f59e0b', linestyle='--', lw=1.5, label=f"Mean: {df['virality_score'].mean():.1f}")
    ax.axvline(df['virality_score'].quantile(0.9), color='#10b981', linestyle='--', lw=1.5, label=f"Top 10%: {df['virality_score'].quantile(0.9):.1f}+")
    ax.set_xlabel('Virality Score (0–100)')
    ax.set_ylabel('Number of videos')
    ax.set_title('Distribution of Virality Scores across 16K Indian YouTube videos', fontsize=13, pad=12, color='#fff')
    ax.legend(fontsize=9)
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/9_virality_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart 9 done")

    # Chart 10: Feature importance (correlation to virality score)
    fig, ax = plt.subplots(figsize=(9, 5))
    feat_cols = ['like_ratio','engagement_rate','days_to_trend','emotion_score',
                 'tag_count','sentiment','title_length','has_numbers','is_prime_time','caps_words']
    corrs = df[feat_cols + ['virality_score']].corr()['virality_score'].drop('virality_score').sort_values()
    colors = ['#ef4444' if v < 0 else '#10b981' for v in corrs]
    ax.barh(corrs.index, corrs.values, color=colors)
    ax.axvline(0, color='#aaa', lw=0.8)
    ax.set_xlabel('Correlation with Virality Score')
    ax.set_title('What features drive virality?', fontsize=13, pad=12, color='#fff')
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/10_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart 10 done")

    # Chart 11: Virality score by category (boxplot)
    fig, ax = plt.subplots(figsize=(11, 5))
    order = df.groupby('category')['virality_score'].median().sort_values(ascending=False).index
    data = [df[df['category'] == c]['virality_score'].values for c in order]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='#f59e0b', lw=2),
                    whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'),
                    flierprops=dict(marker='.', color='#555', markersize=2))
    colors = ['#7c3aed','#06b6d4','#f59e0b','#10b981','#ef4444','#ec4899',
              '#8b5cf6','#14b8a6','#f97316','#3b82f6','#a3e635','#fb923c','#e879f9']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(order, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Virality Score')
    ax.set_title('Virality Score by Category', fontsize=13, pad=12, color='#fff')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/11_virality_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart 11 done")

    # Chart 12: Emotion score vs virality
    fig, ax = plt.subplots(figsize=(9, 4))
    emo_vs = df.groupby('emotion_score')['virality_score'].mean().head(8)
    ax.bar(emo_vs.index, emo_vs.values, color='#ec4899', width=0.6)
    ax.set_xlabel('Number of emotion/clickbait words in title')
    ax.set_ylabel('Average Virality Score')
    ax.set_title('Do clickbait words boost virality?', fontsize=13, pad=12, color='#fff')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/12_emotion_vs_virality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart 12 done")
    print("\n🎉 Feature charts done!")

if __name__ == '__main__':
    df = load_videos('IN')
    df = engineer_features(df)
    plot_features(df)
