import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_videos

OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'visuals')
os.makedirs(OUTPUT, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2a3e',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'sans-serif',
})
ACCENT  = ['#7c3aed','#06b6d4','#f59e0b','#10b981','#ef4444','#ec4899','#8b5cf6','#14b8a6','#f97316','#3b82f6']

df = load_videos('IN')

# ── Chart 1: Top categories by video count ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
cat_counts = df['category'].value_counts().head(10)
bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=ACCENT[:10])
for bar, val in zip(bars, cat_counts.values[::-1]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9, color='#e0e0e0')
ax.set_xlabel('Number of trending videos')
ax.set_title('Which categories dominate trending in India?', fontsize=13, pad=12, color='#fff')
ax.grid(axis='x')
ax.set_xlim(0, cat_counts.max() * 1.15)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/1_category_counts.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 1 done")

# ── Chart 2: Avg views by category ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
avg_views = df.groupby('category')['views'].median().sort_values(ascending=True).tail(10)
bars = ax.barh(avg_views.index, avg_views.values, color='#06b6d4')
for bar, val in zip(bars, avg_views.values):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f'{val/1e6:.1f}M', va='center', fontsize=9, color='#e0e0e0')
ax.set_xlabel('Median views')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1e6:.0f}M'))
ax.set_title('Which category gets the most views per video?', fontsize=13, pad=12, color='#fff')
ax.grid(axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/2_views_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 2 done")

# ── Chart 3: Best hour to publish ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
hour_views = df.groupby('publish_hour')['views'].median()
colors = ['#f59e0b' if v == hour_views.max() else '#7c3aed' for v in hour_views]
ax.bar(hour_views.index, hour_views.values, color=colors, width=0.7)
ax.set_xlabel('Hour of day (UTC)')
ax.set_ylabel('Median views')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1e6:.1f}M'))
ax.set_xticks(range(0, 24))
ax.set_title('Best hour to publish for maximum views', fontsize=13, pad=12, color='#fff')
best_hour = hour_views.idxmax()
ax.annotate(f'Peak: {best_hour}:00', xy=(best_hour, hour_views.max()),
            xytext=(best_hour+1.5, hour_views.max()*0.95),
            color='#f59e0b', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1))
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/3_best_hour.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 3 done")

# ── Chart 4: Best day to publish ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_views = df.groupby('publish_day')['views'].median().reindex(day_order)
colors = ['#10b981' if v == day_views.max() else '#7c3aed' for v in day_views]
ax.bar(day_views.index, day_views.values, color=colors, width=0.6)
ax.set_ylabel('Median views')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1e6:.1f}M'))
ax.set_title('Best day of week to publish', fontsize=13, pad=12, color='#fff')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/4_best_day.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 4 done")

# ── Chart 5: Days to trend distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
trend_days = df['days_to_trend'].clip(0, 30)
ax.hist(trend_days, bins=31, color='#ec4899', edgecolor='#0f0f0f', linewidth=0.4)
ax.set_xlabel('Days from publish to trending')
ax.set_ylabel('Number of videos')
ax.set_title('How fast do videos go viral? (days to trend)', fontsize=13, pad=12, color='#fff')
pct_fast = (df['days_to_trend'] <= 1).mean() * 100
ax.axvline(1, color='#f59e0b', linestyle='--', linewidth=1.5)
ax.text(1.5, ax.get_ylim()[1]*0.85, f'{pct_fast:.0f}% trend\nwithin 1 day',
        color='#f59e0b', fontsize=9)
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/5_days_to_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 5 done")

# ── Chart 6: Title length vs views scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sample = df[df['views'] < df['views'].quantile(0.99)].sample(2000, random_state=42)
sc = ax.scatter(sample['title_length'], sample['views']/1e6,
                alpha=0.3, s=10, c=sample['like_ratio'], cmap='plasma')
plt.colorbar(sc, ax=ax, label='Like ratio')
ax.set_xlabel('Title length (characters)')
ax.set_ylabel('Views (millions)')
ax.set_title('Does title length affect views?', fontsize=13, pad=12, color='#fff')
ax.grid(True)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/6_title_vs_views.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 6 done")

# ── Chart 7: Tag count vs views ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
tag_bins = pd.cut(df['tag_count'], bins=[0,5,10,20,30,50,200], labels=['1-5','6-10','11-20','21-30','31-50','50+'])
tag_views = df.groupby(tag_bins, observed=True)['views'].median()
ax.bar(tag_views.index, tag_views.values/1e6, color=ACCENT[:6], width=0.6)
ax.set_xlabel('Number of tags')
ax.set_ylabel('Median views (millions)')
ax.set_title('How many tags do viral videos use?', fontsize=13, pad=12, color='#fff')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/7_tags_vs_views.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 7 done")

# ── Chart 8: Correlation heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
cols = ['views','likes','dislikes','comment_count','days_to_trend','title_length','tag_count','like_ratio','engagement_rate']
corr = df[cols].corr()
mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
sns.heatmap(corr, ax=ax, cmap='RdPu', annot=True, fmt='.2f',
            linewidths=0.5, linecolor='#0f0f0f', cbar_kws={'shrink':0.8})
ax.set_title('Feature correlation matrix', fontsize=13, pad=12, color='#fff')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/8_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 8 done")

print("\n🎉 All 8 EDA charts saved to visuals/")
