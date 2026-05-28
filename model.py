import pandas as pd
import numpy as np
import os, sys, joblib
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_videos
from features import engineer_features

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    'like_ratio', 'engagement_rate', 'days_to_trend', 'tag_count',
    'title_length', 'title_words', 'emotion_score', 'sentiment',
    'has_numbers', 'has_question', 'caps_words',
    'publish_hour', 'is_weekend', 'is_prime_time'
]

def train(df):
    X = df[FEATURES].fillna(0)
    y = df['virality_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        'Ridge':    Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        'XGBoost':  XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0),
    }

    print("🏋️  Training models...\n")
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)
        results[name] = {'model': model, 'mae': mae, 'r2': r2}
        print(f"  {name:<22} MAE={mae:.2f}  R²={r2:.3f}")

    best_name = min(results, key=lambda k: results[k]['mae'])
    best = results[best_name]
    print(f"\n🏆 Best model: {best_name}  (MAE={best['mae']:.2f}, R²={best['r2']:.3f})")

    # Save best model
    joblib.dump({'model': best['model'], 'features': FEATURES}, f'{MODEL_DIR}/virality_model.pkl')
    print(f"💾 Model saved to models/virality_model.pkl")

    # Feature importances
    if hasattr(best['model'], 'feature_importances_'):
        fi = pd.Series(best['model'].feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\n📊 Feature importances:")
        for feat, imp in fi.items():
            bar = '█' * int(imp * 80)
            print(f"  {feat:<22} {bar} {imp:.3f}")

    return best['model'], results, X_test, y_test

def plot_model(results, X_test, y_test):
    import matplotlib.pyplot as plt
    OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'visuals')
    plt.rcParams.update({
        'figure.facecolor':'#0f0f0f','axes.facecolor':'#1a1a2e',
        'axes.edgecolor':'#444','axes.labelcolor':'#e0e0e0',
        'xtick.color':'#aaa','ytick.color':'#aaa','text.color':'#e0e0e0',
        'grid.color':'#2a2a3e','grid.linestyle':'--','grid.alpha':0.5,
    })

    # Chart 13: Model comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(results.keys())
    maes  = [results[n]['mae'] for n in names]
    r2s   = [results[n]['r2']  for n in names]
    x = np.arange(len(names))
    b1 = ax.bar(x - 0.2, maes, 0.35, label='MAE (lower=better)', color='#7c3aed')
    ax2 = ax.twinx()
    b2 = ax2.bar(x + 0.2, r2s, 0.35, label='R² (higher=better)', color='#10b981', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('MAE'); ax2.set_ylabel('R² Score')
    ax.set_title('Model comparison: MAE vs R²', fontsize=13, pad=12, color='#fff')
    lines = [b1, b2]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/13_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart 13 done")

    # Chart 14: Predicted vs Actual (best model)
    best_name = min(results, key=lambda k: results[k]['mae'])
    preds = results[best_name]['model'].predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, preds, alpha=0.15, s=8, color='#7c3aed')
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], color='#f59e0b', lw=1.5, linestyle='--', label='Perfect prediction')
    ax.set_xlabel('Actual Virality Score')
    ax.set_ylabel('Predicted Virality Score')
    ax.set_title(f'Predicted vs Actual — {best_name}', fontsize=13, pad=12, color='#fff')
    ax.legend(fontsize=9); ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/14_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart 14 done")

    # Chart 15: Feature importance
    best_model = results[best_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ['#7c3aed' if v >= fi.quantile(0.75) else '#444' for v in fi]
        ax.barh(fi.index, fi.values, color=colors)
        ax.set_xlabel('Feature importance')
        ax.set_title(f'Top features driving virality ({best_name})', fontsize=13, pad=12, color='#fff')
        ax.grid(axis='x')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT}/15_feature_importance_model.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Chart 15 done")

def predict_new(title, category='Entertainment', tag_count=20,
                publish_hour=19, is_weekend=0):
    """Score any new video idea."""
    from features import emotion_score, title_sentiment, has_numbers, has_question, has_caps_words
    bundle = joblib.load(f'{MODEL_DIR}/virality_model.pkl')
    model  = bundle['model']

    row = {
        'like_ratio': 0.92,        # assume decent quality
        'engagement_rate': 0.04,
        'days_to_trend': 1,        # assume 1 day
        'tag_count': tag_count,
        'title_length': len(title),
        'title_words': len(title.split()),
        'emotion_score': emotion_score(title),
        'sentiment': title_sentiment(title),
        'has_numbers': has_numbers(title),
        'has_question': has_question(title),
        'caps_words': has_caps_words(title),
        'publish_hour': publish_hour,
        'is_weekend': is_weekend,
        'is_prime_time': int(17 <= publish_hour <= 21),
    }
    X = pd.DataFrame([row])[FEATURES]
    score = model.predict(X)[0]
    return round(float(score), 1)

if __name__ == '__main__':
    df = load_videos('IN')
    df = engineer_features(df)
    best_model, results, X_test, y_test = train(df)
    plot_model(results, X_test, y_test)

    print("\n🔮 Live predictions on new video ideas:")
    test_titles = [
        "TOP 10 SHOCKING Facts About India You Won't Believe!",
        "My Morning Routine",
        "INCREDIBLE Mind Blowing Science Experiments - MUST WATCH!!",
        "Episode 245 - Daily News Update",
        "Secret Life Hack That Will Change Everything REVEALED",
    ]
    for t in test_titles:
        score = predict_new(t)
        bar = '█' * int(score / 5)
        print(f"  {score:5.1f}/100  {bar}  {t[:55]}")
