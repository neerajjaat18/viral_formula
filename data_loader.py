import pandas as pd
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_categories(country='IN'):
    path = os.path.join(DATA_DIR, f'{country}_category_id.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['id']: item['snippet']['title'] for item in data['items']}

def load_videos(country='IN'):
    path = os.path.join(DATA_DIR, f'{country}videos.csv')
    df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')

    # Parse dates and times
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
    df['publish_time']  = pd.to_datetime(df['publish_time'], utc=True)

    # Map category names
    cats = load_categories(country)
    df['category'] = df['category_id'].astype(str).map(cats).fillna('Unknown')

    # Engineered features
    df['days_to_trend'] = (df['trending_date'] - df['publish_time'].dt.tz_localize(None)).dt.days.clip(lower=0)
    df['publish_hour']  = df['publish_time'].dt.hour
    df['publish_day']   = df['publish_time'].dt.day_name()
    df['title_length']  = df['title'].str.len()
    df['tag_count']     = df['tags'].apply(lambda x: 0 if x == '[none]' else len(str(x).split('|')))
    df['like_ratio']    = df['likes'] / (df['likes'] + df['dislikes'] + 1)
    df['engagement_rate'] = (df['likes'] + df['dislikes'] + df['comment_count']) / (df['views'] + 1)

    # Remove error/removed videos
    df = df[df['video_error_or_removed'] == False].copy()
    df.drop_duplicates(subset='video_id', keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

if __name__ == '__main__':
    df = load_videos('IN')
    print(f"✅ Loaded {len(df):,} unique trending videos")
    print(f"📅 Date range: {df['trending_date'].min().date()} → {df['trending_date'].max().date()}")
    print(f"📂 Categories: {df['category'].nunique()} unique")
    print(f"\nSample engineered features:")
    print(df[['title','category','views','like_ratio','days_to_trend','tag_count','engagement_rate']].head(5).to_string())
