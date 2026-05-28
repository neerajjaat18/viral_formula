# 🔥 Viral Formula Analyzer
### YouTube Trending Video Analysis — India

## Project Structure
```
viral_formula/
├── data/               # INvideos.csv + IN_category_id.json
├── src/
│   ├── data_loader.py  # Load & clean data + feature engineering base
│   ├── features.py     # Virality score formula + feature engineering
│   └── model.py        # ML model training & evaluation
├── models/             # Saved GradientBoosting model
├── visuals/            # All 15 charts generated
├── app.py              # Streamlit dashboard
└── README.md
```

## Setup & Run
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost streamlit textblob joblib

# Step 1 - Load data
python src/data_loader.py

# Step 2 - Engineer features
python src/features.py

# Step 3 - Train model
python src/model.py

# Step 4 - Launch dashboard
streamlit run app.py
```

## Model Performance
- Algorithm: Gradient Boosting Regressor
- R² Score: 0.852 (85.2% accuracy)
- MAE: 2.09 (on 0-100 scale)

## Virality Score Formula
| Signal         | Weight |
|---------------|--------|
| Views (log)    | 30%    |
| Like ratio     | 20%    |
| Engagement     | 15%    |
| Speed to trend | 15%    |
| Emotion words  | 10%    |
| Tag count      | 5%     |
| Sentiment      | 5%     |

## Dashboard Pages
1. 🎯 Score My Video — predict virality of any title
2. 📊 EDA Dashboard — explore trending patterns
3. 🏆 Top Videos — browse most viral content
4. 🧠 Model Insights — understand what drives virality
