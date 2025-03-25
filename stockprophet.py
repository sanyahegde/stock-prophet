import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Step 1: Download dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update")
print("Dataset downloaded to:", path)

# Step 2: Load stock data (e.g., AAPL)
csv_path = os.path.join(path, "AAPL.csv")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Step 3: Feature engineering
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()

# Step 4: Label creation: 1 if tomorrow's close > today's, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Step 5: Clean and drop NaN
df.dropna(inplace=True)

# Step 6: Define features and labels
features = ['Open', 'Close', 'Volume', 'MA5', 'MA10', 'MA20']
X = df[features]
y = df['Target']

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 8: Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Predict and evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Optional: Show feature importances
importances = model.feature_importances_
for feature, imp in zip(features, importances):
    print(f"{feature}: {imp:.4f}")

# Step 9.5: Binary classification with threshold
probs = model.predict_proba(X_test)[:, 1]  # Probabilities of class 1 (stock going up)
binary_preds = ['Yes' if prob > 0.5 else 'No' for prob in probs]

print("\nBinary classification results (Yes = stock predicted to go up):")
for i in range(10):
    print(f"Prediction {i+1}: {binary_preds[i]} (Prob: {probs[i]:.2f})")

# Step 10: Apply SM-2 to track learning on hard-to-predict samples
def sm2(quality, previous_interval=1, repetition=0, easiness=2.5):
    if quality < 3:
        return 1, 0, easiness
    repetition += 1
    if repetition == 1:
        interval = 1
    elif repetition == 2:
        interval = 6
    else:
        interval = round(previous_interval * easiness)
    easiness = max(1.3, easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return interval, repetition, easiness

# Simulate SM-2 scores on predictions
print("\nApplying SM-2 spaced repetition scores on test predictions:")
interval, repetition, easiness = 1, 0, 2.5
for i in range(len(preds)):
    correct = preds[i] == y_test.iloc[i]
    quality = 5 if correct else 2  # simulate high or low quality recall
    interval, repetition, easiness = sm2(quality, interval, repetition, easiness)
    print(f"Day {i+1}: Correct={correct}, Interval={interval}, Repetition={repetition}, Easiness={easiness:.2f}")
    if i > 10:
        break  # limit output
