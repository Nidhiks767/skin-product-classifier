import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example dummy training data (replace with real labeled data)
data = [
    ("spf sunscreen lotion", "Sunscreen"),
    ("vitamin c serum", "Serum"),
    ("face wash cleanser", "Cleanser"),
    ("moisturizing cream", "Moisturizer"),
]

df = pd.DataFrame(data, columns=["text", "label"])

# Model pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Train
pipeline.fit(df["text"], df["label"])

# Save model
joblib.dump(pipeline, "category_model.pkl")

print("✅ Model trained and saved!")