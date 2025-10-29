import os, mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # atau mlruns lokal default
mlflow.set_experiment("netflix_reviews_rf")

df = pd.read_csv("namadataset_preprocessing/balanced.csv")
X, y = df['clean_text'], df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("rf", RandomForestClassifier(random_state=42))
])

mlflow.sklearn.autolog()
with mlflow.start_run():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
