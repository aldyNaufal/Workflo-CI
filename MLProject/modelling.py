import os, json, mlflow
import pandas as pd
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Gunakan file store lokal
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))

df = pd.read_csv("data_preprocessed/balanced.csv")
X, y = df['clean_text'], df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid = {
    "tfidf__max_features": [5000, 10000],
    "rf__n_estimators": [200, 400],
    "rf__max_depth": [None, 20]
}

def plot_and_log_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=['Negative','Neutral','Positive'])
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    path = "cm.png"; fig.savefig(path); plt.close(fig)
    mlflow.log_artifact(path)

# Jalankan langsung di dalam parent run (tidak perlu nested)
for params in product(*grid.values()):
    param_dict = dict(zip(grid.keys(), params))
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("rf", RandomForestClassifier(random_state=42))
    ])
    pipe.set_params(**{
        "tfidf__max_features": param_dict["tfidf__max_features"],
        "rf__n_estimators": param_dict["rf__n_estimators"],
        "rf__max_depth": param_dict["rf__max_depth"]
    })

    with mlflow.start_run(nested=False):  # gunakan run biasa saja
        mlflow.log_params(param_dict)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        plot_and_log_cm(y_test, y_pred)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print(param_dict, "=> acc:", acc, "f1:", f1)
