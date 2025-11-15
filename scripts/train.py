import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import joblib
import sys

dataset = sys.argv[1]
exp_name = sys.argv[2]

mlflow.set_experiment(exp_name)

df = pd.read_csv(dataset)
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("data_source", dataset)

    joblib.dump(model, "models/model.joblib")
    mlflow.log_artifact("models/model.joblib")

print("Training complete:", dataset, "Accuracy:", acc)
