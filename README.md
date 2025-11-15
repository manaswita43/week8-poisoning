# IRIS MLOps Pipeline — Data Poisoning, DVC, MlFlow

This assignment builds a complete, production-ready MLOps pipeline for the IRIS dataset using modern tools such as DVC, MLflow, while also exploring data poisoning vulnerabilities and evaluating system robustness.

## Project Structure
```
week8-poisoning/
│
├── data/
│   └── iris.csv
│
├── poisoned_data/
│   ├── iris_poisoned_5.csv
│   ├── iris_poisoned_10.csv
│   └── iris_poisoned_50.csv
│
├── models/
│   └── model.joblib
├── requirements.txt
├── sanity_test.py
│
├── scripts/
│   ├── poison_data.py
│   └── train.py
│
└── .github/
    └── workflows/
        └── ci.yml
```
