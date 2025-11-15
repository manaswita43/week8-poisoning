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

---

## Implementation
The workflow begins by organizing and versioning data using DVC with a Google Cloud Storage backend, ensuring reproducible datasets. To study model reliability, the dataset is intentionally modified with 5%, 10%, and 50% random noise (data poisoning). Each poisoned dataset is tracked through DVC and used to train separate machine learning models using MLflow for experiment tracking. The results show expected degradation in accuracy as data quality worsens, helping illustrate real-world data integrity risks.

CI runs sanity tests (sanity_test.py) on every push.

**Note:** Please refer **results** folder to compare the comparisons of model run on different poisoned datasets.
