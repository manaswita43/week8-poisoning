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

---

## How to mitigate Poison attacks
### 1. Outlier Detection Pre-Processing
Use algorithms like:
- Isolation Forest
- DBSCAN
- Local Outlier Factor
These detect abnormal feature values before training.

### 2. Use Robust Models
Some models tolerate poisons better:
- Random Forest
- Gradient Boosting
- RANSAC

### 3. Increase Dataset Size
When data quality drops, data quantity must increase.

If noise = 10%,
Need ~1.5x to 2x more samples to maintain performance.

If noise = 50%,
Need ~4x to 5x more clean samples.

### 4. Differential Privacy During Training
Bound the influence of individual samples.

### 5. Monitor Data Drift
Integrate tools such as:
- EvidentlyAI
- WhyLabs
- Google Vertex AI Drift Detection

### 6. Keep Raw Data in GCS and Run Data Validation
Use GCP:
- Tensorflow Data Validation
- Great Expectations
