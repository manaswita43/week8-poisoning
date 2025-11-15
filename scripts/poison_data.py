import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/iris.csv")

def poison(df, level):
    poisoned = df.copy()
    n = int(len(df) * level / 100)
    idx = np.random.choice(df.index, n, replace=False)
    poisoned.loc[idx, ["sepal_length", "sepal_width", "petal_length", "petal_width"]] = np.random.uniform(
        0, 10, size=(n, 4)
    )
    return poisoned

levels = [5, 10, 50]
os.makedirs("poisoned_data", exist_ok=True)

for lvl in levels:
    p = poison(df, lvl)
    p.to_csv(f"poisoned_data/iris_poisoned_{lvl}.csv", index=False)
    print(f"Generated poisoned dataset: iris_poisoned_{lvl}.csv")
