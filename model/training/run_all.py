"""Train all five models on the same split and print the comparison table."""
import pandas as pd

import decision_tree
import knn
import logistic_regression
import naive_bayes
import svm
from common import MODEL_DIR, load_and_split

MODULES = [logistic_regression, decision_tree, knn, naive_bayes, svm]

if __name__ == "__main__":
    split = load_and_split()  # one split shared by every model
    print(f"Train rows: {len(split[0])} | Test rows: {len(split[1])}\n")

    rows = {}
    for mod in MODULES:
        print(f"Training {mod.LABEL} ...")
        rows[mod.LABEL] = mod.run(split)

    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "ML Model Name"
    table.to_csv(MODEL_DIR / "metrics_summary.csv")

    print("\n" + table.to_string())

    # Markdown version, ready to paste into README.md
    header = "| ML Model Name | " + " | ".join(table.columns) + " |"
    divider = "|" + "---|" * (len(table.columns) + 1)
    print("\n" + header + "\n" + divider)
    for name, r in table.iterrows():
        print(f"| {name} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
