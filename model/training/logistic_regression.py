"""Logistic Regression: train, evaluate, save."""
from sklearn.linear_model import LogisticRegression

from common import RANDOM_STATE, fit_evaluate_save, load_and_split

LABEL = "Logistic Regression"


def run(split=None):
    split = split or load_and_split()
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return fit_evaluate_save("logistic_regression.joblib", clf, split)


if __name__ == "__main__":
    print(LABEL, run())
