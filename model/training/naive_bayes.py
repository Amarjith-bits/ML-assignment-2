"""Gaussian Naive Bayes: train, evaluate, save."""
from sklearn.naive_bayes import GaussianNB

from common import fit_evaluate_save, load_and_split

LABEL = "Naive Bayes"


def run(split=None):
    split = split or load_and_split()
    clf = GaussianNB()
    return fit_evaluate_save("naive_bayes.joblib", clf, split)


if __name__ == "__main__":
    print(LABEL, run())
