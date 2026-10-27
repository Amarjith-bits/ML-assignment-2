"""Support Vector Machine (RBF kernel): train, evaluate, save."""
from sklearn.svm import SVC

from common import RANDOM_STATE, fit_evaluate_save, load_and_split

LABEL = "Support Vector Machine"


def run(split=None):
    split = split or load_and_split()
    # probability=True is needed so predict_proba (and therefore AUC) works
    clf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    return fit_evaluate_save("svm.joblib", clf, split)


if __name__ == "__main__":
    print(LABEL, run())
