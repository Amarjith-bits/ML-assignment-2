"""Decision Tree: train, evaluate, save."""
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, fit_evaluate_save, load_and_split

LABEL = "Decision Tree"


def run(split=None):
    split = split or load_and_split()
    # depth and leaf size are capped so the tree does not memorise the training set
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE)
    return fit_evaluate_save("decision_tree.joblib", clf, split)


if __name__ == "__main__":
    print(LABEL, run())
