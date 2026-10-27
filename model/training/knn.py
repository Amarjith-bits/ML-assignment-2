"""K-Nearest Neighbors: train, evaluate, save."""
from sklearn.neighbors import KNeighborsClassifier

from common import fit_evaluate_save, load_and_split

LABEL = "KNN"


def run(split=None):
    split = split or load_and_split()
    clf = KNeighborsClassifier(n_neighbors=7)
    return fit_evaluate_save("knn.joblib", clf, split)


if __name__ == "__main__":
    print(LABEL, run())
