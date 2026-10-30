"""Naive Bayes: train, evaluate, save.

Variants (all use the same train/test split):
  "gaussian_numeric"  - GaussianNB on the numeric columns only
  "multinomial"       - MultinomialNB on min-max scaled numerics + one-hot categories (all 17 features)
  "multinomial_equal" - same as "multinomial" but with equal class priors (fit_prior=False),
                        which plays the same role for Naive Bayes that class_weight does elsewhere
"""
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from common import CATEGORICAL_COLS, NUMERIC_COLS, fit_evaluate_save, load_and_split

LABEL = "Naive Bayes"
VARIANT = "multinomial"  # the one that gets saved as naive_bayes.joblib


def make_variant(variant):
    if variant == "gaussian_numeric":
        # Gaussian NB expects continuous features, so drop the one-hot columns.
        prep = ColumnTransformer([("num", "passthrough", NUMERIC_COLS)])
        return prep, GaussianNB()

    if variant in ("multinomial", "multinomial_equal"):
        # MultinomialNB cannot take negative numbers, so use MinMaxScaler
        # (clip=True keeps unseen test values inside 0-1) instead of StandardScaler.
        prep = ColumnTransformer([
            ("num", MinMaxScaler(clip=True), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ])
        return prep, MultinomialNB(alpha=1.0, fit_prior=(variant == "multinomial"))

    raise ValueError(f"unknown variant: {variant}")


def run(split=None, variant=VARIANT, filename="naive_bayes.joblib"):
    split = split or load_and_split()
    prep, clf = make_variant(variant)
    return fit_evaluate_save(filename, clf, split, preprocessor=prep)


if __name__ == "__main__":
    split = load_and_split()
    for v in ("gaussian_numeric", "multinomial", "multinomial_equal"):
        # try-out runs go to temp files so they don't overwrite the real model
        print(v, run(split, variant=v, filename=f"_try_{v}.joblib"))