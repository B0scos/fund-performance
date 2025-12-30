import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.base import clone



def backward_feature_selection(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    model,
    min_features: int = 3,
    tol: float = 1e-3,
    max_iter: int | None = None,
):
    """
    Backward feature elimination using SILHOUETTE SCORE on TEST data.

    Fit on train → predict on test → silhouette evaluated on test.

    Parameters
    ----------
    X_train : DataFrame or ndarray
    X_test  : DataFrame or ndarray
    model : clustering model (must support fit + predict)
    min_features : stop when features <= this
    tol : min improvement required to continue
    max_iter : optional cap

    Returns
    -------
    remaining_features : list[int]
        feature indices kept
    history : list
        (num_features_remaining, removed_index, silhouette_score_test)
    """

    # Normalize input
    if isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
        train = X_train.values
        test = X_test.values
    else:
        feature_names = list(range(X_train.shape[1]))
        train = X_train
        test = X_test

    remaining = list(range(train.shape[1]))
    max_iter = max_iter or len(remaining)
    history = []

    def compute_test_silhouette(idx):
        mdl = clone(model)

        # fit on train subset
        mdl.fit(train[:, idx])

        # must predict test labels
        labels = mdl.predict(test[:, idx])

        # silhouette requires >=2 clusters present in test
        if len(np.unique(labels)) < 2:
            return -1e9  # punish invalid configs

        return silhouette_score(test[:, idx], labels)

    # baseline score
    best_score = compute_test_silhouette(remaining)
    print(f"Initial TEST silhouette: {best_score:.4f}")

    it = 0
    while len(remaining) > min_features and it < max_iter:
        it += 1
        scores = []

        for f in remaining:
            idx = [i for i in remaining if i != f]
            s = compute_test_silhouette(idx)
            scores.append((f, s))

        # best removal = max silhouette on test
        f_remove, candidate_score = max(scores, key=lambda x: x[1])
        improvement = candidate_score - best_score

        if improvement < tol:
            print("Stopping — no meaningful improvement on TEST")
            break

        remaining.remove(f_remove)
        best_score = candidate_score

        removed_name = feature_names[f_remove]
        history.append((len(remaining), f_remove, best_score))

        print(
            f"Removed: {removed_name} | "
            f"Remaining: {len(remaining)} | "
            f"TEST silhouette: {best_score:.4f}"
        )

    return remaining, history