import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.stats import wasserstein_distance

def evaluate(df, features_df, look_features):
    """
    Combines cluster feature means, value counts, and silhouette score
    into a single DataFrame.
    """
    means = df.groupby("pred")[look_features].mean()
    counts = df["pred"].value_counts().rename("cluster_size")
    results_df = pd.concat([means, counts], axis=1)

    # Calculate silhouette score, requires at least 2 clusters
    if df['pred'].nunique() > 1:
        score = silhouette_score(features_df, df['pred'])
    else:
        score = np.nan  # Not defined for a single cluster
    results_df['silhouette_score'] = score
    return results_df.reset_index().rename(columns={'index': 'pred'})

def calculate_stability(df1_eval: pd.DataFrame, df2_eval: pd.DataFrame, metrics_to_compare: list) -> dict:
    """
    Calculates the statistical similarity (stability) between the evaluation
    results of two clustered datasets using the Wasserstein distance.

    A lower distance score indicates more stability/similarity.

    Parameters
    ----------
    df1_eval : pd.DataFrame
        The evaluation DataFrame for the first dataset (e.g., train set).
    df2_eval : pd.DataFrame
        The evaluation DataFrame for the second dataset (e.g., test set).
    metrics_to_compare : list
        A list of column names (metrics) to compare for stability.

    Returns
    -------
    dict
        A dictionary where keys are the metrics and values are their
        Wasserstein distance.
    """
    stability_scores = {}

    for feature in metrics_to_compare:
        # Sort values to compare distributions, which works even if the
        # number of clusters found is different.
        vals1 = df1_eval[feature].sort_values().values
        vals2 = df2_eval[feature].sort_values().values

        stability_scores[f'dist_{feature}'] = wasserstein_distance(vals1, vals2)

    return stability_scores


def run_training(model_cls,
                df_train : pd.DataFrame,
                df_test : pd.DataFrame,
                df_val : pd.DataFrame, 
                pre_processing = False, 
                **kwargs):

    train_features, test_features, val_features = df_train.copy(), df_test.copy(), df_val.copy()
    if pre_processing:
        train_features, test_features, val_features = pre_processing(train_features, test_features, val_features)

    model = model_cls(**kwargs)
    train_pred, test_pred, val_pred = model.train_and_predict(
        train_features, test_features, val_features
    )

    df_train["pred"] = train_pred
    df_test["pred"] = test_pred
    df_val["pred"] = val_pred

    return (df_train, df_test, df_val), (train_features, test_features, val_features)
