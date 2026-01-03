import pandas as pd

def evaluate(df, look_features):
    """Combines cluster feature means and value counts into a single DataFrame."""
    means = df.groupby("pred")[look_features].mean()
    counts = df["pred"].value_counts().rename("cluster_size")
    results_df = pd.concat([means, counts], axis=1)
    return results_df



def run_training(model_cls,
                df_train : pd.DataFrame,
                df_test : pd.DataFrame,
                df_val : pd.DataFrame, 
                pre_processing = False, 
                **kwargs):

    train, test, val = df_train, df_test, df_val
    if pre_processing:
        train, test, val = pre_processing(df_train, df_test, df_val)

    model = model_cls(**kwargs)
    train_pred, test_pred, val_pred = model.train_and_predict(
        train, test, val
    )

    df_train["pred"] = train_pred
    df_test["pred"] = test_pred
    df_val["pred"] = val_pred

    return df_train, df_test, df_val
