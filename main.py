from src.pipelines.data_pipeline import data_pipeline
from src.pipelines.train_pipeline import pre_processing
from src.pipelines.feature_selection_pipeline import backward_feature_selection
from sklearn.cluster import KMeans

# if __name__ == "__main__":
#     # preprocessing(n_components=5)
#     train_pca, test_pca, val_pca = pre_processing(n_components=4)
    
#     model = KMeans(n_clusters=5, random_state=0, n_init="auto")

#     remaining, history = backward_feature_selection(train_pca, test_pca, model)

#     print(remaining, history)
    

from src.pipelines.data_pipeline import data_pipeline
from src.pipelines.train_pipeline import pre_processing
from src.pipelines.feature_selection_pipeline import backward_feature_selection
from sklearn.cluster import KMeans
import pandas as pd

if __name__ == "__main__":

    train_pca, test_pca, val_pca = pre_processing(n_components=4)

    model = KMeans(n_clusters=3, random_state=0, n_init="auto")

    # backward selection → returns remaining features
    # remaining, history = backward_feature_selection(train_pca, test_pca, model, min_features=10)

    # print("Remaining features:", remaining)
    # print("History:", history)

    # --- Ensure we only use the selected features ---
    X_train = train_pca #[remaining]
    X_test  = test_pca #[remaining]
    X_val   = val_pca #[remaining]

    # --- Fit and predict ---
    model.fit(X_train)

    train_clusters = model.predict(X_train)
    test_clusters  = model.predict(X_test)
    val_clusters   = model.predict(X_val)

    # --- Merge back ---
    train_pca["cluster"] = train_clusters
    test_pca["cluster"]  = test_clusters
    val_pca["cluster"]   = val_clusters

    # --- Groupby summaries ---
    def summarize(df, name):
        print(f"\n==== {name} GROUPBY SUMMARY ====")

        grouped = df.groupby("cluster").mean(numeric_only=True)

        # If you ONLY want these columns and they actually exist:
        cols = ['sharpe','avg_drawdown', 'avg_time_drawdown', 'max_time_drawdown']
        available = [c for c in cols if c in grouped.columns]

        print(grouped[available])

        print("\nCounts per cluster:")
        print(df["cluster"].value_counts())

    summarize(train_pca, "TRAIN")
    summarize(test_pca, "TEST")
    summarize(val_pca, "VAL")
