import pandas as pd
from src.process.pre_processing import pre_processing
from src.utils.load_data import load_data_with_features
from src.models.kmeans import KMeansTrainer
from src.pipelines.train_pipeline import run_training, evaluate
from src.pipelines.train_pipeline import run_training, evaluate, calculate_stability
from src.process.pre_processing import PCA_scalling, scalling, PCA, just_filter
from src.models.gmm import GMMTrainer


if __name__ == "__main__":
    all_results = []
    cluster_results = []
    experiment_summary = []
    look_features = ['mean_return', 'median_return', 'std_return', 'avg_time_drawdown', 'sharpe', 'max_drawdown']


    df_train, df_test, df_val = load_data_with_features()

    models = [KMeansTrainer, GMMTrainer]

    clusters = [2, 3, 4, 5]

    pre_processing_flags = [just_filter, scalling, PCA, PCA_scalling]

    for model in models:
        for cluster in clusters:
            for type_process in pre_processing_flags:
                model_name = model.__name__
                preprocessing_name = str(type_process.__name__) if type_process else "None"
                print(f"Running experiment: Model={model_name}, Clusters={cluster}, Preprocessing={preprocessing_name}")

                (df_train, df_test, df_val), (train_features, test_features, val_features) = run_training(
                    model, df_train, df_test, df_val, pre_processing=type_process, n_clusters=cluster)

                train_results = evaluate(df_train, train_features, look_features=look_features)
                train_results['dataset'] = 'train'
                print("\nTrain Results:")
                print(train_results)

                test_results = evaluate(df_test, test_features, look_features=look_features)
                test_results['dataset'] = 'test'
                print("\nTest Results:")
                print(test_results)

                val_results = evaluate(df_val, val_features, look_features=look_features)
                val_results['dataset'] = 'validation'
                print("\nValidation Results:")
                print(val_results)

                # Add experiment parameters to the results
                for results_df in [train_results, test_results, val_results]:
                    results_df['model'] = model_name
                    results_df['n_clusters'] = cluster
                    results_df['preprocessing'] = preprocessing_name
                    all_results.append(results_df)

                print('#' * 50 + '\n')

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv('experiment_results.csv', index=False)
    print("Results saved to experiment_results.csv")
