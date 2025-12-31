from src.process.pre_processing import pre_processing
from src.utils.load_data import load_data_with_features
from src.models.kmeans import KMeansTrainer
from src.pipelines.train_pipeline import run_training, evaluate
from src.process.pre_processing import pre_processing, scalling
from src.models.gmm import GMMTrainer


if __name__ == "__main__":
    look_features = ['mean_return', 'median_return', 'std_return', 'avg_time_drawdown', 'sharpe', 'max_drawdown']


    df_train, df_test, df_val = load_data_with_features()

    models = [KMeansTrainer]

    clusters = [2, 3, 4, 5]

    pre_processing_flags = [False, scalling]

    for model in models:
        for cluster in clusters:
            for type_process in pre_processing_flags:
                print('#' * 10, f'Custer {cluster}', '#' * 10)
                print('\n')
                print('#' * 10, f'model {model.__class__}', '#' * 10)
                print('\n')
                print('#' * 10, f'pre_processing_flags {type_process}', '#' * 10)

                df_train, df_test, df_val = run_training(model, df_train, df_test, df_val, pre_processing=type_process, n_clusters=cluster)

                print(evaluate(df_train, look_features=look_features))
                print('\n test')
                print(evaluate(df_test, look_features=look_features))

                print('#' * 50)

