import pandas as pd
from scipy.stats import wasserstein_distance


df = pd.read_csv('experiment_results.csv')

# --- Part 1: Find the best model based on validation silhouette score ---
df = df[df['dataset'] == 'validation']

max_score = df['silhouette_score'].max()
best_row = df[df['silhouette_score'] == max_score]

print("--- Best Model Configuration ---")
print(best_row[['pred', 'mean_return', 'std_return',
        'sharpe', 'max_drawdown', 'cluster_size',
       'silhouette_score', 'model', 'n_clusters', 'preprocessing']])
print("\n" + "="*50 + "\n")


# --- Part 2: Analyze the stability of the best model ---

# Isolate all data for the best model configuration
best_model_config = best_row.iloc[0]
full_df = pd.read_csv('experiment_results.csv')

best_model_df = full_df[
    (full_df['model'] == best_model_config['model']) &
    (full_df['n_clusters'] == best_model_config['n_clusters']) &
    (full_df['preprocessing'] == best_model_config['preprocessing'])
]

train_metrics = best_model_df[best_model_df['dataset'] == 'train']
test_metrics = best_model_df[best_model_df['dataset'] == 'test']
val_metrics = best_model_df[best_model_df['dataset'] == 'validation']

look_features = ['mean_return', 'median_return', 'std_return', 'avg_time_drawdown', 'sharpe', 'max_drawdown', 'cluster_size']
stability_scores = []

for feature in look_features:
    train_vals = train_metrics[feature].sort_values().values
    test_vals = test_metrics[feature].sort_values().values
    val_vals = val_metrics[feature].sort_values().values

    # Calculate Wasserstein distance (Earth Mover's Distance)
    dist_train_test = wasserstein_distance(train_vals, test_vals)
    dist_train_val = wasserstein_distance(train_vals, val_vals)
    stability_scores.append({'feature': feature, 'train_vs_test_dist': dist_train_test, 'train_vs_val_dist': dist_train_val})

stability_df = pd.DataFrame(stability_scores)
print("--- Best Model Stability (Wasserstein Distance) ---")
print("Lower scores are better, indicating more similar distributions between datasets.")
print(stability_df)