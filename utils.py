import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def load_artificial(root='data/', seed=42):
    artificial_train = pd.read_csv(f'{root}artificial_train.data', sep=' ', header=None).dropna(axis=1)
    artificial_train_labels = pd.read_csv(f'{root}artificial_train.labels', sep=' ', header=None).dropna(axis=1)
    artificial_train_labels = (artificial_train_labels[0].values+1)/2
    artificial_test = pd.read_csv(f'{root}artificial_valid.data', sep=' ', header=None).dropna(axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(artificial_train, artificial_train_labels, test_size=0.2, random_state=seed)
    return X_train, X_valid, y_train, y_valid, artificial_test

def scale_minmax(X_train, X_valid, X_test):
    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X_train)
    X_valid = minmax.transform(X_valid)
    X_test = minmax.transform(X_test)
    return X_train, X_valid, X_test


from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score

def get_importance_mutual_information(X_train, y_train):
    importance = mutual_info_classif(X_train, y_train, random_state=42)
    return importance

def get_top_features(X, importances, num_of_features):
    importance_threshold = sorted(importances, reverse=True)[num_of_features]
    return X[:, importances >= importance_threshold]

def get_accuracies(model, model_params_dict, X_train, y_train, X_valid, y_valid):
    model_object = model(**model_params_dict)
    model_object.fit(X_train, y_train)
    y_pred_train = model_object.predict(X_train)
    y_pred_valid = model_object.predict(X_valid)
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
    balanced_accuracy_valid = balanced_accuracy_score(y_valid, y_pred_valid)
    return balanced_accuracy_train, balanced_accuracy_valid

def get_score(accuracy, desired_features, num_of_selected_features):
    return accuracy - 0.01 * max(0, num_of_selected_features / desired_features - 1)

def get_results_df_by_features(importance, model, model_params_dict, X_train, y_train, X_valid, y_valid, desired_features, features_min, features_max, step):
    results = []
    for i in tqdm(list(range(features_min, features_max, step))):
        X_train_filtered, X_valid_filtered = get_top_features(X_train, importance, i), get_top_features(X_valid, importance, i)
        acc_train, acc_valid = get_accuracies(model, model_params_dict, X_train_filtered, y_train, X_valid_filtered, y_valid)
        score = get_score(acc_valid, desired_features, i)
        results.append({'Features': i, 'Train acc': acc_train, 'Valid acc': acc_valid, 'Score': score})
    results_df = pd.DataFrame(results)
    return results_df


import matplotlib.pyplot as plt
import seaborn as sns

def make_plot_by_features(results_df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Features', y='Score', ax=ax, label='Score', color='blue', linestyle='--')
    sns.lineplot(data=results_df, x='Features', y='Train acc', ax=ax, label='Train acc', color='green')
    sns.lineplot(data=results_df, x='Features', y='Valid acc', ax=ax, label='Valid acc', color='red')
    ax.set_title(title)
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Score/Accuracy')
    ax.legend()
    plt.show()
    

        