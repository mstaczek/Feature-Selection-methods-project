import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from boruta import BorutaPy
import numpy as np
np.int = int
np.float = float
np.bool = bool

def load_artificial(root='data/', seed=42):
    artificial_train = pd.read_csv(f'{root}artificial_train.data', sep=' ', header=None).dropna(axis=1)
    artificial_train_labels = pd.read_csv(f'{root}artificial_train.labels', sep=' ', header=None).dropna(axis=1)
    artificial_train_labels = (artificial_train_labels[0].values+1)/2

    artificial_test = pd.read_csv(f'{root}artificial_valid.data', sep=' ', header=None).dropna(axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(artificial_train, artificial_train_labels, test_size=0.2, random_state=seed)
    return X_train, X_valid, y_train, y_valid, artificial_test

def load_sms(root='data/', seed=42):
    sms_train_whole = pd.read_csv(f"{root}sms_train.csv", sep=',')
    sms_train_whole.columns = ['label', 'message']
    sms_train = sms_train_whole['message']
    sms_train_labels = sms_train_whole['label']

    sms_test = pd.read_csv("data/sms_test.csv", sep=',')
    sms_test.columns = ['message']
    

    X_train, X_valid, y_train, y_valid = train_test_split(sms_train, sms_train_labels, test_size=0.2, random_state=seed)

    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    word_to_id_df = pd.DataFrame([{'label':k, 'id':i} for k, i in vectorizer.vocabulary_.items()])
    word_to_id_df = word_to_id_df.sort_values(by='id').reset_index(drop=True)
    def encode_dataset(X, words_to_ids):
        X_transformed = vectorizer.transform(X)
        X_encoded = pd.DataFrame(X_transformed.toarray(), columns=words_to_ids['label'])
        X_encoded.columns.name = ''
        return X_encoded
    X_train_encoded = encode_dataset(X_train, word_to_id_df)
    X_valid_encoded = encode_dataset(X_valid, word_to_id_df)
    sms_test_encoded = encode_dataset(sms_test['message'], word_to_id_df)

    return X_train_encoded, X_valid_encoded, y_train, y_valid, sms_test_encoded

def scale_minmax(X_train, X_valid, X_test):
    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X_train)
    X_valid = minmax.transform(X_valid)
    X_test = minmax.transform(X_test)
    return X_train, X_valid, X_test

def get_importance_mutual_information(X_train, y_train):
    importance = mutual_info_classif(X_train, y_train, random_state=42)
    return importance

def get_importance_chi2(X_train, y_train):
    importance = chi2(X_train, y_train)[0]
    return importance

def get_importance_random_forest(X_train, y_train):
    importance = RandomForestClassifier(max_depth=7, random_state=42, n_estimators=100).fit(X_train, y_train).feature_importances_
    return importance
    
def get_importance_boruta(X_train, y_train):
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
    feat_selector.fit(X_train.values, y_train)
    print(f'Number of selected features: {feat_selector.n_features_}')
    return feat_selector.ranking_

def get_top_features(X, importances, num_of_features):
    importance_threshold = sorted(importances, reverse=True)[num_of_features]
    return X.iloc[:, importances > importance_threshold]

def get_accuracies(model, model_params_dict, X_train, y_train, X_valid, y_valid):
    model_object = model(**model_params_dict)
    model_object.fit(X_train, y_train)
    y_pred_train = model_object.predict(X_train)
    y_pred_valid = model_object.predict(X_valid)
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
    balanced_accuracy_valid = balanced_accuracy_score(y_valid, y_pred_valid)
    return balanced_accuracy_train, balanced_accuracy_valid, model_object

def get_score(accuracy, desired_features, num_of_selected_features):
    return accuracy - 0.01 * max(0, num_of_selected_features / desired_features - 1)

def get_results_df_by_features(importance, model, model_params_dict, X_train, y_train, X_valid, y_valid, desired_features, features_min, features_max, step):
    results = []
    for i in tqdm(list(range(features_min, features_max, step))):
        X_train_filtered, X_valid_filtered = get_top_features(X_train, importance, i), get_top_features(X_valid, importance, i)
        acc_train, acc_valid, model_trained = get_accuracies(model, model_params_dict, X_train_filtered, y_train, X_valid_filtered, y_valid)
        score = get_score(acc_valid, desired_features, i)
        results.append({'Features': i, 'Train acc': acc_train, 'Valid acc': acc_valid, 'Score': score, 'Model': model_trained, 'Features List': X_train_filtered.columns})
    results_df = pd.DataFrame(results)
    return results_df

def get_results_df_by_features_boruta(importance, model, model_params_dict, X_train, y_train, X_valid, y_valid, desired_features, features_min, features_max, step):
    X_train_boruta, X_valid_boruta = X_train.iloc[:, importance == 1], X_valid.iloc[:, importance == 1]
    corr_matrix = X_train_boruta.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    thresholds = np.arange(0.01, 1.02, 0.05)
    results = []
    for threshold in tqdm(thresholds):
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_train_filtered, X_valid_filtered = X_train_boruta.drop(to_drop, axis=1), X_valid_boruta.drop(to_drop, axis=1)
        i = X_train_filtered.shape[1]
        if i < features_min or i > features_max:
            continue
        acc_train, acc_valid, model_trained = get_accuracies(model, model_params_dict, X_train_filtered, y_train, X_valid_filtered, y_valid)
        score = get_score(acc_valid, desired_features, i)
        results.append({'Features': i, 'Train acc': acc_train, 'Valid acc': acc_valid, 'Score': score, 'Model': model_trained, 'Features List': X_train_filtered.columns})
    results_df = pd.DataFrame(results)
    return results_df

def make_plot_by_features(results_df, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.lineplot(data=results_df, x='Features', y='Score', ax=ax, label='Score', color='blue', linestyle='--')
    sns.lineplot(data=results_df, x='Features', y='Train acc', ax=ax, label='Train acc', color='green')
    sns.lineplot(data=results_df, x='Features', y='Valid acc', ax=ax, label='Valid acc', color='red')
    ax.set_title(title)
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Score/Accuracy')
    ax.legend()
    plt.show()

def plot_first_k_percent_importances(importances, title, k):
    sorted_importances = sorted(importances, reverse=True)[:int(len(importances) * k/100)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(1, 1+len(sorted_importances)), sorted_importances)
    ax.set_title(title)
    ax.set_xlabel('Feature number')
    ax.set_ylabel('Importance')
    plt.show()
    
def print_results_for_specific_features(results_df, min_features, max_features, method_name):
    results_filtered = results_df[(results_df['Features'] >= min_features) & (results_df['Features'] <= max_features)]
    results_filtered = results_filtered.sort_values(by='Score', ascending=False)
    print(results_filtered.head(1)[['Features', 'Score', 'Train acc', 'Valid acc']])
    dict_result = results_filtered.head(1).to_dict('records')[0] 
    dict_result['Method'] = method_name
    dict_result = {key: dict_result[key] for key in ['Method'] + list(dict_result.keys())[:-1]}
    return dict_result
    
def plot_summary(final_results, dataset_name):
    final_results_df = pd.DataFrame(final_results)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(x='Method', y='Valid acc', data=final_results_df, ax=ax[0])
    ax[0].set_title(f'Validation Accuracy on {dataset_name} dataset')

    sns.scatterplot(x='Features', y='Valid acc', hue='Method', data=final_results_df, ax=ax[1], s=200)
    ax[1].set_title('Validation accuracy vs number of features')
    fig.suptitle(f'Comparison of feature selection methods on {dataset_name} dataset', fontsize=16, y=1.03)
    plt.show()

    final_results_df[['Method', 'Features', 'Train acc', 'Valid acc', 'Score']]

def save_predictions(final_results, method, dataset_name, X_test):
    final_results_df = pd.DataFrame(final_results)
    row = final_results_df[final_results_df['Method'] == method]
    features_list = row['Features List'].values[0].values
    X_test_filtered = X_test[features_list]

    # save posterior probabilities to file
    y_pred_proba = row['Model'].values[0].predict_proba(X_test_filtered)[:,1]
    df_y_pred_proba = pd.DataFrame(y_pred_proba)
    df_y_pred_proba.to_csv(f'MACPAW_{dataset_name}_prediction.txt', index=False, header=False)
    print('Predictions saved to file')

    # save feature names to file
    with open(f'MACPAW_{dataset_name}_features.txt', 'w') as f:
        for feature in features_list:
            f.write(str(feature) + '\n')
    print(f'Features saved to file (there are {len(features_list)} features):')
    print(features_list)