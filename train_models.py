"""
Author: Ron Keinan
February 2023
AI course Final Project
"""
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data_preperation

FILES = {'emails1': 'data/emails1.csv',
         'emails2': 'data/emails2.csv',
         'emails1+2': 'data/emails1+2.csv'}

PREPROCESS_LIST = ['punctuation', 'lowercase', 'subject']

CLASSIFIERS = {'KNN': KNeighborsClassifier(),
               'Logistic Regression': LogisticRegression(),
               'Multinomial Naive Bayes': MultinomialNB(),
               'Support Vector Machine': SVC(),
               'Decision Tree': DecisionTreeClassifier(),
               'Random Forest': RandomForestClassifier(n_jobs=-1),
               'Multi-layer Perceptron': MLPClassifier(max_iter=1000)}

ANALYZER_SIZE_LIST = [['word', 1],
                      ['word', 2],
                      ['word', 3]]

DATA_SIZES = [100, 500, 1000]

TERM_NUMBER_LIST = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

MIN_NUMBER_OF_TEXTS = 3

ROUND_DIGITS = 8


def load_data(prep_list, file_name, data_size):
    """
    func that receive list of preprocess and load data from data_preperation.py
    :param prep_list: list of preprocess methods
    :param file_name: name of file to load
    :param data_size: number of data to load
    :return: emails, labels
    """
    df = data_preperation.data_preperation_main(prep_list, file_name, data_size)
    return df


def create_feature_matrix(emails, labels, term_number, analyzer, ngram_range):
    """
    func that create feature matrix from emails and labels using TfidfVectorizer
    :param emails: list of emails
    :param labels: list of labels
    :param term_number: number of terms to use
    :param analyzer: analyzer to use - word or char
    :param ngram_range: number of ngrams to use
    :return: feature matrix
    """
    vectorizer = TfidfVectorizer(max_features=term_number, min_df=MIN_NUMBER_OF_TEXTS, analyzer=analyzer,
                                 ngram_range=(ngram_range, ngram_range))
    response = vectorizer.fit_transform(emails)
    X_train, X_test, y_train, y_test = train_test_split(response, labels, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer


def create_feature_matrix_list(emails, labels):
    """
    func that create list of feature matrix for all the options of analyzer and ngram_range
    :param emails: list of emails
    :param labels: list of labels of spam or not spam
    :return: list of feature matrix
    """
    feature_matrix_dict = {}
    for term_number in TERM_NUMBER_LIST:
        for analyzer, ngram_range in ANALYZER_SIZE_LIST:
            feature_matrix_dict[f'{analyzer}_{ngram_range}_{term_number}'] = create_feature_matrix(emails, labels,
                                                                                                   term_number,
                                                                                                   analyzer,
                                                                                                   ngram_range)
    return feature_matrix_dict


def train_and_test_model(full_model_name, classifier, X_train, y_train, X_test, y_test):
    """
    func that train and test a single classifying model
    :param full_model_name: name of the model
    :param classifier: classifier type to use
    :param X_train: feature matrix of train data
    :param y_train: labels of train data
    :param X_test: feature matrix of test data
    :param y_test: labels of test data
    :return: accuracy, precision, recall, f1 score of the model, and the model itself
    """
    print(f'Training {full_model_name}...')
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = round(accuracy_score(y_test, predictions), ROUND_DIGITS)
    precision = round(precision_score(y_test, predictions, zero_division=0), ROUND_DIGITS)
    recall = round(recall_score(y_test, predictions, zero_division=0), ROUND_DIGITS)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = round(2 * (precision * recall) / (precision + recall), ROUND_DIGITS)

    return accuracy, precision, recall, f1_score, classifier


def train_and_test_all_models(feature_matrix_dict, file_name, data_size):
    """
    func that train and test all the models for all the feature matrix. save the results in a csv file
    :param feature_matrix_dict: dict of feature matrix
    :param file_name: name of file to load
    :param data_size: number of data to load
    :return: df of results of all the models
    """
    results_list = []
    for model_name, classifier in CLASSIFIERS.items():
        for feature_matrix_name, [X_train, X_test, y_train, y_test, vectorizer] in feature_matrix_dict.items():
            full_model_name = f'{model_name}_{feature_matrix_name}'
            accuracy, precision, recall, f1_score, classifier = train_and_test_model(full_model_name, classifier,
                                                                                     X_train,
                                                                                     y_train, X_test, y_test)

            # save the model to dir of trained model in dir with file_
            pickle.dump(classifier,
                        open(f'trained_models/{file_name}_datasize_{data_size}/{full_model_name}.pkl', 'wb'))

            results_list.append([model_name, feature_matrix_name, accuracy, precision, recall, f1_score])
            print(
                f'{full_model_name} with dataset {file_name} and data size {data_size} done with accuracy: {accuracy}, '
                f'precision: {precision}, recall: {recall}, '
                f'f1_score: {f1_score}')
    results_df = pd.DataFrame(results_list, columns=['model', 'feature_matrix', 'accuracy', 'precision', 'recall',
                                                     'f1_score'])
    return results_df


def train_models_main(file_name, data_size):
    """
    main func that load data, create feature matrix, train and test all the models, and save the results in a csv file
    :param file_name: name of file to load
    :param data_size: number of data to load
    :return: df of results of all the models
    """
    # load data
    df = load_data(PREPROCESS_LIST, file_name, data_size)
    # shuffle the DataFrame rows
    df = df.sample(frac=1)
    emails = df['text'].tolist()
    labels = df['spam'].tolist()

    # create feature matrix
    feature_matrix_dict = create_feature_matrix_list(emails, labels)

    # save feature matrix dict
    pickle.dump(feature_matrix_dict,
                open(f'trained_models/{file_name}_datasize_{data_size}/feature_matrix_dict.pkl', 'wb'))

    # train and test all models
    results_df = train_and_test_all_models(feature_matrix_dict, file_name, data_size)
    results_df.to_csv(f'trained_models/{file_name}_datasize_{data_size}/models_results.csv', index=False)


def extract_best_models(file_name, data_size):
    """
    func that extract the best models from the results csv file
    :param file_name: name of file to load
    :param data_size: number of data to load
    :return: df of the best models
    """
    # if there is no results.csv file, run train_models_main() func
    if not os.path.exists(f'trained_models/{file_name}_datasize_{data_size}/models_results.csv'):
        train_models_main(file_name, data_size)
    # create df with best models, one from each classifier. the df shoulf contain the model name,
    # the feature matrix name, the accuracy.
    results_df = pd.read_csv(f'trained_models/{file_name}_datasize_{data_size}/models_results.csv')
    best_models_df = results_df.groupby('model').apply(lambda x: x.nlargest(1, 'accuracy')).reset_index(drop=True)
    best_models_df.to_csv(f'trained_models/{file_name}_datasize_{data_size}/best_models.csv', index=False)
    # create a dictionary with the best models themselves
    best_models_dict = {}
    for model_name, feature_matrix_name in zip(best_models_df['model'], best_models_df['feature_matrix']):
        best_models_dict[model_name] = pickle.load(
            open(f'trained_models/{file_name}_datasize_{data_size}/{model_name}_{feature_matrix_name}.pkl', 'rb'))
    return best_models_dict, best_models_df


def extract_feature_matrices(file_name, data_size):
    """
    func that extract the feature matrices from the feature matrix dict
    :param file_name: name of file to load
    :param data_size: number of data to load
    :return: dict of feature matrices
    """
    # if there is no feature_matrix_dict.pkl file, run train_models_main() func
    if not os.path.exists(f'trained_models/{file_name}_datasize_{data_size}/feature_matrix_dict.pkl'):
        # load data
        df = load_data(PREPROCESS_LIST, file_name, data_size)
        # shuffle the DataFrame rows
        df = df.sample(frac=1)
        emails = df['text'].tolist()
        labels = df['spam'].tolist()
        # create feature matrix
        create_feature_matrix_list(emails, labels)
    # create a dictionary with the feature matrices
    feature_matrix_dict = pickle.load(
        open(f'trained_models/{file_name}_datasize_{data_size}/feature_matrix_dict.pkl', 'rb'))
    return feature_matrix_dict


def classify_email(email, chosen_model, file_name, data_size):
    """
    func that classify a single email as spam or not spam
    :param email: the email to classify
    :param chosen_model: the model to use for classification
    :param file_name: name of file to load
    :param data_size: number of data to load
    :return: the classification result
    """
    # check if there is a dir for the file_name and data_size, if not, return -1
    if not os.path.exists(f'trained_models/{file_name}_datasize_{data_size}/best_models.csv'):
        return -1, "Error"

    email = data_preperation.preprocess_email(PREPROCESS_LIST, email)

    best_models_dict, best_models_df = extract_best_models(file_name, data_size)
    feature_matrices_dict = extract_feature_matrices(file_name, data_size)
    # get the feature matrix name of the chosen model
    line = best_models_df[best_models_df['model'] == chosen_model]
    feature_matrix = list(line['feature_matrix'])[-1]
    vectorizer = feature_matrices_dict[feature_matrix][-1]
    response = vectorizer.transform([email])
    best_model = best_models_dict[chosen_model]
    prediction = best_model.predict(response)
    return list(prediction)[0], email


def main():
    """
    main func, runs the model training for every file and data size
    """
    # if there is no trained_models dir, create it
    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')
    for data_size in DATA_SIZES:
        for file_name in FILES:
            # if there is no dir with the file name, create it
            if not os.path.exists(f'trained_models/{file_name}_datasize_{data_size}/best_models.csv'):
                os.mkdir(f'trained_models/{file_name}_datasize_{data_size}')
            extract_best_models(file_name, data_size)


if __name__ == '__main__':
    main()
