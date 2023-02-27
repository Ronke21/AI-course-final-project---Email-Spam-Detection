# read files from dir data - files emails1_dev.csv, emails2_dev.csv
import pandas as pd

import train_models


def load_data(file_path):
    """
    Load a CSV file and drop rows with missing values.
    :param file_path: The path to the CSV file.
    :return: A dataframe with the loaded data and missing values dropped.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df


def create_dev_df():
    """
    create a dataframe that contains the emails from the dev set
    :return: a dataframe that contains the emails from the dev set
    """
    emails1_dev = load_data('data/emails1_dev.csv')
    emails2_dev = load_data('data/emails2_dev.csv')
    # create one united df with a column that indicates the source of the email
    emails1_dev['source'] = 'emails1'
    emails2_dev['source'] = 'emails2'
    emails_dev = pd.concat([emails1_dev, emails2_dev])
    return emails_dev


def classify_dev_emails():
    """
    classify the emails from the dev set using the best models that were trained in train_models.py
    :return: a dataframe that contains the emails from the dev set with the classification of each email
    """
    emails_dev = create_dev_df()
    emails1_classification = []
    emails2_classification = []
    emails12_classification = []
    classifiers = ['Decision Tree', 'KNN', 'Logistic Regression', 'Multi-layer Perceptron', 'Multinomial Naive Bayes',
                   'Random Forest', 'Support Vector Machine']
    datasizes = [100, 500, 1000]
    for index, row in emails_dev.iterrows():
        email_text = row['text']
        predictions_emails1 = []
        predictions_emails2 = []
        predictions_emails12 = []
        for classifier in classifiers:
            for datasize in datasizes:
                try:
                    predictions_emails1.append(train_models.classify_email(email_text, classifier, "emails1", datasize)[0])
                    predictions_emails2.append(train_models.classify_email(email_text, classifier, "emails2", datasize)[0])
                    predictions_emails12.append(
                        train_models.classify_email(email_text, classifier, "emails1+2", datasize)[0])
                except:
                    print(f"error in {classifier} {datasize} {row['source']}")
        # choose the most common prediction
        emails1_classification.append(max(set(predictions_emails1), key=predictions_emails1.count))
        emails2_classification.append(max(set(predictions_emails2), key=predictions_emails2.count))
        emails12_classification.append(max(set(predictions_emails12), key=predictions_emails12.count))
    emails_dev['emails1_classification'] = emails1_classification
    emails_dev['emails2_classification'] = emails2_classification
    emails_dev['emails12_classification'] = emails12_classification

    return emails_dev


def calc_dev_accuracy(dev_df):
    """
    calculate the accuracy of the classification of the emails from the dev set
    :param dev_df: a dataframe that contains the emails from the dev set with the classification of each email
    :return: the accuracy of the classification of the emails from the dev set by each corpus models
    """
    # calc accuracy
    emails1_accuracy = 0
    emails2_accuracy = 0
    emails12_accuracy = 0
    for index, row in dev_df.iterrows():
        if row['emails1_classification'] == row['spam']:
            emails1_accuracy += 1
        if row['emails2_classification'] == row['spam']:
            emails2_accuracy += 1
        if row['emails12_classification'] == row['spam']:
            emails12_accuracy += 1
    emails1_accuracy = emails1_accuracy / len(dev_df)
    emails2_accuracy = emails2_accuracy / len(dev_df)
    emails12_accuracy = emails12_accuracy / len(dev_df)
    print('emails1 accuracy: ', emails1_accuracy)
    print('emails2 accuracy: ', emails2_accuracy)
    print('emails12 accuracy: ', emails12_accuracy)
    return emails1_accuracy, emails2_accuracy, emails12_accuracy


def verification_main():
    """
    main function of verification.py
    calassify the emails from the dev set using the best models that were trained in train_models.py
    and calculate the accuracy of the classification of the emails from the dev set
    """
    emails_dev = classify_dev_emails()
    emails_dev.to_csv('emails_dev_classification.csv', index=False)
    emails1_accuracy, emails2_accuracy, emails12_accuracy = calc_dev_accuracy(emails_dev)
    print('emails1 accuracy: ', emails1_accuracy)
    print('emails2 accuracy: ', emails2_accuracy)
    print('emails12 accuracy: ', emails12_accuracy)


if __name__ == '__main__':
    verification_main()
