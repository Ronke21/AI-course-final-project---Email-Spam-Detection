"""
Author: Ron Keinan
February 2023
AI course Final Project
"""
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

FILES = {'emails1': 'data/emails1.csv',
         'emails2': 'data/emails2.csv',
         'emails1+2': 'data/emails1+2.csv'}


def load_data(file_name):
    """
    load data from csv file and return it as a dataframe
    :param file_name: the name of the file to load
    :return: dataframe
    """
    file_path = FILES[str(file_name)]
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df


def lower_case(emails_list):
    """
    convert all letters in the emails to lower case
    :param emails_list: the list of emails to convert
    :return: the list of emails after converting
    """
    cleaned_emails = []
    for email in emails_list:
        cleaned_email = email.lower()
        cleaned_emails.append(cleaned_email)
    return cleaned_emails


def remove_punctuation(emails_list):
    """
    remove punctuation from the emails
    :param emails_list: the list of emails to convert
    :return: the list of emails after converting
    """
    punctuation_marks = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    cleaned_emails = []
    for email in emails_list:
        cleaned_email = ''
        for char in email:
            if char not in punctuation_marks:
                cleaned_email += char
        # remove extra spaces
        cleaned_email = re.sub(' +', ' ', cleaned_email)
        cleaned_emails.append(cleaned_email)
    return cleaned_emails


def remove_stop_words(emails_list):
    """
    remove stop words from the emails
    stop words are words that are not important for the meaning of the sentence
    :param emails_list: the list of emails to convert
    :return: the list of emails after converting
    """
    stop_words = set(stopwords.words('english'))
    cleaned_emails = []
    for email in emails_list:
        cleaned_email = [word for word in email.split() if word not in stop_words]
        cleaned_emails.append(' '.join(cleaned_email))
    return cleaned_emails


def stem_words(emails_list):
    """
    stem words in the emails
    stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form
    :param emails_list: the list of emails to convert
    :return: the list of emails after converting
    """
    stemmer = PorterStemmer()
    cleaned_emails = []
    for email in emails_list:
        cleaned_email = [stemmer.stem(word) for word in email.split()]
        cleaned_emails.append(' '.join(cleaned_email))
    return cleaned_emails


def lemmatize_words(emails_list):
    """
    lemmatize words in the emails
    lemmatization is the process of grouping together the different inflected forms of a word so they can be analysed as
     a single item
    :param emails_list: the list of emails to convert
    :return: the list of emails after converting
    """
    lemmatizer = WordNetLemmatizer()
    cleaned_emails = []
    for email in emails_list:
        cleaned_email = [lemmatizer.lemmatize(word) for word in email.split()]
        cleaned_emails.append(' '.join(cleaned_email))
    return cleaned_emails


def remove_subject(emails_list):
    """
    remove the word ;subject from the beginning of each email
    :param emails_list: the list of emails to convert
    :return: the list of emails after converting
    """
    cleaned_emails = []
    for email in emails_list:
        # if mail starts with "subject: no subject", remove it
        if email.startswith('subject no subject'):
            cleaned_email = email[18:]
        else:
            cleaned_email = email
        if cleaned_email.startswith('subject '):
            cleaned_email = cleaned_email[8:]
        cleaned_emails.append(cleaned_email)
    return cleaned_emails


def data_preprocess_texts(prep_list, df):
    """
    preprocess the emails in the dataframe
    :param prep_list: the list of preprocessing methods to apply
    :param df: the dataframe to preprocess
    :return: the dataframe after preprocessing
    """
    emails_list = df['text'].tolist()
    if 'lowercase' in prep_list:
        emails_list = lower_case(emails_list)
    if 'punctuation' in prep_list:
        emails_list = remove_punctuation(emails_list)
    if 'subject' in prep_list:
        emails_list = remove_subject(emails_list)
    if 'stopwords' in prep_list:
        emails_list = remove_stop_words(emails_list)
    if 'stemming' in prep_list:
        emails_list = stem_words(emails_list)
    if 'lemmatization' in prep_list:
        emails_list = lemmatize_words(emails_list)

    # create df with cleaned emails and spam column
    cleaned_df = pd.DataFrame({'text': emails_list, 'spam': df['spam']})
    return cleaned_df


def preprocess_email(prep_list, email):
    """
    receive a single email and preprocess it
    :param prep_list: the list of preprocessing methods to apply
    :param email: the email to preprocess
    :return: the email after preprocessing
    """
    if 'lowercase' in prep_list:
        email = email.lower()
    if 'subject' in prep_list:
        email = email.replace('subject', '')
    if 'punctuation' in prep_list:
        email = re.sub(r'[^\w\s]', '', email)
    if 'stop_words' in prep_list:
        for word in stopwords.words('english'):
            token = ' ' + word + ' '
            email = email.replace(token, ' ')
    if 'stemming' in prep_list:
        email = email.stem()
    if 'lemmatization' in prep_list:
        email = email.lemmatize()

    return email


def balance_data(df):
    """
    balance the data by removing rows from the majority class
    :param df: the dataframe to balance
    :return: the balanced dataframe
    """
    # number of spam emails
    spam_count = df['spam'].value_counts()[1]
    # number of not spam emails
    not_spam_count = df['spam'].value_counts()[0]
    # number of emails to take from each class
    num_of_emails = min(spam_count, not_spam_count)
    df = (df.groupby('spam', as_index=False)
          .apply(lambda x: x.sample(n=num_of_emails))
          .reset_index(drop=True))
    return df


def sample_data(num, df):
    """
    sample the data by taking a random sample of the data
    :param num: the number of rows to take
    :param df: the dataframe to sample
    :return: the sampled dataframe
    """
    df = (df.groupby('spam', as_index=False)
          .apply(lambda x: x.sample(n=num, replace=True))
          .reset_index(drop=True))
    return df


def data_preperation_main(prep_list, file_name, data_size):
    """
    the main function of the data preperation module
    :param prep_list: the list of preprocessing methods to apply
    :param file_name: the name of the file to read
    :param data_size: the size of the data to use
    :return: the preprocessed dataframe
    """
    df = load_data(file_name)
    df = data_preprocess_texts(prep_list, df)
    df = sample_data(data_size, df)
    return df
