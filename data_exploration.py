"""
Author: Ron Keinan
February 2023
AI course Final Project
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FILES = {'emails1': 'data/emails1.csv',
         'emails2': 'data/emails2.csv',
         'emails1+2': 'data/emails1+2.csv'}


def load_data(file_path):
    """
    Load a CSV file and drop rows with missing values.
    :param file_path: The path to the CSV file.
    :return: A dataframe with the loaded data and missing values dropped.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df


def plot_data(df):
    """
    Plot the count of spam vs. non-spam emails and save the plot to a file.
    :param df: A dataframe containing email data.
    """
    # plot the data
    sns.countplot(x='spam', data=df)
    plt.title('Spam vs. Not')
    # save the plot to a file in the plots directory
    plt.savefig('spam_vs_not.png')
    plt.clf()


def describe_emails(df):
    """
    create a describe table for the data set and save it to a file
    the describe table should include the length of the email, the number of words in the email,
    and the number of unique words in the email
    :param df: A dataframe containing email data.
    """
    # create a new column that is the length of the email
    df['email_length'] = df['text'].apply(len)
    # create a new column that is the number of words in the email
    df['email_words'] = df['text'].apply(lambda x: len(x.split()))
    # create a new column that is the number of unique words in the email
    df['email_unique_words'] = df['text'].apply(lambda x: len(set(x.split())))
    # create a describe table for the data set and save it to a file
    df.describe().to_excel('describe.xlsx')


def plot_email_length(df):
    """
    create a histogram of the length of the emails
    :param df: A dataframe containing email data.
    """
    # create a histogram of the length of the emails
    df['email_length'].plot.hist(bins=50)
    plt.title('Email Length')
    # save the plot to a file in the plots directory
    plt.savefig('email_length.png')
    plt.clf()


def plot_email_words(df):
    """
    create a histogram of the word number of the emails
    :param df: A dataframe containing email data.
    """
    # create a histogram of the number of words in the emails
    df['email_words'].plot.hist(bins=50)
    plt.title('Email Words')
    # save the plot to a file in the plots directory
    plt.savefig('email_words.png')
    plt.clf()


def plot_email_unique_words(df):
    """
    reate a histogram of the number of unique words in the emails
    :param df: A dataframe containing email data.
    """
    # create a histogram of the number of unique words in the emails
    df['email_unique_words'].plot.hist(bins=50)
    plt.title('Email Unique Words')
    # save the plot to a file in the plots directory
    plt.savefig('email_unique_words.png')
    plt.clf()


def plot_pair(df):
    """
    reate a pair plot of the data
    :param df:
    """
    # create a pai plot of the data
    sns.pairplot(df)
    plt.title('Pair Plot')
    # save the plot to a file in the plots directory
    plt.savefig('pair_plot.png')
    plt.clf()


def plot_pie(df):
    """
    reate a pie plot of the data
    :param df:
    """
    # create a pie plot with distribution of the spam and not spam emails
    df['spam'].value_counts().plot.pie()
    plt.title('Spam vs. Not')
    # add a legend
    plt.legend(['Not Spam', 'Spam'])
    # save the plot to a file in the plots directory
    plt.savefig('spam_vs_not_pie.png')
    plt.clf()


def data_exploration_main():
    """
    run all the plot functions in this file
    """
    for dataset_name, file_path in FILES.items():
        print('Exploring {}'.format(dataset_name))

        # create a directory for the plots with the dataset name
        os.makedirs(f'data_exploration/{dataset_name}', exist_ok=True)

        df = load_data(file_path)

        # change the directory to the new directory
        os.chdir(f'data_exploration/{dataset_name}')

        plot_data(df)
        describe_emails(df)
        plot_email_length(df)
        plot_email_words(df)
        plot_email_unique_words(df)
        plot_pair(df)
        plot_pie(df)

        # change the directory back to the main directory
        os.chdir('../..')


if __name__ == '__main__':
    data_exploration_main()
