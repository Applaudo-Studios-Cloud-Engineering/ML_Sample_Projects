from typing import List, Dict

import pandas as pd
import numpy as np
import re

from sklearn.tree import DecisionTreeClassifier


# Feature Engineering functions

def create_dataset(path: str) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from a file retrieved from specified path
    :param path: of the CSV file to use to create the DataFrame
    :return: Pandas DataFrame dataset from CSV file
    """

    return pd.read_csv(path)


def drop_unnecessary_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drops a list of columns from a given pandas DataFrame
    :param df: DataFrame with columns
    :param columns: List of columns to drop from DataFrame
    :return: DataFrame with dropped columns, keeping only the necessary
    """

    return df.drop(columns, axis=1, inplace=True)


def fill_empty_age_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills empty values in Age feature column. Generally speaking, using the average/mean as a filler values is ok.
    Uses standard deviation of ages and the mean to compute some range of filler values for more real-to-life results
    :param df: dataset containing
    :return: DataFrame with filled Age values
    """

    mean = df['Age'].mean()
    std = df['Age'].std()
    total_nulls = df['Age'].isnull().sum()

    randon_age_range = np.random.randint(mean - std, mean + std, size=total_nulls)
    age_feat_slice = df['Age'].copy()
    age_feat_slice[np.isnan(age_feat_slice)] = randon_age_range

    df['Age'] = age_feat_slice
    df['Age'] = df['Age'].astype(int)

    return df


def fill_empty_embarked_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills empty values in Embarked feature column with S.
    :param df:
    :return: DataFrame with nonempty values in Embarked column
    """

    common_val = 'S'

    df['Embarked'] = df['Embarked'].fillna(common_val)

    return df


def fill_empty_fare_values(df: pd.Data) -> pd.DataFrame:
    df['Fare'] = df['Fare'].fillna(0)
    df['Fare'] = df['Fare'].astype(int)

    return df


def encode_embarked_ports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes Embarked feature column values as numbers. Computes like numbers.
    :param df: dataset containing Embarked feature column
    :return: DataFrame with encoded values for Embarked feature
    """

    encoded_ports = {'S': 0, 'C': 1, 'Q': 2}

    df['Embarked'] = df['Embarked'].map(encoded_ports)

    return df


def create_deck_feature(df: pd.DataFrame, drop_cabin: bool=True) -> pd.DataFrame:
    """
    Creates Deck feature which is based on the cabin feature.
    :param df: dataset with Cabin feature
    :return: DataFrame with Deck feature and dropped Cabin feature
    """

    decks = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    df['Cabin'] = df['Cabin'].fillna('U0')
    df['Deck'] = df['Cabin'].apply(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = df['Deck'].map(decks)
    df['Deck'] = df['Deck'].fillna(0)
    df['Deck'] = df['Deck'].astype(int)

    if drop_cabin:
        df.drop(['Cabin'], axis=1)

    return df


def encode_age_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes age ranges from Age feature column as a number.
    :param df: dataset containing
    :return: DataFrame with Age feature
    """

    df['Age'] = df['Age'].astype(int)

    df.loc[df['Age'] <= 11, 'Age Class'] = 0
    df.loc[(df['Age'] > 11 & df['Age'] <= 18), 'Age Class'] = 1
    df.loc[(df['Age'] > 18 & df['Age'] <= 22), 'Age Class'] = 2
    df.loc[(df['Age'] > 22 & df['Age'] <= 27), 'Age Class'] = 3
    df.loc[(df['Age'] > 27 & df['Age'] <= 33), 'Age Class'] = 4
    df.loc[(df['Age'] > 33 & df['Age'] <= 40), 'Age Class'] = 5
    df.loc[(df['Age'] > 40 & df['Age'] <= 66), 'Age Class'] = 6
    df.loc[df['Age'] > 66, 'Age Class'] = 7

    return df


def create_title_feature(df: pd.DataFrame, drop_name: bool = False) -> pd.DataFrame:
    """
    Creates title feature extracted from the name of the passenger.
    :param df: dataset containing name column
    :param drop_name: since the Name feature is of no further use it can be dropped from the DF given a True value
    for this param
    :return: DataFrame with Title feature
    """

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other'
    )
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    if drop_name:
        df.drop(['Name'], axis=1, inplace=True)

    return df


def encode_title_feature(df: pd.DataFrame, titles_dic: Dict[str, int]) -> pd.DataFrame:
    """
    Encodes the title feature as a number. Computers like numbers.
    :param df: dataset containing Title feature
    :param titles_dic: dictionary that contains titles as keys and numbers as their values
    :return: DataFrame with Title feature with number values
    """
    df['Title'] = df['Title'].map(titles_dic)
    df['Title'] = df['Title'].fillna(0)

    return df


def encode_sex(df: pd.DataFrame, sex_dict: Dict[str, int]) -> pd.DataFrame:
    """
    Encodes Sex feature as a number. Computers like numbers.
    :param df: dataset containing Sex feature
    :param sex_dict: dictionary that contains sexes as keys and numbers as values
    :return: DataFrame with Sex feature encoded
    """
    df['Sex'] = df['Sex'].map(sex_dict)

    return df


def encode_fare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes Fare feature as a number.
    :param df: dataset containing Fare feature
    :return: dataset with encoded Fare feature
    """

    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare'] = 3
    df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare'] = 4
    df.loc[df['Fare'] > 250, 'Fare'] = 5
    df['Fare'] = df['Fare'].astype(int)

    return df


def create_age_class_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Age Class feature which is the product of the Age and Pclass.
    :param df: dataset containing Pclass and Age features
    :return: dataset with Age Class feature
    """

    df['Age_Class'] = df['Age'] * df['Pclass']

    return df


def create_relatives_feature(df: pd.DataFrame, drop_features: bool=True) -> pd.DataFrame:
    """
    Create Relatives feature which is the sum of Siblings Spouses [SibSp] and Parents Children [Parch]
    :param df: dataset containing SibSp and Parch features
    :param drop_features: if true drops SibSp and Parch features since they are part of relatives
    :return: dataset with Relatives feature
    """

    df['Relatives'] = df['SibSp'] + df['Parch']

    if drop_features:
        df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    return df


# Model training and testing functions

def split_dataset_for_training(df: pd.DataFrame, label_col: str) -> [pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into X and Y
    :param df: dataset with features and labels
    :param label_col: label column name
    :return: X features dataset and Y label dataset
    """
    X = df.drop([label_col], axis=1)
    Y = df[label_col]

    return X, Y


def create_and_train_decision_tree_model(X_train, Y_train) -> DecisionTreeClassifier:
    """
    Create and trains a Decision Tree Classifier model
    :param X_train: dataset of training features
    :param Y_train: labels
    :return: trained Decision Tree Classifier model
    """

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    return model


def compute_accuracy(model: DecisionTreeClassifier, X_test, Y_test) -> float:
    """
    Evaluates accuracy of the model given X and Y testing datasets
    :param model: trained DecisionTreeClassifier model
    :param X_test: datasets of features with which to test the model
    :param Y_test: dataset of labels with which to test the model (Survived column)
    :return: accuracy of the model
    """

    accuracy = model.score(X_test, Y_test)
    accuracy = round(accuracy * 100, 2)

    return accuracy
