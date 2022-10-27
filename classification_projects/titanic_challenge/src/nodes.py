from typing import List

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


def create_deck_feature(df: pd.DataFrame) -> pd.DataFrame:
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

    df.drop(['Cabin'], axis=1)

    return df


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


def test_model(model: DecisionTreeClassifier, X_test, Y_test) -> float:
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

