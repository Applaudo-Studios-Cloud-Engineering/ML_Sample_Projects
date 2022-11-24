import os

import pandas as pd
import pickle

from nodes import create_dataset, drop_unnecessary_columns, fill_empty_age_values, fill_empty_embarked_values, \
    encode_embarked_ports, create_deck_feature, encode_age_ranges, create_title_feature, encode_title_feature, \
    encode_sex, encode_fare, create_age_class_feature, create_relatives_feature, split_dataset_for_training, \
    create_and_train_decision_tree_model, compute_accuracy, fill_empty_fare_values


def create_preprocessing_pipeline\
                (origin_dataset_path: str, results_dataset_path: str, drop_passenger_id: bool = False):

    df = create_dataset(origin_dataset_path)

    if drop_passenger_id:
        df = drop_unnecessary_columns(df, ['PassengerId'])

    df = fill_empty_age_values(df)

    df = fill_empty_embarked_values(df)

    df = fill_empty_fare_values(df)

    df.to_csv(results_dataset_path)

    # return is optional


def create_feature_engineering_pipeline(origin_dataset_path: str, results_dataset_path: str):

    df = create_dataset(origin_dataset_path)

    df = create_deck_feature(df, False)

    df = create_title_feature(df)

    sexes = {"male": 0, "female": 1}

    df = encode_sex(df, sexes)

    df = create_relatives_feature(df, False)

    df = drop_unnecessary_columns(df, ['Cabin', 'Name', 'Ticket', 'SibSp', 'Parch'])

    df = encode_embarked_ports(df)

    df = encode_fare(df)

    df = encode_age_ranges(df)

    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    df = encode_title_feature(df, titles)

    df = create_age_class_feature(df)

    df.to_csv(results_dataset_path)

    # Return is optional


def create_ml_pipeline(origin_dataset_path: str, model_path: str, model_name: str):

    train_df = create_dataset(origin_dataset_path)

    X_train, Y_train = split_dataset_for_training(train_df, 'Survived')

    model = create_and_train_decision_tree_model(X_train, Y_train)

    training_acc = compute_accuracy(model, X_train, Y_train)

    os.chdir(model_path)

    pickle.dump(model, open(model_name, 'wb'))

    return training_acc
