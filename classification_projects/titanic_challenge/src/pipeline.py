import pandas as pd

from .nodes import create_dataset, drop_unnecessary_columns, fill_empty_age_values, fill_empty_embarked_values, \
    encode_embarked_ports, create_deck_feature, encode_age_ranges, create_title_feature, encode_title_feature, \
    encode_sex, encode_fare, create_age_class_feature, create_relatives_feature, split_dataset_for_training, \
    create_and_train_decision_tree_model, compute_accuracy, fill_empty_fare_values


def create_preprocessing_pipeline(dataset_path: str, drop_passenger_id: bool) -> pd.DataFrame:
    df = create_dataset(dataset_path)

    if drop_passenger_id:
        df = drop_unnecessary_columns(df, ['PassengerId'])

    df = fill_empty_age_values(df)

    df = fill_empty_embarked_values(df)

    df = fill_empty_fare_values(df)

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

    return df


def create_ml_pipeline(train_df: pd.DataFrame):
    X_train, Y_train = split_dataset_for_training(train_df, 'Survived')

    model = create_and_train_decision_tree_model(X_train, Y_train)

    training_acc = compute_accuracy(model, X_train, Y_train)

    return model, training_acc


def prepare_submission(model, test_df_path):
    df = pd.read_csv(test_df_path)
    ids = df['PassengerId']
    X_test = df.drop(['PassengerId'], axis=1)
    Y_pred = model.predict(X_test)

    data = {'PassengerId': ids, 'Survived': Y_pred}

    submission_df = pd.DataFrame(data)

    submission_df.to_csv('data/submission.csv')





