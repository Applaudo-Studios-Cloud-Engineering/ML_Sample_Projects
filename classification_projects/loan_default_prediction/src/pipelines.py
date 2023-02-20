"""
This module provides functions for performing various mathematical operations.
"""
import os

from src.data_preprocessing.pipelines import create_preprocessing_pipeline
from src.feature_engineering.pipelines import create_feature_engineering_pipeline
from src.machine_learning.pipelines import create_ML_pipeline
from src.aws import download_s3_train_data, uploud_s3_model, remove_data_dir, create_paths, save_model

from project_pipelines import create_pipeline, Node

ABS_PATH = os.path.abspath('')

BUCKET_NAME = 'mlflow-experiments-bucket' # replace with your bucket name
KEY = 'data/'
KEY_MODEL = 'models/'

def generate_pipeline():
    """
    This is a docstring that describes what the function does.

    Parameters:
    arg1 (int): The first argument.
    arg2 (str): The second argument.

    Returns:
    str: The result of the function.
    """

    return create_pipeline([
        Node(
            download_s3_train_data,
            {
                "path": ABS_PATH,
                "bucket_name": BUCKET_NAME,
                "key": KEY,
                "filename": "train.csv"
            },
            None,
            "download_train_data"),
        Node(
            create_paths,
            {
                "path": ABS_PATH,
            },
            ["train_path", "preprocessing_path", "feature_eng_path", "path"],
            "create_paths"),
        Node(
            create_preprocessing_pipeline,
            {
                "path_to_csv": "train_path", 
                "path_to_save": "preprocessing_path"
            },
            ["train_df"],
            "preprocessing_pipeline"),
        Node(
            create_feature_engineering_pipeline,
            {
                "path_to_csv": "preprocessing_path", 
                "path_to_save": "feature_eng_path"
            },
            ["features_df"],
            "feature_engineering_pipeline"),
        Node(
            create_ML_pipeline,
            {
                "path_to_csv": "feature_eng_path", 
            },
            ["model", "training_acc"],
            "Machine_learning_pipeline"),
        Node(
            save_model,
            {
                "model": "model",
                "training_acc": "training_acc"
            },
            None,
            "save_model"),
        Node(
            uploud_s3_model,
            {
                "path": ABS_PATH,
                "bucket_name": BUCKET_NAME,
                "key": KEY_MODEL
            },
            None,
            "uploud_model_to_s3"),
        Node(
            remove_data_dir,
            ["path"],
            None,
            "remove_data_directory")
    ])