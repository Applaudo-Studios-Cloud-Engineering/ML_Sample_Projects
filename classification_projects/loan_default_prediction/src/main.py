import os

from data_preprocessing import create_preprocessing_pipeline
from feature_engineering import create_feature_engineering_pipeline
from machine_learning import create_ML_pipeline
from aws import download_s3_train_data, uploud_s3_model, remove_data_dir, create_paths, save_model

ABS_PATH = os.path.abspath('')

BUCKET_NAME = 'mlflow-experiments-bucket' # replace with your bucket name
KEY = 'data/'
KEY_MODEL = 'models/'

# downloading from s3 the train data
download_s3_train_data(ABS_PATH, BUCKET_NAME, KEY, "train.csv")

train_path, preprocessing_path, feature_eng_path, path = create_paths(ABS_PATH)

# create a preprocessing pipeline using pipelines modules
train_df = create_preprocessing_pipeline(train_path, preprocessing_path)

# create a feature engineering using pipelines modules
features_df = create_feature_engineering_pipeline(preprocessing_path, feature_eng_path)

# create a ML model using pipelines modules
model, training_acc = create_ML_pipeline(feature_eng_path)

# save the model
save_model(model)

# uploud the model to a s3 bucket
uploud_s3_model(ABS_PATH, BUCKET_NAME, KEY_MODEL)

# remove the directory used to download the train data
remove_data_dir(path)
