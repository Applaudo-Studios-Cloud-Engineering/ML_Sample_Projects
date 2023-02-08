import pickle
import os

from data_preprocessing import create_preprocessing_pipeline
from feature_engineering import create_feature_engineering_pipeline
from machine_learning import create_ML_pipeline
from aws import download_s3_train_data, uploud_s3_model, remove_data_dir

ABS_PATH = os.path.abspath('')

BUCKET_NAME = 'mlflow-experiments-bucket' # replace with your bucket name
KEY = 'data/'
KEY_MODEL = 'models/'

# downloading from s3 the train data
download_s3_train_data(ABS_PATH, BUCKET_NAME, KEY, "train.csv")

path = f'{ABS_PATH}/data'

train_path = path.split('/')
train_path.append('train.csv')
train_path = '/'.join(train_path)

preprocessing_path = path.split('/')
preprocessing_path.append('preprocessing.csv')
preprocessing_path = '/'.join(preprocessing_path)

feature_eng_path = path.split('/')
feature_eng_path.append('feature_engineering.csv')
feature_eng_path = '/'.join(feature_eng_path)

os.chdir(path)

# create a preprocessing pipeline using pipelines modules
train_df = create_preprocessing_pipeline(train_path, preprocessing_path)

# create a feature engineering using pipelines modules
features_df = create_feature_engineering_pipeline(preprocessing_path, feature_eng_path)

# create a ML model using pipelines modules
model, training_acc = create_ML_pipeline(feature_eng_path)

print('Model trained successfully, acc: ', training_acc)

# Save the model in a file called "dt_classifier_acc_*"
pickle.dump(model, open(f'dt_classifier_acc_{round(training_acc)}.pkl', 'wb'))

# uploud the model to a s3 bucket
uploud_s3_model(ABS_PATH, BUCKET_NAME, KEY_MODEL)

# remove the directory used to download the train data
remove_data_dir(path)
