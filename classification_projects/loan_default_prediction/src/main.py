from data_preprocessing import create_preprocessing_pipeline
from feature_engineering import create_feature_engineering_pipeline

clean_ds = create_preprocessing_pipeline('./data/train.csv', './data/train_processed.csv')
print(clean_ds.head())

features = create_feature_engineering_pipeline('./data/train_processed.csv', './data/features.csv')
print(features.head())
