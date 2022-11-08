import pickle
import os

from pipeline import create_preprocessing_pipeline, create_feature_engineering_pipeline, create_ml_pipeline, \
    prepare_submission

os.chdir('../data')

train_df = create_preprocessing_pipeline(os.getcwd() + '\\train.csv', True)

features_df = create_feature_engineering_pipeline(train_df)

model, training_acc = create_ml_pipeline(features_df)

print('Model trained successfully, acc: ', training_acc)

pickle.dump(model, open(f'dt_classifier_acc_{round(training_acc)}', 'wb'))

submission_df = prepare_submission(model, os.getcwd() + '\\test.csv', os.getcwd() + '\\submission.csv')

print(submission_df.head(10))
