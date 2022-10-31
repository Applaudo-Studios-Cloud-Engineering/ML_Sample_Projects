import pickle

from pipeline import create_preprocessing_pipeline, create_feature_engineering_pipeline, create_ml_pipeline, \
    prepare_submission

train_df = create_preprocessing_pipeline("C:\\Users\\rmora\\Documents\\Work\\Cloud Engineering\\MLOps_ML_Projects\\classification_projects\\titanic_challenge\\data\\train.csv", True)

features_df = create_feature_engineering_pipeline(train_df)

model, training_acc = create_ml_pipeline(features_df)

print('Model trained successfully, acc: ', training_acc)

pickle.dump(model, open(f'dt_classifier_acc_{round(training_acc)}', 'wb'))

submision_df = prepare_submission(model, 'C:\\Users\\rmora\\Documents\\Work\\Cloud Engineering\\MLOps_ML_Projects\\classification_projects\\titanic_challenge\\data\\test.csv',
                                  'C:\\Users\\rmora\\Documents\\Work\\Cloud Engineering\\MLOps_ML_Projects\\classification_projects\\titanic_challenge\\data\\submission.csv')

print(submision_df.head(10))
