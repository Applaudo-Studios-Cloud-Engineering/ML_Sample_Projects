from pipeline import create_preprocessing_pipeline, create_feature_engineering_pipeline, create_ml_pipeline, prepare_submission
import pickle

train_df = create_preprocessing_pipeline('../data/train.csv', True)

print("Finished preprocessing pipeline")

# features_df = create_feature_engineering_pipeline(train_df)
#
# model, training_acc = create_ml_pipeline(features_df)
#
# print('Model trained successfully, acc: ', training_acc)
#
# pickle.dump(model, open(f'dt_classifier_acc_{round(training_acc)}', 'wb'))
