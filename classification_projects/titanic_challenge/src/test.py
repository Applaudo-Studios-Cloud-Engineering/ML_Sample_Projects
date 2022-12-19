import argparse
import pickle
import pandas as pd
import os
from pipeline import prepare_submission
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input', type=str, help='Input CSV dataset')

args = parser.parse_args()

# Check if input and output were provided
if args.input is None:
    print('Please provide input CSV file')
    exit(1)

os.chdir('models')

model_name = os.listdir('.')[0]

model: DecisionTreeClassifier = pickle.load(open(model_name, 'rb'))

os.chdir('../data')

submission_df = prepare_submission(model, os.getcwd() + '/test.csv', os.getcwd() + '/submission.csv')

print(submission_df.head(10))