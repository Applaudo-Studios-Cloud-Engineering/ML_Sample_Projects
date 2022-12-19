import argparse
import pickle
import pandas as pd
from pipeline import create_ml_pipeline
import os

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--input', type=str, help='Input pickle dataframe')
parser.add_argument('--output_dir', type=str, help='Output folder')

args = parser.parse_args()

# Check if input and output were provided
if args.input is None or args.output_dir is None:
    print('Please provide input and output_dir args')
    exit(1)

df: pd.DataFrame = pickle.load(open(args.input, 'rb'))

model, training_acc = create_ml_pipeline(df)

print('Model trained successfully, acc: ', training_acc)

# if folder "args.output_dir" does not exist, create it
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

pickle.dump(model, open(f'{args.output_dir}/dt_classifier_acc_{round(training_acc, 2)}', 'wb'))
