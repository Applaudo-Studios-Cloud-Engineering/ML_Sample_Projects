import argparse
import pickle
import pandas as pd
from pipeline import create_feature_engineering_pipeline

parser = argparse.ArgumentParser(description='Feature engineering')
parser.add_argument('--input', type=str, help='Input pickle dataframe')
parser.add_argument('--output', type=str, help='Output pickle dataframe')

args = parser.parse_args()

# Check if input and output were provided
if args.input is None or args.output is None:
    print('Please provide input and output files')
    exit(1)

df: pd.DataFrame = pickle.load(open(args.input, 'rb'))

features_df = create_feature_engineering_pipeline(df)

pickle.dump(features_df, open(args.output, 'wb'))
