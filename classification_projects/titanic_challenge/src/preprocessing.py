import argparse
import pickle
import os
from pipeline import create_preprocessing_pipeline

parser = argparse.ArgumentParser(description='Pre-processing')
parser.add_argument('--input', type=str, help='Input CSV dataset')
parser.add_argument('--output', type=str, help='Output pickle dataframe')

args = parser.parse_args()

# Check if input and output were provided
if args.input is None or args.output is None:
    print('Please provide input and output files')
    exit(1)

processed_df = create_preprocessing_pipeline(args.input, True)

pickle.dump(processed_df, open(args.output, 'wb'))
