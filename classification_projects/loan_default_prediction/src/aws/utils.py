import shutil
import re
import os

import boto3
import botocore

def download_s3_train_data(PATH, BUCKET_NAME, KEY, FILENAME):
    s3_client = boto3.client('s3')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    models = []
    
    for s3_object in bucket.objects.all():
        for key in bucket.objects.all():
            x = re.search("^data/*", key.key)
            if x:
                models.append(key.key)
    
    FOLDER = models[models.index(''.join([KEY, FILENAME]))]
    print(FOLDER)
    
    try:
        s3_client.download_file(BUCKET_NAME, FOLDER, FILENAME)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    
    DIR_NAME = f'{PATH}/data'
    
    if not os.path.isdir(DIR_NAME):
        os.mkdir(DIR_NAME, 0o777)
        
    src_path = f"{PATH}/{FILENAME}"
    dst_path = f'{PATH}/data/{FILENAME}'
    shutil.move(src_path, dst_path)

def uploud_s3_model(PATH, BUCKET_NAME, KEY):
    client = boto3.client('s3')
    entries = os.listdir(f'{PATH}/data')
    filenames = [value for value in entries if re.search('^dt_classifier_acc_*', value)]
    if filenames:
        filename = filenames[-1]
        client.upload_file(f"{PATH}/data/{filename}", BUCKET_NAME, f'{KEY}{filename}')
    else:
        print("No matching file found.")

def remove_data_dir(PATH):
    shutil.rmtree(PATH)
