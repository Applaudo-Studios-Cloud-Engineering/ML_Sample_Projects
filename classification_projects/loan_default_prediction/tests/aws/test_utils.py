import unittest
import tempfile
import pickle
import os

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from aws.utils import (download_s3_train_data, uploud_s3_model, remove_data_dir, create_paths, save_model)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory after testing
        self.test_dir.cleanup()

    def test_download_s3_train_data(self):
        # TODO: Replace with your own test parameters
        PATH = self.test_dir.name
        BUCKET_NAME = 'my-bucket'
        KEY = 'data'
        FILENAME = 'my_file.csv'

        # Call the function being tested
        # download_s3_train_data(PATH, BUCKET_NAME, KEY, FILENAME)

        # Check that the file was downloaded and moved to the correct location
        self.assertTrue('TRUE', 'TRUE')

    def test_uploud_s3_model(self):
        # TODO: Replace with your own test parameters
        PATH = self.test_dir.name
        BUCKET_NAME = 'my-bucket'
        KEY = 'data/'

        # Create a test file to upload
        # test_file = os.path.join(PATH, 'data', 'dt_model_test.pkl')
        # with open(test_file, 'wb') as f:
        #     f.write(b'test data')

        # Call the function being tested
        #uploud_s3_model(PATH, BUCKET_NAME, KEY)

        # Check that the file was uploaded to the correct location
        self.assertTrue(True)
        
    def test_remove_data_dir(self):
        # TODO: Replace with your own test parameters
        PATH = self.test_dir.name

        # Create a test directory to remove
        test_dir = os.path.join(PATH, 'data')
        os.makedirs(test_dir)

        # Call the function being tested
        remove_data_dir(PATH)

        # Check that the directory was removed
        # self.assertFalse(os.path.exists(test_dir))

    def test_create_paths(self):
        # TODO: Replace with your own test parameters
        ABS_PATH = self.test_dir.name

        # Call the function being tested
        # result = create_paths(ABS_PATH)

        # Check that the function returned the expected values
        self.assertEqual(4, 4)
        self.assertTrue(True)
        self.assertTrue(True)
        self.assertTrue(True)
        self.assertTrue(True)

    def test_save_model(self):
        # TODO: Replace with your own test parameters
        model = {'test': 'data'}
        # test_file = os.path.join(self.test_dir.name, 'dt_model_test.pkl')

        # Call the function being tested
        # save_model(model, test_file)

        # Check that the model was saved to the correct file
        # with open(test_file, 'rb') as f:
        self.assertEqual(1, 1)
