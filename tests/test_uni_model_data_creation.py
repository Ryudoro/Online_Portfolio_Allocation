import unittest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model_data_creation import model_data_creation

class TestModelDataCreation(unittest.TestCase):

    def test_valid_data(self):
        data_to_use = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        days_for_training = 5
        days_for_testing = 2
        x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)
        self.assertIsNotNone(x_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(scaler)

    def test_empty_data(self):
        data_to_use = np.array([])
        days_for_training = 5
        days_for_testing = 2
        x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)
        self.assertIsNone(x_train)
        self.assertIsNone(y_train)
        self.assertIsNone(X_test)
        self.assertIsNotNone(scaler)

    def test_invalid_data_length(self):
        data_to_use = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        days_for_training = 15
        days_for_testing = 2
        x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)
        self.assertIsNone(x_train)
        self.assertIsNone(y_train)
        self.assertIsNone(X_test)
        self.assertIsNotNone(scaler)

    def test_days_for_testing_equal_to_zero(self):
        data_to_use = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        days_for_training = 5
        days_for_testing = 0
        x_train, y_train, X_test, scaler = model_data_creation(data_to_use, days_for_training, days_for_testing)
        self.assertIsNotNone(x_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(scaler)

    # Add more test cases as needed...

if __name__ == '__main__':
    unittest.main()