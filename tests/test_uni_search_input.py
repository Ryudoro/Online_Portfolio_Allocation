import unittest
import pandas as pd
import numpy as np
from Model.input_creation import search_input

class TestSearchInput(unittest.TestCase):

    def test_valid_stock_symbol_default_period(self):
        stock_symbol = 'ALO.PA'
        data, data_to_use = search_input(stock_symbol)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(data_to_use, np.ndarray)
        self.assertEqual(data_to_use.shape[0], len(data))
        self.assertEqual(len(data_to_use.shape), 1)

    def test_valid_stock_symbol_custom_period(self):
        stock_symbol = 'ALO.PA'
        period = '2y'
        data, data_to_use = search_input(stock_symbol, period)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(data_to_use, np.ndarray)
        self.assertEqual(data_to_use.shape[0], len(data))
        self.assertEqual(len(data_to_use.shape), 1)

    def test_invalid_stock_symbol(self):
        stock_symbol = 'INVALID'
        data, data_to_use = search_input(stock_symbol)
        self.assertIsNone(data)
        self.assertIsNone(data_to_use)

    def test_invalid_period_format(self):
        stock_symbol = 'AAPL'
        period = 'abc'
        data, data_to_use = search_input(stock_symbol, period)
        self.assertIsNone(data)
        self.assertIsNone(data_to_use)

    def test_no_data_for_period(self):
        stock_symbol = 'AAPL'
        period = '50y'
        data, data_to_use = search_input(stock_symbol, period)
        self.assertIsNone(data)
        self.assertIsNone(data_to_use)

    def test_consistency_data_and_data_to_use(self):
        stock_symbol = 'GOOGL'
        data, data_to_use = search_input(stock_symbol)
        self.assertEqual(len(data), data_to_use.shape[0])
        self.assertEqual(len(data_to_use.shape), 1)

if __name__ == '__main__':
    unittest.main()
