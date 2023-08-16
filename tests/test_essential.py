# Standard Library Imports
import unittest
# External Imports
import pandas as pd
# Local Imports
from consensus_essentiality.essential import aggstrat_all, aggstrat_majority, aggstrat_any


class TestAggStratFunctions(unittest.TestCase):
    """
    Unittest test case for the Aggregation strategy Functions
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up essentiality series for testing
        """
        cls.all_true = pd.Series([True, True, True, True], dtype="boolean")
        cls.all_true_na = pd.Series([True, True, True, True, pd.NA], dtype="boolean")
        cls.all_false = pd.Series([False, False, False, False], dtype="boolean")
        cls.all_false_na = pd.Series([False, False, False, False, pd.NA], dtype="boolean")
        cls.most_true = pd.Series([True, True, True, False], dtype="boolean")
        cls.most_true_na = pd.Series([True, True, True, False, pd.NA], dtype="boolean")
        cls.most_false = pd.Series([False, False, False, False, True], dtype="boolean")
        cls.most_false_na = pd.Series([False, False, False, False, True, pd.NA], dtype="boolean")
        cls.half_true = pd.Series([True, True, True, False, False, False], dtype="boolean")
        cls.half_true_na = pd.Series([True, True, True, False, False, False, pd.NA], dtype="boolean")

    def test_strat_all(self):
        """
        Test the aggstrat_all function
        """
        # If all true should be true
        self.assertTrue(aggstrat_all(self.all_true))
        # If all are true except NAs, should return NA
        self.assertTrue(pd.isna(aggstrat_all(self.all_true_na)))
        # If all are false should return false
        self.assertFalse(aggstrat_all(self.all_false))
        # If any are false should return false
        self.assertFalse(aggstrat_all(self.most_true))
        # Even if there is an NA
        self.assertFalse(aggstrat_all(self.most_true_na))
        # If all are true except NAs, and ignore_na is true, should return true
        self.assertTrue(aggstrat_all(self.all_true_na, ignore_na=True))
        # If there is a single false, and ignore_na is true, should return False
        self.assertFalse(aggstrat_all(self.most_true_na, ignore_na=True))

    def test_strat_any(self):
        """
        Test the aggstrat_any function
        """
        # If any Trues are present, should return True
        self.assertTrue(aggstrat_any(self.most_true))
        self.assertTrue(aggstrat_any(self.most_true_na))
        self.assertTrue(aggstrat_any(self.most_false))
        self.assertTrue(aggstrat_any(self.most_false_na))
        # If all False, should return False
        self.assertFalse(aggstrat_any(self.all_false))
        # Should return NA if all False with an NA
        self.assertTrue(pd.isna(aggstrat_any(self.all_false_na)))
        # Unless ignore na is True
        self.assertFalse(aggstrat_any(self.all_false_na, ignore_na=True))

    def test_strat_majority(self):
        """
        Test the aggstrat_majority function
        """
        # If all or majority are true, should return True
        self.assertTrue(aggstrat_majority(self.all_true))
        self.assertTrue(aggstrat_majority(self.all_true_na))
        self.assertTrue(aggstrat_majority(self.most_true))
        self.assertTrue(aggstrat_majority(self.most_true_na))
        # If all or majority are false, should return false
        self.assertFalse(aggstrat_majority(self.all_false))
        self.assertFalse(aggstrat_majority(self.all_false_na))
        self.assertFalse(aggstrat_majority(self.most_false))
        self.assertFalse(aggstrat_majority(self.most_false_na))
        # If half false, should return false
        self.assertFalse(aggstrat_majority(self.half_true))
        # Unless NA is present
        self.assertTrue(pd.isna(aggstrat_majority(self.half_true_na)))
        # If half false and half true other than NA, should return false
        self.assertFalse(aggstrat_majority(self.half_true_na, ignore_na=True))


if __name__ == '__main__':
    unittest.main()
