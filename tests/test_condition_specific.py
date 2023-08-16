# Standard Library Imports
import unittest
from unittest.mock import patch, MagicMock, Mock
# External Imports
from cobra import Model, Reaction, Metabolite
import numpy as np
import pandas as pd
# Local imports
from consensus_essentiality.condition_specific import _force_inactive, _force_active


class TestHelperFunctions(unittest.TestCase):
    """
    Tests for the helper functions in condition specific module
    """

    def test_force_inactive(self):
        """
        Test the _force_inactive function
        """
        self.assertEqual(_force_inactive(epsilon=1, lb=-2, ub=2, reversible=True), (-1, 1))
        self.assertEqual(_force_inactive(epsilon=2, lb=-1, ub=1, reversible=True), (-1, 1))
        self.assertEqual(_force_inactive(epsilon=1, lb=0, ub=2, reversible=False), (0, 1))
        self.assertEqual(_force_inactive(epsilon=2, lb=0, ub=1, reversible=False), (0, 1))
        self.assertEqual(_force_inactive(epsilon=1, lb=-2, ub=0, reversible=False), (-1, 0))
        self.assertEqual(_force_inactive(epsilon=2, lb=-1, ub=0, reversible=False), (-1, 0))
        with self.assertRaises(ValueError):
            _ = _force_inactive(epsilon=1, lb=3, ub=4, reversible=False)
        with self.assertRaises(ValueError):
            _ = _force_inactive(epsilon=1, lb=-3, ub=-2, reversible=False)
        with self.assertRaises(ValueError):
            _ = _force_inactive(epsilon=-1, lb=0, ub=2, reversible=False)

    def test_force_active(self):
        """
        Test the _force_active function
        """
        self.assertEqual(_force_active(epsilon=1, lb=-2, ub=2, reversible=True, forward=True), (1, 2))
        self.assertEqual(_force_active(epsilon=1, lb=2, ub=4, reversible=False, forward=True), (2, 4))
        self.assertEqual(_force_active(epsilon=1, lb=-4, ub=-2, reversible=False, forward=False), (-4, -2))
        self.assertEqual(_force_active(epsilon=4, lb=3, ub=5, reversible=False, forward=True), (4, 5))
        self.assertEqual(_force_active(epsilon=4, lb=-5, ub=-3, reversible=False, forward=False), (-5, -4))
        self.assertEqual(_force_active(epsilon=1, lb=-2, ub=2, reversible=True, forward=False), (-2, -1))
        with self.assertRaises(ValueError):
            _ = _force_active(epsilon=5, lb=-1, ub=4, reversible=True, forward=True)
        with self.assertRaises(ValueError):
            _ = _force_active(epsilon=5, lb=-4, ub=1, reversible=True, forward=False)
        with self.assertRaises(ValueError):
            _ = _force_active(epsilon=5, lb=0, ub=4, reversible=False, forward=True)
        with self.assertRaises(ValueError):
            _ = _force_active(epsilon=5, lb=-4, ub=0, reversible=False, forward=False)
        with self.assertRaises(ValueError):
            _ = _force_active(epsilon=-4, lb=1, ub=5, reversible=False, forward=True)


class TestConditionSpecificFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up data for the tests, creating a test model
        """
        test_model = Model("test")
        metabolite_list = []
        for i in range(1, 11):
            met = Metabolite("M%i_c" % i, formula="C%iH%iN%i" % (i, 2 * i + 1, 1), name="Metabolite %i" % i,
                             compartment='c')
            metabolite_list.append(met)
        met1_ext = Metabolite("M1_e", formula="C1H3N1", name="Metabolite 1 ext", compartment="e")
        R1 = Reaction("R1", name="Reaction 1", subsystem="subsystem 1", lower_bound=0, upper_bound=10)
        R1.add_metabolites({
            metabolite_list[0]: -1,
            metabolite_list[1]: -1,
            metabolite_list[2]: 1,
        })
        R1.gene_reaction_rule = '(G1 or G2)'

        R2 = Reaction("R2", name="Reaction 2", subsystem="subsystem 2", lower_bound=-5, upper_bound=5)
        R2.add_metabolites({
            metabolite_list[3]: -1,
            metabolite_list[4]: 1,
        })
        R2.gene_reaction_rule = '(G3 and G5)'

        R3 = Reaction("R3", name="Reaction 3", subsystem="subsystem 2", lower_bound=5, upper_bound=10)
        R3.add_metabolites({
            metabolite_list[5]: 1,
            metabolite_list[6]: -1,
            metabolite_list[4]: 1,
        })
        R3.gene_reaction_rule = 'G4'

        R4 = Reaction("R4", name="Reaction 4", subsystem="subsystem 1", lower_bound=-6, upper_bound=0)
        R4.add_metabolites({
            metabolite_list[7]: -1,
            metabolite_list[8]: -1,
            metabolite_list[9]: 1,
        })
        R4.gene_reaction_rule = '(G6 or G7)'

        R5 = Reaction("R5", name="Reaction 5", subsystem="subsystem 3", lower_bound=-5, upper_bound=-2)
        R5.add_metabolites({
            metabolite_list[4]: -1,
            metabolite_list[2]: -1,
            metabolite_list[6]: 1,
            metabolite_list[7]: 1,
        })
        R5.gene_reaction_rule = '(G1 or G2)'

        R6 = Reaction("R6", name="Reaction 6", subsystem="boundary", lower_bound=-5, upper_bound=5)
        R6.add_metabolites({
            met1_ext: -1,
            metabolite_list[1]: 1
        })
        R6.gene_reaction_rule = '(G1 or G2)'

        R7 = Reaction("R7", name="Reaction 7", subsystem="exchange", lower_bound=-10, upper_bound=0)
        R7.add_metabolites({
            met1_ext: -1
        })

        test_model.add_reactions([R1, R2, R3, R4, R5, R6, R7])
        cls.test_model = test_model
        fluxes = pd.Series([0,0,0,0,0,0,0], index = ["R1","R2","R3","R4","R5","R6","R7"])
        cls.mock_solution = Mock(fluxes = fluxes)

    def test_enforce_off(self):
        """
        Test the enforce_off function
        """
        mock_model = MagicMock()
        mock_solution = MagicMock(fluxes=pd.Series())


if __name__ == '__main__':
    unittest.main()
