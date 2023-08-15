"""
Wrapper around dexom_python for iMat functionality, including creating condition specific models
"""
# Core imports
from typing import Callable, Union

# External imports
import cobra
from dexom_python.imat_functions import imat
from dexom_python.default_parameter_values import DEFAULT_VALUES
from dexom_python.gpr_rules import apply_gpr
import numpy as np
import pandas as pd

import consensus_essentiality.utils
# Local imports
from consensus_essentiality.condition_specific import enforce_off, \
    enforce_inactive, enforce_active, enforce_both, enforce_inactive_off
from consensus_essentiality.gene_expression import expression_to_weights


def imat_rxn_weights(model: cobra.Model, reaction_weights, method: str = "enforce off",
                     epsilon=DEFAULT_VALUES['epsilon'],
                     threshold=DEFAULT_VALUES["threshold"]):
    """
    Method to update a model using imat with reaction weights
    :param model: Base model to update to create condition specific model, copy created original model unaffected
    :type model: cobra.Model
    :param reaction_weights: Pandas series indexed by reaction, with values of 1 genes with high expression,
        -1 for genes with low expression values, and 0 for all other genes
    :type reaction_weights: pd.Series
    :param method: Which method to use for integrating imat result into the model, can be "enforce off",
        "enforce active", "enforce inactive", or "enforce both".
    :type method: str
    :param epsilon: Cutoff beneath which reactions are considered inactive, and above which reactions are considered
        active
    :type epsilon: float
    :param threshold: Minimum values for reactions to be considered on
    :type threshold: float
    :return: Condition specific model updated using imat
    :rtype: cobra.Model
    .. note::
       For the method parameter:

       "enforce off" means that reaction whose flux falls below the threshold value will be
       knocked out, with both bounds set to 0.

       "enforce inactive" means that reactions with weight -1 and a flux below epsilon will have their maximum flux
       set to epsilon.

       "enforce active" means that reactions with weight 1 and a flux above epsilon will have their minimum flux set
       to epsilon

       "enforce both" combines "enforce inactive" and "enforce active", forcing reactions with weight -1 and a flux
       below epsilon to have maximum flux set to epsilon, and reactions with weight 1 and a flux above epsilon have
       their minimum flux set to epsilon
    """
    result = imat(model, reaction_weights, epsilon=epsilon, threshold=threshold, full=False)
    low_expr_rxns = list(reaction_weights[np.isclose(reaction_weights, -1)].index)
    high_expr_rxns = list(reaction_weights[np.isclose(reaction_weights, 1)].index)
    method = consensus_essentiality.utils.parse_model_method(method)
    if method == "enforce_off":
        updated_model = enforce_off(model=model, solution=result, thr=threshold)
    elif method == "enforce_inactive":
        updated_model = enforce_inactive(model=model, solution=result, epsilon=epsilon,
                                         low_expr_rxns=low_expr_rxns)
    elif method == "enforce_active":
        updated_model = enforce_active(model=model, solution=result, epsilon=epsilon,
                                       high_expr_rxns=high_expr_rxns)
    elif method == "enforce_both":
        updated_model = enforce_both(model=model, solution=result, epsilon=epsilon,
                                     low_expr_rxns=low_expr_rxns, high_expr_rxns=high_expr_rxns)
    elif method == "enforce_inactive_off":
        updated_model = enforce_inactive_off(model=model, solution=result, epsilon=epsilon,
                                             thr=threshold, low_expr_rxns=low_expr_rxns)
    return updated_model


def imat_gene_expression(model: cobra.Model, gene_expression: Union[pd.DataFrame, pd.Series], gene_axis: int = 1,
                         proportion: Union[float, tuple] = 0.25, agg_fun: Callable = np.median,
                         method: str = "enforce_off",
                         epsilon=DEFAULT_VALUES['epsilon'], threshold=DEFAULT_VALUES["threshold"]):
    """
    Create a condition specific model
    :param model: Base model to turn into condition specific model (copy created, original model not affected)
    :type model: cobra.Model
    :param gene_expression: Gene expression dataframe or series, with genes as column labels, or index
    :type gene_expression: pd.Series | pd.DataFrame
    :param gene_axis: Which axis is labelled by gene, 0 for rows, 1 for columns
    :type gene_axis: int
    :param proportion: proportion of genes to be used for determining low and high gene expression
    :type proportion: tuple | float
    :param agg_fun: Function to use to combine gene expression values into 1, should be able to be passed to pandas
        aggregate method
    :type agg_fun: function
    :param method: Method to use for modifying the model with the iMat solution
    :type method: str
    :param epsilon: Cutoff where a reaction is considered active
    :type epsilon: float
    :param threshold: Cutoff where a reaction is considered off
    :type threshold: float
    :return: Condition specific model
    :rtype: cobra.Model
    .. note::
       For the method parameter:

       "enforce off" means that reaction whose flux falls below the threshold value will be
       knocked out, with both bounds set to 0.

       "enforce inactive" means that reactions with weight -1 and a flux below epsilon will have their maximum flux
       set to epsilon.

       "enforce active" means that reactions with weight 1 and a flux above epsilon will have their minimum flux set
       to epsilon

       "enforce both" combines "enforce inactive" and "enforce active", forcing reactions with weight -1 and a flux
       below epsilon to have maximum flux set to epsilon, and reactions with weight 1 and a flux above epsilon have
       their minimum flux set to epsilon
    """
    qual_gene_expression = expression_to_weights(gene_expression, proportion, gene_axis, agg_fun)
    rxn_weights = apply_gpr(model=model, gene_weights=qual_gene_expression, save=False, duplicates='remove', null=0.)
    return imat_rxn_weights(model=model, reaction_weights=rxn_weights, method=method,
                            epsilon=epsilon, threshold=threshold)


