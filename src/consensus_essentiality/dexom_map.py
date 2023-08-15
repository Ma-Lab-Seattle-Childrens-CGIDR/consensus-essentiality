"""
Module with functionality to map a function to DEXOM enumeration methods
"""
# region imports
# Standard Library Imports
from typing import Callable

# External Imports
import cobra
import numpy as np
from dexom_python.default_parameter_values import DEFAULT_VALUES
import pandas as pd

# Local imports
from consensus_essentiality.condition_specific import EnforceOff, EnforceBoth, EnforceActive, EnforceInactive, \
    EnforceInactiveOff
from consensus_essentiality.utils import parse_model_method, create_iterator


# endregion


# region mapping functions
def sol_map(model: cobra.Model, reaction_weights: pd.Series, fun: Callable, fun_kwargs: dict,
            enum_method: str = "diversity",
            **kwargs):
    """
    The sol_map function is a convenience function that allows you to apply a function to each solution from an
    enumeration iterator. It will apply the supplied `fun` function to each solution, gather the results into a list
    then return that.

    :param model: Base metabolic model to use for creating context specific models
    :type model: cobra.Model
    :param reaction_weights: Reaction weights for the enumeration method, indexed by gene, where values of -1 mean
        the gene has a low expression level, values of 1 mean the gene has a high expression level, and 0 is for all
        other genes
    :type reaction_weights: pd.Series
    :param fun: Specify the function that will be applied to each solution, should accept cobra.Solution as first
        argument
    :type fun: Callable
    :param fun_kwargs: Additional keyword arguments passed to the function fun
    :type fun_kwargs: dict
    :param enum_method: Specify the enumeration method to use
    :type enum_method: str
    :param kwargs: Keyword arguments passed to the enumeration method class init, see documentation for enumeration
        method for possible arguments
    :type kwargs: dict
    :return: A list of the results of applying the function to the solution objects from
    :rtype: list
    """
    result = []
    iterator = create_iterator(model, reaction_weights, enum_method, kwargs)
    for sol, stats in iterator:
        # If the solution attempt resulted in an error, skip this iteration and continue to the next
        # The iterator should have reverted to a state that is safe to continue from
        if sol.error:
            continue
        result.append(fun(sol, **fun_kwargs))
    return result


def model_map(model: cobra.Model, reaction_weights: pd.Series, fun: Callable, fun_kwargs: dict,
              enum_method: str = "diversity", context_specific_method: str = "enforce_off", **kwargs):
    """
    Apply a function to each context specific model created by the selected DEXOM enumeration method.

    :param model: Base model to create the context specific models from
    :type model: cobra.Model
    :param reaction_weights: Reaction weights for the enumeration method, indexed by reaction id, where values of -1
        mean the reaction has a low expression level, values of 1 mean the reaction has a high expression level,
        and 0 is for all other reactions
    :type reaction_weights: pd.Series
    :param fun: Specify the function that will be applied to each context specific model, should take model as first
        argument
    :type fun: Callable
    :param fun_kwargs: Pass additional arguments to the function that is being mapped
    :type fun_kwargs: dict
    :param enum_method: str: Determine how the context specific models are enumerated, options are diversity, max_dist,
        or icut
    :type enum_method: str
    :param context_specific_method: Determine which method should be used for creating the context specific models,
        options are enforce_off, enforce_inactive, enforce_inactive_off, enforce_active, or enforce_both
    :type context_specific_method: str
    :param kwargs: Pass in arguments to the iterator's constructor function
    :type kwargs: dict
    :return: A list of results
    :rtype: list
    :doc-author: Trelent
    """
    if "tol" in kwargs:
        tolerance = kwargs["tol"]
    else:
        tolerance = DEFAULT_VALUES["tolerance"]
    if "epsilon" in kwargs:
        epsilon = kwargs["epsilon"]
    else:
        epsilon = DEFAULT_VALUES["epsilon"]
    low_expr_rxns = list(reaction_weights[np.isclose(reaction_weights, -1.)].index)
    high_expr_rxns = list(reaction_weights[np.isclose(reaction_weights, 1.)].index)
    results = []
    base_model = model.copy()
    iterator = create_iterator(model, reaction_weights, enum_method, kwargs)
    context_method = parse_model_method(context_specific_method)
    for sol, stat in iterator:
        # Check for possible error, if there is an error skip this step, the iterator should handle reverting to
        #   safe state when there is an error
        if sol.error:
            continue
        if context_method == "enforce_off":
            manager = EnforceOff(base_model, solution=sol, thr=tolerance)
        elif context_method == "enforce_inactive":
            manager = EnforceInactive(base_model, solution=sol, epsilon=epsilon, low_expr_rxns=low_expr_rxns)
        elif context_method == "enforce_inactive_off":
            manager = EnforceInactiveOff(base_model, solution=sol, epsilon=epsilon, thr=tolerance,
                                         low_expr_rxns=low_expr_rxns)
        elif context_method == "enforce_active":
            manager = EnforceActive(base_model, solution=sol, epsilon=epsilon, high_expr_rxns=high_expr_rxns)
        elif context_method == "enforce_both":
            manager = EnforceBoth(base_model, solution=sol, epsilon=epsilon, low_expr_rxns=low_expr_rxns,
                                  high_expr_rxns=high_expr_rxns)
        else:
            raise ValueError("Couldn't parse context specific method: %s" % context_specific_method)
        with manager as context_model:
            results.append(fun(context_model, **fun_kwargs))
    return results
# endregion
