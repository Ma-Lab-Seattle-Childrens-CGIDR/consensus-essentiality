"""
Function for using dexom method to create several models and then compute essential genes from them
"""

# region Imports
# Standard Library Imports
from functools import reduce
from typing import Union, Callable
# External Imports
import cobra
from cobra.flux_analysis import single_gene_deletion
from dexom_python.default_parameter_values import DEFAULT_VALUES
import numpy as np
import pandas as pd
# Local Imports
import consensus_essentiality.utils
from consensus_essentiality.condition_specific import EnforceBoth, EnforceOff, EnforceActive, EnforceInactive, \
    EnforceInactiveOff
from consensus_essentiality.gene_expression import expression_to_weights


# endregion

# region Consensus Essentiality Functions
def essential_genes(model: cobra.Model, essential_prop: float, return_error_status_genes: bool) -> Union[
    set, tuple[set, set]]:
    """
    Determine which genes are essential for a model
    :param model: Genome scale metabolic model to determine essential genes for
    :type model: cobra.Model
    :param essential_prop: Cutoff for proportion of optimal growth achievable after gene knockout, if the optimal
        flux after a gene knockout is less than essential_prop*optimial_growth, the gene is considered essential
    :type essential_prop: float
    :param return_error_status_genes: Whether to return the genes which have non-optimal solver statuses after KO
        along with the essential genes
    :return:
    """
    unrestricted_growth = model.slim_optimize()
    ko_flux = single_gene_deletion(model)
    ko_flux["growth_prop"] = ko_flux["growth"] / unrestricted_growth
    ess_genes = ko_flux[ko_flux["growth_prop"] <= essential_prop]["ids"]
    essential = reduce(lambda x, y: x | y, ess_genes)
    if not return_error_status_genes:
        return essential
    error_genes = reduce(lambda x, y: x | y, ko_flux[ko_flux["status"] != "optimal"]["ids"])
    return essential, error_genes


# function to find essential genes using the consensus method described in dexom
def consensus_essential_rxn_weights(model: cobra.Model, essential_prop: float, reaction_weights: pd.Series,
                                    enum_method: str = "diversity", context_specific_method: str = "enforce_off",
                                    **kwargs):
    """
    Method to find essential genes using enumerated context specific models

    :param model: Base model to create the context specific models from
    :type model: cobra.Model
    :param essential_prop: Cutoff for proportion of optimal growth achievable after gene knockout, if the optimal
        flux after a gene knockout is less than essential_prop*optimial_growth, the gene is considered essential
    :type essential_prop: float
    :param reaction_weights: Reaction weights for the enumeration method, indexed by reaction id, where values of -1
        mean the reaction has a low expression level, values of 1 mean the reaction has a high expression level,
        and 0 is for all other reactions
    :type reaction_weights: pd.Series
    :param enum_method: str: Determine how the context specific models are enumerated, options are diversity, max_dist,
        or icut
    :type enum_method: str
    :param context_specific_method: Determine which method should be used for creating the context specific models,
        options are enforce_off, enforce_inactive, enforce_inactive_off, enforce_active, or enforce_both
    :type context_specific_method: str
    :param kwargs: Arguments to pass to iterator's class constructor method
    :type kwargs: dicts
    :return: Dataframe of essentiality predictions, True means essential, False means non-essential, and pd.NA means
        that the solver returned an error
    :rtype: pd.DataFrame
    """
    if "maxiter" in kwargs:
        maxiter = kwargs["maxiter"]
    else:
        maxiter = DEFAULT_VALUES["maxiter"]
    genes = [gene.id for gene in model.genes]
    iter_cols = ["iter_%i" % iteration for iteration in range(1, maxiter + 1)]
    essential_genes_df = pd.DataFrame(False, index=genes, columns=iter_cols)
    base_model = model.copy()
    iterator = consensus_essentiality.utils.create_iterator(model=model, reaction_weights=reaction_weights,
                                                            enum_method=enum_method, kwargs=kwargs)
    context_method = consensus_essentiality.utils.parse_model_method(context_specific_method)
    low_expr_rxns = list(reaction_weights[np.isclose(reaction_weights, -1.)].index)
    high_expr_rxns = list(reaction_weights[np.isclose(reaction_weights, 1.)].index)
    if "tolerance" in kwargs:
        tolerance = kwargs["tolerance"]
    else:
        tolerance = DEFAULT_VALUES["tolerance"]
    if "epsilon" in kwargs:
        epsilon = kwargs["epsilon"]
    else:
        epsilon = DEFAULT_VALUES["epsilon"]
    for i, (solution, stats) in enumerate(iterator):
        # Check for possible error, if there is an error skip this step, the iterator should handle reverting to
        #   safe state when there is an error
        if solution.error:
            continue
        if context_method == "enforce_off":
            manager = EnforceOff(base_model, solution=solution, thr=tolerance)
        elif context_method == "enforce_inactive":
            manager = EnforceInactive(base_model, solution=solution, epsilon=epsilon, low_expr_rxns=low_expr_rxns)
        elif context_method == "enforce_inactive_off":
            manager = EnforceInactiveOff(base_model, solution=solution, epsilon=epsilon, thr=tolerance,
                                         low_expr_rxns=low_expr_rxns)
        elif context_method == "enforce_active":
            manager = EnforceActive(base_model, solution=solution, epsilon=epsilon, high_expr_rxns=high_expr_rxns)
        elif context_method == "enforce_both":
            manager = EnforceBoth(base_model, solution=solution, epsilon=epsilon, low_expr_rxns=low_expr_rxns,
                                  high_expr_rxns=high_expr_rxns)
        else:
            raise ValueError("Couldn't parse context specific method: %s" % context_specific_method)
        with manager as context_model:
            ess_genes, error_genes = essential_genes(context_model, essential_prop=essential_prop,
                                                     return_error_status_genes=True)

        essential_genes_df.loc[list(ess_genes), "iter_%i" % (i + 1)] = True
        essential_genes_df.loc[list(error_genes), "iter_%i" % (i + 1)] = pd.NA
        final_iter = i

    return essential_genes_df


def consensus_essential_gene_expr(model: cobra.Model, essential_prop: float,
                                  gene_expression: Union[pd.DataFrame, pd.Series],
                                  gene_axis: int = 1, proportion: Union[float, tuple] = 0.25,
                                  agg_fun: Callable[[pd.Series], Union[float, int]] = np.median,
                                  enum_method: str = "diversity",
                                  context_specific_method: str = "enforce_off",
                                  **kwargs):
    """
    Function to compute essential genes using the consensus method using gene expression data to create context
    specific models

    :param model: Base metabolic model to create the context specific models from
    :type model: cobra.Model
    :param essential_prop: Cutoff for proportion of optimal growth achievable after gene knockout, if the optimal
        flux after a gene knockout is less than essential_prop*optimial_growth, the gene is considered essential
    :type essential_prop: float
    :param gene_expression: Gene expression data, either a dataframe with genes as one index, and samples as the other,
        or a Series with genes as the index
    :type gene_expression: Union[pd.DataFrame, pd.Series]
    :param gene_axis: Which axis the genes are on in gene_expression, 0 for rows, 1 for columns
    :type gene_axis: int
    :param proportion: Proportion to use for trinarizing the data. If a float between 0 and 1, values below that
        percentile will be considered low expression, values whose percentiles are
        above 1-proportion will be considered high expression. If a tuple, the first element will be used as the low
        expression percentile threshold, and the second element will be used as the high expression percentile
        threshold.
    :type proportion: Union[float, tuple[float,float]]
    :param agg_fun: Which function to use for combining multiple expression values into 1 which can then be converted
        into the qualitative reaction weight.
    :type agg_fun: Callable[[pd.Series], Union[float,int]]
    :param enum_method: str: Determine how the context specific models are enumerated, options are diversity, max_dist,
        or icut
    :type enum_method: str
    :param context_specific_method: Determine which method should be used for creating the context specific models,
        options are enforce_off, enforce_inactive, enforce_inactive_off, enforce_active, or enforce_both
    :type context_specific_method: str
    :param kwargs: Arguments to pass to iterator's class constructor method
    :type kwargs: dicts
    :return: Dataframe of essentiality predictions, True means essential, False means non-essential, and pd.NA means
        that the solver returned an error
    :rtype: pd.DataFrame
    """
    rxn_weights = expression_to_weights(gene_expression=gene_expression, proportion=proportion, gene_axis=gene_axis,
                                        agg_fun=agg_fun)
    return consensus_essential_rxn_weights(model=model, essential_prop=essential_prop, reaction_weights=rxn_weights,
                                           enum_method=enum_method, context_specific_method=context_specific_method,
                                           **kwargs)


# endregion

# region Functions for Combining Consensus Essentiality Predictions


def compute_essentiality(consensus_essentiality_df: pd.DataFrame, aggregation_func: Callable[[pd.Series], bool],
                         **kwargs):
    """
    Computes essentiality predictions from the consensus_essentiality dataframe

    :param consensus_essentiality_df:
    :param aggregation_func:
    :return:
    """
    return consensus_essentiality_df.agg(aggregation_func, **kwargs)


# endregion

# region Aggregation Strategies

def aggstrat_any(essentiality_series: pd.Series, ignore_na: bool = False):
    """
    Aggregation strategy function, where a gene is considered essential if it is essential in any of the context
    specific models

    :param essentiality_series: Series of essentiality predictions, with True for predicted essential, False for
        predicted non-essential, and NA for any infeasible status returns from solver.
    :type essentiality_series: pd.Series
    :param ignore_na: Whether to ignore NA values. When False, will return True if any values are True, False if
        all values are False and none are NA, and NA when all non-NA values are False.
    :return: True if gene is considered essential by this strategy, False if gene is considered non-essential, and NA
        if the gene essentiality can't be determined by this strategy
    :rtype: bool
    """
    if not ignore_na:
        res = np.any(essentiality_series)
        if res:
            return res
        return pd.NA if np.any(pd.isna(essentiality_series)) else res
    return np.any(essentiality_series)


def aggstrat_all(essentiality_series: pd.Series, ignore_na: bool = False):
    """
    Aggregation strategy function, where a gene is considered essential if it is essential in all the context
    specific models

    :param essentiality_series: Series of essentiality predictions, with True for predicted essential, False for
        predicted non-essential, and NA for any infeasible status returns from solver.
    :type essentiality_series: pd.Series
    :param ignore_na: Whether to ignore NA values. When False, will return True if all values are True and none are NA,
        will return False when any values are False, will return NA if all non-NA values are True
    :return: True if gene is considered essential by this strategy, False if gene is considered non-essential, and NA
        if the gene essentiality can't be determined by this strategy
    :rtype: bool
    """
    if not ignore_na:
        res = np.all(essentiality_series)
        if not np.any(pd.isna(essentiality_series)):
            return res
        if np.any(essentiality_series.eq(False)):
            return False
        return pd.NA
    return np.all(essentiality_series)


# Note: Ties will be in False's favor
def aggstrat_majority(essentiality_series: pd.Series, ignore_na: bool = False):
    """
    Aggregation strategy function, where a gene is considered essential if the majority of context specific models
    predict it to be essential

    :param essentiality_series:
    :param ignore_na:
    :return:
    """
    essentiality_series = essentiality_series.astype("Int64")
    counts = essentiality_series.value_counts(dropna=False)
    if 0 not in counts:
        counts[0] = 0
    if 1 not in counts:
        counts[1] = 0
    if pd.NA not in counts:
        counts[pd.NA] = 0
    counts = counts.rename({0: False, 1: True})
    if ignore_na:
        return counts[True] > counts[False]
    if pd.NA not in counts:
        counts[pd.NA] = 0
    length = counts.sum()
    if counts[True] > length * 0.5:
        return True
    if counts[False] > length * 0.5:
        return False
    if (counts[False] == counts[True]) and (counts[pd.NA] == 0):
        return False
    return pd.NA

# endregion

def count_bool_vals(series_to_count:pd.Series):
    """
    Helper function for counting a series of boolean values, and making sure that count series has True, False, and
    pd.NA in the index

    :param series_to_count: Boolean series to count values for
    :type series_to_count: pd.Series
    :return: Series of count values, with True, False, and pd.NA as index
    """
    counts = series_to_count.value_counts(dropna=False)
    if 0 not in counts:
        counts[0] = 0
    if 1 not in counts:
        counts[1] = 0
    if pd.NA not in counts:
        counts[pd.NA] = 0
    return counts.rename({0: False, 1: True}).astype("boolean")