"""
Utils for handling gene expression data, specifically formatting for input into the imat functions
"""
# region imports
# Standard library imports
from typing import Callable, Union
# External Imports
import cobra
import numpy as np
import pandas as pd
from dexom_python import expression2qualitative


# endregion

def expression_to_weights(gene_expression: pd.DataFrame, proportion: Union[float, tuple[float, float]],
                          gene_axis: int = 1, agg_fun:Callable[[pd.Series], Union[float, int]] = np.median)->pd.Series:
    """
    Function to convert gene expression data into qualitative gene weights (for use in imat and enumeration methods)

    :param gene_expression: Gene expression data, can be a Dataframe with genes as one index and samples as the other,
        or a Series with genes as the index.
    :type gene_expression: Union[pd.DataFrame, pd.Series]
    :param proportion: Proportion to use for trinarizing the data. If a float between 0 and 1, values below that
        percentile will be considered low expression, values whose percentiles are
        above 1-proportion will be considered high expression. If a tuple, the first element will be used as the low
        expression percentile threshold, and the second element will be used as the high expression percentile
        threshold.
    :type proportion: Union[float, tuple[float,float]]
    :param gene_axis: Which axis represents genes in the gene_expression parameter, with 0 being rows and 1 being
        columns
    :type gene_axis: int
    :param agg_fun: Which function to use for combining multiple expression values into 1 which can then be converted
        into the qualitative reaction weight.
    :type agg_fun: Callable[[pd.Series], Union[float,int]]
    :return: Reaction weights
    :rtype: pd.Series
    """
    # If the genes are columns, transpose the dataframe so that they are now
    if gene_axis == 1:
        gene_expression = gene_expression.transpose()
    # If there are multiple columns, they need to be aggregated
    if (not (type(gene_expression) == pd.Series)) and gene_expression.shape[1] > 1:
        gene_expression = gene_expression.aggregate(agg_fun, axis=1)
    # Convert the one column dataframe into a pandas series
    gene_expression = gene_expression.squeeze()
    qual_gene_expression = expression2qualitative(gene_expression, column_list=None, proportion=proportion,
                                                  significant_genes='both', save=False)
    return qual_gene_expression
