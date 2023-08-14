"""
Function for using dexom method to create several models and then
"""
# Standard Library Imports
from functools import reduce
from typing import Union
# External Imports
import cobra
from cobra.flux_analysis import single_gene_deletion
import pandas as pd


# Function will take in cobra model, and find which genes are essential
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
                                    enum_method: str = "diversity", **kwargs):

    pass

