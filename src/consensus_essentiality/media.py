# Core Imports
from typing import TextIO
# External Imports
import cobra
import pandas as pd


def load_media(model: cobra.Model, media: pd.DataFrame, rxn_col=None, lb_col=None, ub_col=None):
    """
    Function to modify model to reflect bounds in media dataframe
    :param model: Model being updated
    :type model: cobra.Model
    :param media: Dataframe with media bounds, should be 3 columns with a column for reaction id, lower bound,
        and upper bound
    :type media: pd.DataFrame
    :param lb_col: Pandas column label (or other object that can be passed to [], and loc dataframe methods) to
        select the column containing the lower bound for flux
    :type lb_col: str (or other object that can be passed to [], and loc dataframe methods)
    :param ub_col: Pandas column label (or other object that can be passed to [], and loc dataframe methods) to
        select the column containing the upper bound for flux
    :type ub_col: str (or other object that can be passed to [], and loc dataframe methods)
    :param rxn_col: Pandas column label (or other object that can be passed to [], and loc dataframe methods) to
        select the column containing the reaction id
    :type rxn_col: str (or other object that can be passed to [], and loc dataframe methods)
    :return: Updated Model
    :rtype: cobra.Model
    """
    model = model.copy()
    if lb_col is None:
        lb_col = "lower_bound"
    if ub_col is None:
        ub_col = "upper_bound"
    if rxn_col is None:
        rxn_col = "reaction"
    for row in media.index:
        reaction = media.loc[row, rxn_col]
        lb = media.loc[row, lb_col]
        ub = media.loc[row, ub_col]
        if reaction in model.reactions:
            model.reactions.get_by_id(reaction).bounds = (lb, ub)
    return model


def load_media_file(model:cobra.Model, media_file:TextIO, rxn_col:str=None, lb_col:str=None, ub_col:str=None):
    """
    Wrapper around load_media function to load media from a csv file
    :param model: Model being updated
    :type model: cobra.Model
    :param media_file: String or file pointer for csv file containing the media specification, with a column for
        reaction id, flux lower bound, and flux upper bound
    :type media_file: str | file-pointer
    :param lb_col: Pandas column label (or other object that can be passed to [], and loc dataframe methods) to
        select the column containing the lower bound for flux
    :type lb_col: str (or other object that can be passed to [], and loc dataframe methods)
    :param ub_col: Pandas column label (or other object that can be passed to [], and loc dataframe methods) to
        select the column containing the upper bound for flux
    :type ub_col: str (or other object that can be passed to [], and loc dataframe methods)
    :param rxn_col: Pandas column label (or other object that can be passed to [], and loc dataframe methods) to
        select the column containing the reaction id
    :type rxn_col: str (or other object that can be passed to [], and loc dataframe methods)
    :return: Updated Model
    :rtype: cobra.Model
    """
    media = pd.read_csv(media_file)
    return load_media(model, media, rxn_col=rxn_col, lb_col=lb_col, ub_col=ub_col)
