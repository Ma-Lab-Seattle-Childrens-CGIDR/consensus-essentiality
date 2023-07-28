"""
Functions for converting between different gene expression units
"""
import pandas as pd


def count_to_rpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    pass


def count_to_fpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to FPKM normalized expression
    :param count:
    :param feature_length:
    :return:
    """
    return count_to_rpkm(count, feature_length)


def count_to_tpm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    pass


def rpkm_to_tpm(rpkm: pd.DataFrame):
    pass


def fpkm_to_tpm(fpkm: pd.DataFrame):
    """
    Convert FPKM to TPM normalized expression
    :param fpkm:
    :return:
    """
    return rpkm_to_tpm(fpkm)
