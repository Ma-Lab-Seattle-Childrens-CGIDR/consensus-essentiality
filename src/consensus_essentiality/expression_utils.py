"""
Functions for converting between different gene expression units
"""
# Standard library imports
from warnings import warn

# External imports
import pandas as pd


def count_to_rpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Normalize raw count data using RPKM

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: RPKM normalized counts
    :rtype: pd.DataFrame
    """
    # Ensure that the count data frame and feature length series have the same genes
    count_genes = set(count.columns)
    fl_genes = set(feature_length.index)
    if not (count_genes == fl_genes):
        warn("Different genes in count dataframe and feature length series, dropping any not in common")
        genes = count_genes.intersection(fl_genes)
        count = count[genes]
        feature_length = feature_length[genes]
    sum_counts = count.sum(axis=1)
    return count.divide(feature_length, axis=1).divide(sum_counts, axis=0) * 1.e9


def count_to_fpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to FPKM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows. Specifically,
        the count data represents the number of fragments, where a fragment corresponds to a single cDNA molecule, which
        can be represented by a pair of reads from each end.
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: FPKM normalized counts
    :rtype: pd.DataFrame
    """
    return count_to_rpkm(count, feature_length)


def count_to_tpm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to TPM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    # Ensure that the count data frame and feature length series have the same genes
    count_genes = set(count.columns)
    fl_genes = set(feature_length.index)
    if not (count_genes == fl_genes):
        warn("Different genes in count dataframe and feature length series, dropping any not in common")
        genes = count_genes.intersection(fl_genes)
        count = count[genes]
        feature_length = feature_length[genes]
    length_normalized = count.divide(feature_length, axis=1)
    return length_normalized.divide(length_normalized.sum(axis=1), axis=0) * 1.e6


def count_to_cpm(count: pd.DataFrame) -> pd.DataFrame:
    """
    Converts count data to counts per million

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :return: CPM normalized counts
    :rtype: pd.DataFrame
    """
    total_reads = count.sum(axis=1)
    per_mil_scale = total_reads/1e6
    return count.divide(per_mil_scale, axis=0)



def rpkm_to_tpm(rpkm: pd.DataFrame):
    """
    Convert RPKM normalized counts to TPM normalized counts

    :param rpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type rpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm.divide(rpkm.sum(axis=1), axis=0) * 1.e6


def fpkm_to_tpm(fpkm: pd.DataFrame):
    """
    Convert FPKM normalized counts to TPM normalized counts

    :param fpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type fpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm_to_tpm(fpkm)
