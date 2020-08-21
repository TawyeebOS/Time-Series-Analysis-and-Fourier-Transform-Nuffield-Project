#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = []
__all__ = []

import os

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.data_container import nested_to_3d_numpy


def load_data():
    """Load time series classification data from Ford Classification
    Competition at 2008 IEEE World Congress on Computational Intelligence [1].

    Returns
    -------
    X_train : np.array
        Training set features
    X_test : np.array
        Test set features
    y_train : np.array
        Training set target variable
    y_test : np.array
        Test set target variable

    References
    ----------
    [1] http://timeseriesclassification.com/description.php?Dataset=FordA
    """
    # set paths
    DATASET = "FordA"
    PATH = "data/"

    # loads data into nested format
    X_train, y_train = load_from_tsfile_to_dataframe(
        os.path.join(PATH, DATASET + "_TRAIN.ts"))
    X_test, y_test = load_from_tsfile_to_dataframe(
        os.path.join(PATH, DATASET + "_TEST.ts"))

    # convert to 2d array
    X_train = nested_to_3d_numpy(X_train).squeeze(axis=1)
    X_test = nested_to_3d_numpy(X_test).squeeze(axis=1)

    # return data
    return X_train, X_test, y_train, y_test


class FourierTransformer:
    """A Fourier Transformer

    Parameters
    ----------
    y : a pandas dataframe of shape = [n_samples, num_dims]
        The training input samples.

    Returns
    -------
    df: a pandas data frame of shape = [num_intervals, num_dims]
    """

    def __init__(self):
        pass

    def fit(self, y):
        """Fit

        Parameters
        ----------
        y : a pandas dataframe of shape = [n_samples, num_dims]
            The training input samples.

        Returns
        -------
        df: a pandas data frame of shape = [num_intervals, num_dims]
        """
        pass

    def transform(self, y):
        """Transform

        Parameters
        ----------
        y : a pandas dataframe of shape = [n_samples, num_dims]
            The training input samples.

        Returns
        -------
        df: a pandas data frame of shape = [num_intervals, num_dims]
        """
        pass
