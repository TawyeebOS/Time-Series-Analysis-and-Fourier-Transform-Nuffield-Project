#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = []
__all__ = []

import os

import numpy as np
from sktime.utils.data_container import nested_to_3d_numpy
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(split=None):
    """Load time series classification data from Ford Classification
    Competition at 2008 IEEE World Congress on Computational Intelligence [1].

    Parameters
    ----------
    split : str{"train", "test"}, optional (default=None)
        * If "train", returns training set
        * If "test", returns test set
        * If None, returns both concatenated training and test set

    Returns
    -------
    X : np.array
        Features variables
    y : np.array
        Target variables

    References
    ----------
    [1] http://timeseriesclassification.com/description.php?Dataset=FordA
    """
    # check split value for loading training and test set separately
    allowed_splits = ("train", "test")

    # set paths
    DATASET = "FordA"
    PATH = "data/"

    def _load_split(split):
        """Helper function to load split"""
        file = os.path.join(PATH, DATASET + f"_{split.upper()}.ts")
        X, y = load_from_tsfile_to_dataframe(file)
        X = nested_to_3d_numpy(X).squeeze(axis=1)
        return X, y

    # loads data into nested format
    if split in allowed_splits:
        return _load_split(split)

    # if no split is given, load both training and test set
    elif split is None:
        Xs = []
        ys = []
        for split in allowed_splits:
            X, y = _load_split(split)
            Xs.append(X)
            ys.append(y)

        X = np.vstack(Xs)
        y = np.hstack(ys)
        return X, y

    else:
        raise ValueError(f"`split` must be one of {allowed_splits}, "
                         f"but found: {split}")


class FourierTransformer(BaseEstimator, TransformerMixin):
    """A Power-Spectrum Transformer
    """

    def fit(self, X, y=None):
        """Fit

        Parameters
        ----------
        X : array-like

        Returns
        -------
        self : a fitted instance of itself
        """
        pass

    def transform(self, X, y=None):
        """Apply power-spectrum transform over the columns

        Parameters
        ----------
        X : array-like
        
        Returns
        -------
        Xt : array-like
            Power spectrum of X
        """

        # YOUR CODE
        raise NotImplementedError()
    
    def inverse_transform(self, X, y=None):
        raise NotImplementedError()

def powerspectrum(X):
    Xt = np.fft.fft(X)
    return np.abs(Xt) ** 2

