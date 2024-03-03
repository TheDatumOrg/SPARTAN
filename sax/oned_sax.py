# Adapted from tslearn
"""1d-SAX : a Novel Symbolic Representation for Time Series"""

import sys

import numpy as np
import pandas as pd
import scipy.stats

from paa.paa_esax import PAAESAX
from paa.paa import PAA
from tslearn.piecewise import OneD_SymbolicAggregateApproximation 

class OneDSAX():
    """1d-SAX : a Novel Symbolic Representation for Time Series.


    Parameters
    ----------
    word_length:         int, length of word to shorten window to (using
    PAA) (default 1)
    alphabet_size:       int, number of values to discretise each value
    to (default to 5)
    window_size:         int, size of window for sliding. Input series
    length for whole series transform (default to 12)
    remove_repeat_words: boolean, whether to use numerosity reduction (
    default False)
    save_words:          boolean, whether to use numerosity reduction (
    default False)

    return_pandas_data_series:          boolean, default = True
        set to true to return Pandas Series as a result of transform.
        setting to true reduces speed significantly but is required for
        automatic test.

    Attributes
    ----------
    words:      history = []
    """
    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        alphabet_size_slope=4,
        window_size=12,
        remove_repeat_words=False,
        save_words=False,
        return_pandas_data_series=True,
    ):
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.alphabet_size_slope = alphabet_size_slope
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.return_pandas_data_series = return_pandas_data_series
        self.words = []
        self.bp_words = []

        self.word_length = int(word_length / 2)

        self.sax_model = None

    def transform(self, X, y=None):
        """Transform data.

        Parameters
        ----------
        X : 2d numpy array [N_instances,N_timepoints]

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """

        n_instances, dim = X.shape
        # print(self.word_length, self.alphabet_size)

        # X = scipy.stats.zscore(X,axis=1)
        if self.sax_model == None:

            one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=self.word_length,
                                alphabet_size_avg=self.alphabet_size, 
                                alphabet_size_slope=self.alphabet_size_slope, 
                                sigma_l=None)

            self.sax_model = one_d_sax
            
            # print(self.word_length, self.alphabet_size)
            bp_words = self.sax_model.fit_transform(np.expand_dims(X, axis=2)) # np.expand_dims(X, axis=2)

        else:
            bp_words = self.sax_model.transform(np.expand_dims(X, axis=2))
            
        # bp_words = one_d_sax.fit_transform(X)
        bp_words = bp_words.reshape(n_instances, -1)
        bp_words = np.expand_dims(bp_words, axis=1)

        self.bp_words = bp_words
        self.histogram = None
        self.breakpoints = self.sax_model.breakpoints_avg_
        # print(self.bp_words.shape)
        # print(type(bp_words))

        
        return bp_words



# if __name__ == "__main__":

#     sax = OneDSAX(word_length=6, alphabet_size=2, alphabet_size_slope=2,)

#     X = np.array([[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]])
#     Y = np.array([[0., 0., 0.2, 0., 2., -1.], [-0.5, -0.1, 0., -3., 4, -1.]])
    
#     print(X.shape)
#     sax.transform(X)
#     train_bp = sax.bp_words
#     train_bp = np.squeeze(train_bp, axis=1).reshape(train_bp.shape[0], -1, 2)

#     print(train_bp)
#     sax.transform(Y)
#     test_bp = sax.bp_words
#     test_bp = np.squeeze(test_bp, axis=1).reshape(test_bp.shape[0], -1, 2)
#     print(test_bp)
#     oned_dist_val = sax.sax_model.distance_1d_sax(train_bp[0], test_bp[1])
#     print(oned_dist_val)