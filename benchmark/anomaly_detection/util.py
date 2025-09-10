import numpy as np
import pandas as pd
import math
import warnings
from builtins import range
from numpy.linalg import LinAlgError

with warnings.catch_warnings():
    # Ignore warnings of the patsy package
    warnings.simplefilter("ignore", DeprecationWarning)

class Window:
    """ The  class for rolling window feature mapping.
    The mapping converts the original timeseries X into a matrix. 
    The matrix consists of rows of sliding windows of original X. 
    """

    def __init__(self,  window = 100, stride = 1):
        self.window = window
        self.stride = stride
        self.detector = None
    def convert(self, X):
        n = self.window
        X = X.squeeze()     # (len,)
        X = pd.Series(X)
        L = []
        if n == 0:
            df = X
        else:
            for i in range(0, n*self.stride, self.stride):
                L.append(X.shift(i))
            df = pd.concat(L, axis = 1)
            df = df.iloc[(n-1)*self.stride:]
        return df