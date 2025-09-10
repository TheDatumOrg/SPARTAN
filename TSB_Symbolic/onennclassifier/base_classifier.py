import numpy as np
from ..util.distance_vectorized import euclidean_vectorized


class BaseClassifier:
    
    def __init__(self,
                 metric='euclidean'
                 ):
        self.metric=metric


    def fit(self,X,y=None):
        
        self._X = X
        self._y = y
        

    def predict(self,X):
        
        self.test_X = X
        if self.metric.lower() in ['hist_euclidean', 'euclidean']:
            dist_mat =  euclidean_vectorized(self.test_X, self._X) # input shape: (Batch_size, ts_len)
            print(f"Euclidean + 1NN | Train: {self._X.shape} | Test: {self.test_X.shape}")
        self.dist_mat = dist_mat
        ind = np.argmin(dist_mat,axis=1)
        ind = ind.T
        pred = self._y[ind]

        return pred