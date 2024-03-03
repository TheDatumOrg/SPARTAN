# Adapted from Sktime transformer
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt

class PAAVSeg():
    def __init__(self,
            num_intervals=8
        ):

        self.num_intervals = num_intervals
    
    def set_num_intervals(self,n):
        self.num_intervals = n

    # todo: looks like this just loops over series instances
    # so should be refactored to work on Series directly
    def transform(self, X, y=None):
        """Transform data.

        Parameters
        ----------
        X : nested numpy array of shape [n_instances, n_timepoints]
            Nested dataframe with multivariate time-series in cells.

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero,
              second in column one etc.
        """
        # Get information about the dataframe
        # num_atts = len(X.iloc[0, 0])
        # col_names = X.columns

        # Check the parameters are appropriate
        # self._check_parameters(num_atts)

        # On each dimension, perform PAA
        dataFrames = []
        # for x in col_names:
        dataFrames.append(self._perform_paa_along_dim(X))

        # Combine the dimensions together
        result = pd.concat(dataFrames, axis=1, sort=False)
        #result.columns = col_names

        return result

    def _perform_paa_along_dim(self, X):
        # X = from_nested_to_2d_array(X, return_numpy=True)
        num_atts = X.shape[1]
        num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []

        for i in range(num_insts):
            series = X[i,:]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0
            
            # changing point detection
            # algo = rpt.Dynp(model="l2")
            # algo = rpt.BottomUp(model="l2")
            algo = rpt.Binseg(model='l2')
            # algo = rpt.Window(width=int(0.1*len(series)), model='l2')

            algo.fit(series)
            # result = algo.predict(pen=1)

            try:
                bkps = algo.predict(n_bkps=self.num_intervals-1)
                segments = np.split(series, bkps[:-1])
            except:
                print("Segmentation failed. Even segment is used instead.")
                segments = np.split(series, self.num_intervals)

            if len(segments) != self.num_intervals:
                # print("Segmentation failed. Even segment is used instead.")
                print(bkps)
                segments = np.split(series, self.num_intervals)

            # print(series)
            # print(bkps)
            # print(segments)

            for j in range(len(segments)):

                frames.append(np.mean(segments[j]))
            
            data.append(pd.Series(frames))

            # if i <= 10:
            #     # print(i)
                
            #     seed = np.random.randint(low=0, high=5)
            #     plt.plot(series)
            #     plt.title(f'{np.around(frames, 2)}')
                
            #     for m in range(len(bkps)):
            #         plt.axvline(bkps[m])
            #     plt.savefig(f'./ts_example_{seed}.png')
                
            #     plt.close()
        
        assert len(data[0]) == self.num_intervals
        dims[0] = data

        return dims

    def _check_parameters(self, num_atts):
        """Check parameters of PAA.

        Function for checking the values of parameters inserted into PAA.
        For example, the number of subsequences cannot be larger than the
        time series length.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError(
                    "num_intervals must have the \
                                  value of at least 1"
                )
            if self.num_intervals > num_atts:
                raise ValueError(
                    "num_intervals cannot be higher \
                                  than the time series length."
                )
        else:
            raise TypeError(
                "num_intervals must be an 'int'. Found '"
                + type(self.num_intervals).__name__
                + "' instead."
            )


# if __name__ == "__main__":

#     paa = PAA(8)
#     X = np.arange(100).reshape(2,50)
#     X[1] = X[1] - (np.random.rand(50)*3)
#     result = paa.transform(X, y=None)

    
