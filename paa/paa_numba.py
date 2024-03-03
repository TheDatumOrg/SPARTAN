import pandas as pd
import numpy as np

from numba import njit, prange

@njit(parallel=True,cache=True)
def paa_whole_dataset(X,num_intervals):
    num_atts = X.shape[2]
    num_win = X.shape[1]
    num_insts = X.shape[0]
    paa_all = np.zeros((num_insts,num_win,num_intervals))
    for i in prange(num_insts):
        x = X[i,:]
        paa = _perform_paa_along_dim_numba(x,num_intervals)
        paa_all[i,:,:] = paa
    return paa_all

@njit(cache=True)
def _perform_paa_along_dim_numba(X,num_intervals):
    # X = from_nested_to_2d_array(X, return_numpy=True)
    num_atts = X.shape[1]
    num_insts = X.shape[0]
    # dims = pd.DataFrame()
    # paa_output = np.zeros((num_insts))
    data = np.zeros((num_insts,num_intervals))

    for i in range(num_insts):
        series = X[i,:]

        frames = []
        current_frame = 0
        current_frame_size = 0
        frame_length = num_atts / num_intervals
        frame_sum = 0

        # if num_atts % num_intervals == 0:
        if True:
            series_split = np.array_split(series,num_intervals)
            frames = [np.mean(interval) for interval in series_split]
        else:
            # frames = np.zeros(self.num_intervals)
            # space_size = np.arange(0,num_atts * self.num_intervals - 1)
            # output_index = space_size // num_atts
            # input_index = space_size // self.num_intervals

            # uniques,n_uniques = np.unique(output_index, return_counts=True)
            # frames = [series[indices].sum() / num_atts for indices in
            #    np.split(input_index, n_uniques.cumsum())[:-1]]

            for n in range(num_atts):
                remaining = frame_length - current_frame_size

                if remaining > 1:
                    frame_sum += series[n]
                    current_frame_size += 1
                else:
                    frame_sum += remaining * series[n]
                    current_frame_size += remaining

                if current_frame_size == frame_length:
                    frames.append(frame_sum / frame_length)
                    current_frame += 1

                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = 1 - remaining

            # if the last frame was lost due to double imprecision
            if current_frame == num_intervals - 1:
                frames.append(frame_sum / frame_length)
        data[i,:] = np.array(frames)

    return data