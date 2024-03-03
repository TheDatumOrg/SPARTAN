import os
import multiprocessing

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def initialise_multithread(num_cores=-1):
    """
    Initialise pool workers for multi processing
    :param num_cores:
    :return:
    """
    if (num_cores == -1) or (num_cores >= multiprocessing.cpu_count()):
        num_cores = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(num_cores)
    return p


def create_directory(directory_path):
    """
    Create a directory if path doesn't exists
    :param directory_path:
    :return:
    """
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path
    
def compute_classification_metrics(y_true,y_pred,y_true_val=None,y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float32), index=[0],
                       columns=['acc', 'precision','recall','f1'])
    res['acc'] = accuracy_score(y_true,y_pred)
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['f1'] = f1_score(y_true,y_pred,average='macro')
    
    return res

# 128 UCR univariate time series classification problems [1]
univariate = {
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
}

def permutations_w_constraints(n_perm_elements, sum_total, min_value, max_value):
    # base case
    if n_perm_elements == 1:
        if (sum_total <= max_value) & (sum_total >= min_value):
            yield (sum_total,)
    else:
        for value in range(min_value, max_value + 1):
            for permutation in permutations_w_constraints(
                n_perm_elements - 1, sum_total - value, min_value, max_value
            ):
                if value >= permutation[0]:
                    yield (value,) + permutation

def entropy(signal, prob="standard"):
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)

    Returns
    -------
    float
        The normalized entropy value

    """

    if prob == "standard":
        value, counts = np.unique(signal, return_counts=True)
        p = counts / counts.sum()
    elif prob == "kde":
        p = kde(signal)
    elif prob == "gauss":
        p = gaussian(signal)

    if np.sum(p) == 0:
        return 0.0

    # Handling zero probability values
    p = p[np.where(p != 0)]

    # If probability all in one value, there is no entropy
    if np.log2(len(signal)) == 1:
        return 0.0
    elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return -np.sum(p * np.log2(p)) / np.log2(len(signal))


def slope(X):

    max_pt = np.argmax(X)
    min_pt = np.argmin(X)

    max_val = X[max_pt]
    min_val = X[min_pt]

    return (max_val - min_val) / (max_pt - min_pt)



def dynamic_bit_allocation(total_bit, EV, min_bit, max_bit):

    K = len(EV)
    N = total_bit
    DP = np.zeros((K+1,N+1))
    alloc = np.zeros_like(DP).astype(np.int32) # store the num of bits for each component

    # init
    for i in range(0, K+1):
        for j in range(0, N+1):
            
            DP[i][j] = -1e9

    DP[0][0] = 0
    
    # non-recursive
    for i in range(1, K+1):
        for j in range(0, N+1):
            
            max_reward = -1e9

            for x in range(min_bit, min(max_bit, j)+1):
                
                current_reward = DP[i-1][j-x]+x*EV[i-1]
                
                if current_reward > max_reward:

                    alloc[i][j] = x
                    max_reward = current_reward
                    DP[i][j] = current_reward

    
    def print_sol(alloc, K, N):
        
        bit_arr = []  
        unused_bit = N
        for i in range(K, 1, -1):
            bit_arr.append(alloc[i][unused_bit])
            unused_bit -= alloc[i][unused_bit]

        bit_arr.append(unused_bit)
        return bit_arr
    
    bit_arr = print_sol(alloc, K, N)

    return DP[K][N], bit_arr[::-1]


def dynamic_bit_allocation_update(total_bit, EV, min_bit, max_bit, delta=0.5):

    def ScaleFactor(x, A, delta=0.5):
        return 1 - delta * max(0, (x - A) / A)

    K = len(EV)
    N = total_bit
    A = N/K
    DP = np.zeros((K+1,N+1))
    alloc = np.zeros_like(DP).astype(np.int32) # store the num of bits for each component

    # init
    for i in range(0, K+1):
        for j in range(0, N+1):
            
            DP[i][j] = -1e9

    DP[0][0] = 0
    
    # non-recursive
    for i in range(1, K+1):
        for j in range(0, N+1):
            
            max_reward = -1e9

            for x in range(min_bit, min(max_bit, j)+1):
                
                current_reward = DP[i-1][j-x]+x*EV[i-1]*ScaleFactor(x,A,delta)
                
                if current_reward > max_reward:

                    alloc[i][j] = x
                    max_reward = current_reward
                    DP[i][j] = current_reward

    
    def print_sol(alloc, K, N):
        
        bit_arr = []  
        unused_bit = N
        for i in range(K, 1, -1):
            bit_arr.append(alloc[i][unused_bit])
            unused_bit -= alloc[i][unused_bit]

        bit_arr.append(unused_bit)
        return bit_arr
    
    bit_arr = print_sol(alloc, K, N)

    assert np.sum(bit_arr) == N

    return DP[K][N], bit_arr[::-1]