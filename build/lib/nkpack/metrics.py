import numpy as np
from numpy.typing import NDArray

def cobb_douglas(weights,vec):
    """A Cobb-Douglas utility function with given weights

    Args:
        weights (list): Weights
        vec (numpy.ndarray): An input vector

    Returns:
        float: A Cobb-Douglas value
    """

    x = [z+1 for z in vec]
    w = weights
    tmp = np.power(x,w).prod()
    output = tmp
    return output

def satisfice(x,y,u):
    """A satisficing utility

    Args:
        x (float): first value
        y (float): second value
        u (float): first goal

    Returns:
        float: Utility value
    """

    cond = x >= u
    tmp = cond * (y + u) + (1-cond) * x
    output = tmp
    return output

def weighted(x,y,p1,p2):
    """A weighted sum

    Args:
        x (float): first value
        y (float): second value
        p1 (float): first weight
        p2 (float): second weight

    Returns:
        float: A weighted sum
    """

    tmp = p1*x + p2*y
    output = tmp
    return output

def gp_score(perfs: NDArray[np.float32], goals: NDArray[np.float32], weights: NDArray[np.float32]) -> float:
    """
    Calculates Goal Programming Objective function value (score)
    for given weights, goals, and performances of interest.

    Args:
        perfs: performances for each goal; shape=1D
        goals: goals themselves; shape=1D
        weights: weights for each goal; shape=1D

    Returns:
        A float value for the GP score
        
    """
    
    deviations = np.maximum(goals-perfs, 0)
    score = np.dot(weights, deviations)
    
    # negative because zero is the best case scenario
    # the less deviations, the better,
    # so we minimize the score actually
    return -score

def calculate_frequency(bstring: NDArray[np.int8], lookup_table: NDArray[np.int8]) -> float:
    """
    Calclates frequency of a bitstring in a (pre-flattened) lookup table of bistrings.
    If sees a row that contains at least one value which is not 0 or 1, simply ignores it.

    Args:
        bstring: 1xNSOC sized array
        lookup_table: (TM*DEG)xNSOC sized array of bstrings

    Returns:
        Frequency of bstring in lookup_table excluding non (0,1) rows

    Example:
        bstring=np.array([1,1])
        lookup_table = np.array([
            [1,1],
            [0,0],
            [0,0]
        ])

        should return 1/3
    """
    
    # ignore rows that contain any value that is not 0 or 1
    nonempty = np.all((lookup_table==0) | (lookup_table==1), axis=(1,2))
    lookup = lookup_table[nonempty]
    # if no social bits are received then return 1.0
    # because this guy doesn't care about others
    if lookup.size == 0:
        return 1.0
    
    return np.mean(lookup==bstring)

def beta_mean(x,y):
    """Calculates the mean of the Beta(x,y) distribution

    Args:
        x (float): alpha (shape) parameter
        y (float): beta (scale) parameter

    Returns:
        float: A float that returns x/(x+y)
    """
    
    return x / (x+y)


def decompose_performances(performances: NDArray[np.float32], agent_id: int) \
    -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Takes individual performances for multiple bit strings
    and returns own performance and mean of other agents'
    performances

    Args:
        performances: AnyxP matrix of floats
        agent_id
    Returns:
        Anyx1 array of own performances and
        Anyx1 array of mean of others' performances

    """

    perf_own = performances[:, agent_id]
    perf_other = (np.sum(performances, axis=1) - perf_own) / (performances.shape[1] - 1)

    return perf_own, perf_other