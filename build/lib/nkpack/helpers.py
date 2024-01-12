
from __future__ import annotations
import itertools
from typing import Iterator
import numpy as np

def variate(param_dict: dict[str, list]) -> Iterator[dict]:
    """Creates parameter combinations

    Args:
        param_dict (dict): dictionary of parameters

    Returns:
        list: List of dicts
    """
    param_keys = param_dict.keys()
    value_combinations = itertools.product(*param_dict.values())
    output = (dict(zip(param_keys, i)) for i in value_combinations)
    return output

def pick(vec, count):
    """Picks random elements from a vector

    Args:
        vec (numpy.ndarray): An input vector
        count (int): Number of elements to pick

    Returns:
        numpy.ndarray: A vector of size count
    """

    tmp = np.random.choice( range(vec), count, replace=False)
    return np.sort(tmp)


def update_values(A, m, i, j, Val):
    """Updates the square subset of the matrix in place

    Args:
        A (numpy.ndarray): a matrix to be updated
        m (int): size of a square matrix
        i (int): i\'th row-wise
        j (int): j\'th column-wise
        Val (numpy.ndarray): values set to the subset
    """
    A[i*m : (i+1)*m, j*m : (j+1)*m] = Val


def flatten(x):
    """Converts a list of lists into a flat list

    Args:
        x (list): An input list

    Returns:
        list: A flat list
    """
    
    return [_ for z in x for _ in z]

def with_noise(vec,pb=0):
    """Adds a noise to a vector

    Args:
        vec (list or numpy.ndarray): An input vector
        pb (float): Probability of error/noise. Takes values between 0 (default) and 1.

    Returns:
        numpy.ndarray: A vector of same size as vec but with errors
    """

    if vec.size==0:
        return vec
    tmp = vec
    nvec = len(vec)
    noise = np.random.choice(2,p=[1-pb, pb])
    if noise:
        tmp = random_neighbour(vec,0,nvec)
    output = tmp
    return output