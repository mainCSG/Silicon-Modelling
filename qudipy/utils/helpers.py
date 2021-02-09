"""
General helper utilities

@author: simba
"""

import numpy as np
import operator as op
from functools import reduce
        
def find_nearest(array, value):
    '''
    Function to find the closest value to a number in a given array.
    If array contains duplicate values of the nearest number, then first
    instance in the array will be returned.
    
    Parameters
    ----------
    array : ND float array-like object
        An ND array type object of floats.
    value : float
        Value which we want to find the closest number to in the array.

    Returns
    -------
    near_idx : tuple of ints
        Tuple of indices for the nearest value in array.
    near_value : float
        Nearest value in array.

    '''
    
    # Convert to numpy array if not already
    array = np.asarray(array)
    
    # Obtain the indices corresponding to which element in array is closest 
    # to value
    near_idx = np.unravel_index((np.abs(array - value)).argmin(), array.shape)
        
    # Return also the nearest value
    near_value = array[near_idx]
    
    return near_idx, near_value

def nchoosek(n, k):
    '''
    Had a lot of trouble getting scipy comb to work, so just copied one from 
    stack exchange. This function calculates the number of combinations of k 
    elements from a set of n items.

    Parameters
    ----------
    n : int
        Total number of items in set.
    k : int
        Size of subset to make out of the full set.

    Returns
    -------
    nCr
        The number of possible combinations.

    '''
    
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return numer // denom


