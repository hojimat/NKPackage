
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.stats import norm
from .bitstrings import random_binary_matrix


def interaction_matrix(N:int, K:int, shape:str="roll") -> NDArray[np.int8]:
    """Creates an interaction matrix for a given K

    Args:
        N: Number of bits
        K: Level of interactions
        shape: Shape of interactions. Takes values 'roll' (default), 'random', 'diag'.

    Returns:
        An NxN numpy array with diagonal values equal to 1 and rowSums=colSums=K+1
    """

    output = None
    if K == 0:
        output = np.eye(N,dtype=int)
    elif shape=="diag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range((-K),(K+1))]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif shape=="updiag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range(0,(K+1))]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif shape=="downdiag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range((-K),1)]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif shape=="sqdiag":
        tmp = np.eye(N,dtype=int)
        tmp = tmp.repeat(K+1,axis=0)
        tmp = tmp.repeat(K+1,axis=1)
        output = tmp[0:N,0:N]
    elif shape=="roll":
        tmp = [1]*(K+1) + [0]*(N-K-1)
        tmp = [np.roll(tmp,z) for z in range(N)]
        tmp = np.array(tmp)
        output = tmp.transpose()
    elif shape=="random":
        output = random_binary_matrix(N,K+1,1)
    elif shape=="chess":
        print(f"Uncrecognized interaction type '{type}'")
    # print(f"Interaction shape '{shape}' selected")
    return output


###############################################################################


def generate_landscape(p: int, n: int, k: int, c: int, s: int, rho: float) -> NDArray[np.float32] :
    """
    Defines a matrix of performance contributions for given parameters.
    This is a matrix with N*P columns each corresponding to a bit,
    and 2**(1+K+C*S) rows each corresponding to a possible bitstring.

    Args:
        p: Number of landscapes (population size)
        n: Number of tasks per landscape
        k: Number of internal bits interacting with each bit
        c: Number of external bits interacting with each bit
        s: Number of landscapes considered for external bits
        rho: Correlation coefficient between landscapes

    Returns:
        An (N*P)x2**(1+K+C*S) matrix of P contribution matrices with correlation rho
    """

    corrmat = np.repeat(rho,p*p).reshape(p,p) + (1-rho) * np.eye(p)
    corrmat = 2*np.sin((np.pi / 6 ) * corrmat)
    base_matrix = np.random.multivariate_normal(mean=[0]*p, cov=corrmat, size=(n*2**(1+k+c*s)))
    cdf_matrix = norm.cdf(base_matrix)
    landscape = np.reshape(cdf_matrix.T, (n*p, (2**(1+k+c*s)))).T
    
    return landscape


@njit
def calculate_performances(bstring: NDArray[np.int8], imat: NDArray[np.int8], cmat: NDArray[np.float32], n: int, p: int) -> NDArray[np.float32]:
    """
    Computes a performance of a bitstring given contribution matrix (landscape) and interaction matrix

    Notes:
        Uses Numba's njit compiler, so the advanced numpy operations such as np.mean(axis=1)
        are not supported. That is why the code might seem to be dumbed down. But njit
        speeds up any list comprehensions or numpy tricks by 4 times at least
        in this particular case.

    Args:
        x : An input vector
        imat: Interaction matrix
        cmat: Contribution matrix (landscape)
        n: Number of tasks per landscape
        p: Number of landscapes (population size)

    Returns:
        A list of P performances for P agents.

    """

    # get performance contributions for every bit:
    phi = np.zeros(n*p)
    for i in range(n*p):
        # subset only coupled bits, i.e. where
        # interaction matrix is not zero:
        coupled_bits = bstring[np.where(imat[:,i]>0)]

        # convert coupled_bits to decimal. this long weird function 
        # does exactly that very fast and can be jit-compiled,
        # unlike a more straightforward function. This is equivalent to
        # the function nk.bin2dec but is inserted here to avoid jit-ing it.
        bin_to_dec = sum(coupled_bits * 2**(np.arange(coupled_bits.size)[::-1]))

        # performance contribution of x[i]:
        phi[i] = cmat[bin_to_dec, i] 

    # get agents' performances by averaging their bits'
    # performances, thus getting vector of P mean performances
    Phis = np.zeros(p, dtype=np.float32)
    for i in range(p):
        Phis[i] = phi[n*i : n*(i+1)].mean()
        
    return Phis

@njit
def get_globalmax(imat: NDArray[np.int8], cmat: NDArray[np.float32], n: int, p: int) -> float:
    """
    Calculate global maximum by calculating performance for every single bit string.
    There is a reason for why it does not save every performance 
    somewhere, so that we can have a giant lookup table and never have to 
    calculate performances ever again, however, the performances are float32 (4 Bytes),
    which means that for 5 agents with 4 tasks each we have (2^20)*4 = 

    Notes:
        Uses Numba's njit compiler.

    Args:
        imat: Interaction matrix
        cmat: Contribution matrix (landscape)
        n: Number of tasks per landscape
        p: Number of landscapes (population size)
    
    Returns:
        The float value with the maximum performance (sum of performance contributions phi[i])

    """

    max_performance = 0.0

    for i in range(2 ** (n*p) ):
        # convert the decimal number i to binary.
        # this long weird function does exactly that very fast 
        # and can be jit-compiled, unlike a more straightforward function.
        # This is equivalent to nk.dec2bin but is inserted here to avoid jit-ing it.
        dec_to_bin = ( (i // 2**np.arange(n*p)[::-1]) % 2 ).astype(np.int8)

        # calculate performances for p agents
        phis = calculate_performances(dec_to_bin, imat, cmat, n, p)

        # find global max for aggregate performance
        if sum(phis) > max_performance:
            max_performance = sum(phis)

    return max_performance

@njit
def calculate_all_performances(imat: NDArray[np.int8], cmat: NDArray[np.float32], n: int, p: int) -> tuple[NDArray[np.float32], float]:
    """
    Calculate global maximum by calculating performance for every single bit string.
    There is a reason for why it does not save every performance 
    somewhere, so that we can have a giant lookup table and never have to 
    calculate performances ever again, however, the performances are float32 (4 Bytes),
    which means that for 5 agents with 4 tasks each we have (2^20)*4 = 

    Notes:
        Uses Numba's njit compiler.

    Args:
        imat: Interaction matrix
        cmat: Contribution matrix (landscape)
        n: Number of tasks per landscape
        p: Number of landscapes (population size)
    
    Returns:
        The p x (2^n*p) numpy array of each agent's performance for each full-sized bit string

    """
    
    max_performance = 0.0
    performances = np.empty((2**(n*p), p), dtype=np.float32)

    for i in range(2 ** (n*p)):
        # convert the decimal number i to binary.
        # this long weird function does exactly that very fast 
        # and can be jit-compiled, unlike a more straightforward function.
        # This is equivalent to nk.dec2bin but is inserted here to avoid jit-ing it.
        dec_to_bin = ( (i // 2**np.arange(n*p)[::-1]) % 2 ).astype(np.int8)

        # calculate performances for p agents
        phis = calculate_performances(dec_to_bin, imat, cmat, n, p)
        performances[i,:] = phis

        # find global max for aggregate performance
        if sum(phis)/p > max_performance:
            max_performance = sum(phis)/p

    return performances, max_performance