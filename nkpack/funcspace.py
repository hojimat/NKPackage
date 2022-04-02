import numpy as np
from numba import jit,njit
from scipy.stats import norm
from itertools import combinations as comb

###############################################################################

def interaction_matrix(N,K,shape="roll"):
    """Creates an interaction matrix for a given K

    Args:
        N (int): Number of bits
        K (int): Level of interactions
        shape (str): Shape of interactions. Takes values 'roll' (default), 'random', 'diag'.

    Returns:
        numpy.ndarray: an NxN numpy array with diagonal values equal to 1 and rowSums=colSums=K+1
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

def binary_combinations(N,R):
    """Generates all binary vectors with sum R

    Args:
        N (int): Size of a binary vector
        R (int): Sum of elements of a vector

    Returns:
        numpy.ndarray: an (N R)xN numpy array, where rows are vectors of size N and sum of elements R
    """

    tmp = []
    idx = comb(range(N),R)
    for i in idx:
        A = [0]*N
        for j in i:
            A[j] = 1
        tmp.append(A)
    output = np.reshape(tmp,(-1,N))
    return(output)
    
def random_binary_matrix(N,R,diag=None):
    """Generates a random binary square matrix with a given row/col sum

    Args:
        N (int): Number of rows/cols
        R (int): Sum of rows/cols
        diag (int or None): Fixed value for diagonal. Takes values None (default), 0, 1.

    Returns:
        numpy.ndarray: an NxN numpy array
    """

    if N==R:
        tmp = np.ones((N,R),dtype=int)
        return tmp
    elif N<R:
        print("Incorrect binary matrix. Check parameters.")
        return
    # create a minimal 2d matrix of zeros for easier indexing
    tmp = np.zeros(2*N,dtype=int).reshape(2,N) 
    cl = binary_combinations(N,R)
    for i in range(N):
        colsums = np.sum(tmp,0)
        # remove excess ones
        idx = np.empty(0,dtype=int)
        for j in np.where(colsums>=R)[0]:
            k = np.where(cl[:,j]>0)[0]
            idx = np.union1d(idx,k)
        cl = np.delete(cl,idx,0)
        # remove excess zeros
        inx = np.empty(0,dtype=int)
        for j in np.where(colsums+N-i == R)[0]:
            k = np.where(cl[:,j]==0)[0]
            inx = np.union1d(inx,k)
        cl = np.delete(cl,inx,0)
        # temporarily ignore diagonal 1s or 0s 
        cli = cl.copy()
        if diag is not None:
            ivx = np.where(cl[:,i]==diag)[0]
            cli = cl[ivx,]
            clk = (cli + colsums)[:,i+1:]
            tp = N-i-1 if diag==0 else 0 # tuning parameter
            ikx = np.where(clk+tp==R)[0]
            cli = np.delete(cli,ikx,0)

        ncl = cl.shape[0]
        ncli = cli.shape[0]
        if ncli > 0:
            ind = np.random.choice(ncli)
            tmp = np.vstack((tmp,cli[ind,:]))
        elif ncli==0 and ncl>0:
            print('Error creating non-zero diagonals. Rerun the function')
            return 0
        else:
            print('Incorrect binary matrix. Check the dimensions.')
            return 
    output = np.delete(tmp,[0,1],0) # remove first 2 empty rows created above
    return(output)

###############################################################################
def binx(x,size=4,out=None):
    """Converts values to binary and back

    Args:
        x: Input value (can be any type)
        size (int): Desired output vector size (adds leading zeros if the output size is less than desired, ignores otherwise)
        out (str or None): Specifies output type. Is ignored at the moment.

    Returns:
        list: A list of 0s and 1s if x is int.
        int: A decimal equivalent if x is str, numpy.ndarray, list.
    """

    tmp = x
    if type(tmp) is int:
        tmp = np.binary_repr(tmp,size)
        tmp = [int(z) for z in tmp]
    elif type(tmp) is str:
        tmp = int(tmp,2)
    elif type(tmp) is np.ndarray or type(tmp) is list:
        tmp = np.sum(np.flip(tmp) * 2 ** (np.arange(len(tmp))))
    else:
        print("incorrect input for function binx")
    return tmp

def contrib_define(p,n,k,c,s,rho):
    """Defines a contribution matrix for given parameters

    Args:
        p (int): Number of landscapes (population size)
        n (int): Number of tasks per landscape
        k (int): Number of internal bits interacting with each bit
        c (int): Number of external bits interacting with each bit
        s (int): Number of landscapes considered for external bits
        rho (float): Correlation coefficient between landscapes

    Returns:
        numpy.ndarray: An (N*P)x2**(1+K+C*S) matrix of P contribution matrices with correlation RHO
    """

    #output = np.random.uniform(0,1,(2**(1+k+c*s),n*p))
    corrmat = np.repeat(rho,p*p).reshape(p,p) + (1-rho) * np.eye(p)
    corrmat = 2*np.sin((np.pi / 6 ) * corrmat)
    tmp = np.random.multivariate_normal(mean=[0]*p,cov=corrmat,size=(n*2**(1+k+c*s)))
    tmp = norm.cdf(tmp)
    tmp = np.reshape(tmp.T,(n*p,(2**(1+k+c*s)))).T
    output = tmp
    return output

@njit
def contrib_solve(x,imat,cmat,n,p):
    """Computes a performance for vector x from given contribution matrix and interaction matrix

    Notes:
        Uses Numba's njit compiler.

    Args:
        x: An input vector
        imat (numpy.ndarray): Interaction matrix
        cmat (numpy.ndarray): Contribution matrix
        n (int): Number of tasks per landscape
        p (int): Number of landscapes (population size)

    Returns:
        float: A mean performance of an input vector x given cmat and imat.
    """
    
    n_p = n*p
    phi = np.zeros(n_p)
    for i in range(n_p):
        tmp = x[np.where(imat[:,i]>0)] # coupled bits
        tmp_loc = np.sum(np.flip(tmp) * 2 ** (np.arange(len(tmp)))) # convert to integer
        phi[i] = cmat[tmp_loc,i]

    output = [0.0]*p
    for i in range(p):
        output[i] = phi[i*n : (i+1)*n].mean()
    return output

def contrib_full(imat,cmat,n,p):
    """Computes performances for all binary vectors of size N*P

    Notes:
        The most processing-heavy part of any simulation. get_global_max() is a 'lazy' alternative

    Args:
        imat (numpy.ndarray): Interaction matrix
        cmat (numpy.ndarray): Contribution matrix
        n (int): Number of tasks per landscape
        p (int): Number of landscapes (population size)

    Returns:
        numpy.ndarray: Array of performances for all vectors of size N*P, normalized by the global maximum
        float: the global maximum value
    """

    n_p = n*p
    perfmat = np.empty((2**n_p,p),dtype=float)
    for i in range(2**n_p):
        bstring = np.array(binx(i,n_p))
        bval = contrib_solve(bstring,imat,cmat,n,p) # !!! processing heavy !!!
        perfmat[i,] = bval
    idxmax = np.argmax(np.mean(perfmat,1))
    perfmax = perfmat[idxmax] 
    
    perfmean = np.mean(perfmat,axis=1)
    perfglobalmax = perfmean.max()
    #perfargmax = perfmean.argmax() 

    output1 = perfmat / perfmax
    #output2 = perfargmax
    #output3 = perfglobalmax
    output3 = perfmax
    return output1, output3#, output2, output3

def get_globalmax(imat,cmat,n,p,brute=True,t0=20,t1=0,alpha=0.1,kk=1):
    """Estimates a global maximum for a landscape using Simulated Annealing algorithm

    Args:
        imat (numpy.ndarray): Interaction matrix
        cmat (numpy.ndarray): Contribution matrix
        n (int): Number of tasks per landscape
        p (int): Number of landscapes (population size)
        t0 (float): Initial temperature. Suggested values are 20 (default) or 1.
        t1 (float): Final temperature. Suggested value is 0 (default)
        alpha (float): Step for temperature. Suggested value are 0.1 (default) or 0.001
        kk (float): Adjustment parameter. Not used at the moment.

    Returns:
        numpy.ndarray: The estimates of the global maximum for each of the P landscapes
    """

    n_p = n*p
    output = None
   
    if brute:
        bstrings = map(lambda x: np.array(binx(x,n_p)), range(2**n_p))
 
        perfmax = [0.0]*p#np.zeros(p,dtype=float)
        for i in bstrings:#range(2**n_p):
            bstring = i
            bval = contrib_solve(bstring,imat,cmat,n,p) # !!! processing heavy !!!
            if np.mean(bval)>np.mean(perfmax):
                perfmax = bval
 
        output = np.array(perfmax)
    else:
        state = np.array(binx(0,n_p))
        value = contrib_solve(state,imat,cmat,n,p)
        
        t = t0
        while t>t1:
            state_ = random_neighbour(state,0,n_p)
            value_ = contrib_solve(state_,imat,cmat,n,p)
            value_ = np.array(value_)
            
            d_mean = np.mean(value_) - np.mean(value)
            #d_sep = value_ - value

            if d_mean > 0 or np.exp(d_mean/t) > np.random.uniform():
                state = state_
                value = value_
            t -= alpha

        output = value
    return output
###############################################################################

def assign_tasks(N,POP,shape="solo"):
    """Assigns N tasks to POP agents

    Args:
        N (int): Number of tasks 
        POP (int): Number of agents (population size)
        shape (str): Type of allocation. Takes values 'solo' (default) and 'overlap' (not used at the moment)

    Returns:
        numpy.ndarray: Returns a POPxN matrix where rows represent agents and cols represent tasks
    """

    output = None
    if shape=="solo":
        perCapita = N / POP
        tmp = np.eye(POP,dtype=int)
        tmp = tmp.repeat(perCapita,axis=1)
        output = tmp
    else:
        print("Task assignment shape unrecognized")
    # print(f"Assignment shape {shape} selected")
    return output

###############################################################################

def generate_network(POP,S=2,pcom=1.0,shape="random",absval=False):
    """Generates a unidirectional network topology

    Args:
        POP (int): Number of agents (population size)
        S (int): Network degree
        pcom (float): Probability of communicating through the channel
        shape (str): Network topology. Takes values 'random' (default), 'ring', 'cycle', 'line', 'star'
        absval (bool): Indexing convention. If True, converts negative indices to positive.

    Returns:
        numpy.ndarray: A POPxPOP matrix with probabilities of connecting to other agents
    """

    if S>=POP:
        print("Error: wrong network degree")
        return 0
    if pcom>1 or pcom<0:
        print("Error: wrong probability of communication")
        return 0
    output = None
    if S == 0:
        output = []
    elif shape=="cycle":
        tmp = np.eye(POP)
        tmp = np.vstack((tmp[1:,:],tmp[0,:]))
        #tmp = [[z-1] + [_%POP for _ in range(z+1,z+S)] for z in range(POP)]
        #if absval==True:
        #    tmp = [[(z-1)%POP] + [_%POP for _ in range(z+1,z+S)] for z in range(POP)]
        output = tmp * pcom
    elif shape == "line":
        tmp = np.eye(POP)
        tmp = np.vstack((tmp[1:,:],np.zeros(POP)))
        output = tmp * pcom
    elif shape == "random":
        tmp = random_binary_matrix(POP,S,0)
        output = tmp * pcom
    elif shape == "star":
        tmp = np.zeros((POP,POP))
        ii = np.random.choice(POP)
        tmp[ii,:] = 1
        tmp[ii,ii] = 0
        output = tmp * pcom
    elif shape == "ring":
        tmp = np.eye(POP)
        tmpA = np.vstack((tmp[1:,:],tmp[0,:]))
        tmpB = np.vstack((tmp[-1:,:],tmp[:-1,:]))
        tmp = tmpA + tmpB
        output = tmp * pcom
    else:
        print(f"Unrecognized network shape '{shape}'")

    return(output)

def generate_couples(POP,S=2,shape="cycle"):
    """Generates couplings between landscapes (external interaction) 

    Args:
        POP (int): Number of landscapes (population size)
        S (int): Number of landscapes considered for external bits
        shape (str): A network topology. Takes values 'cycle' (default)

    Returns:
        list: A list of S-sized vectors with couples for every landscape.
    """

    if S>=POP:
        print("Error: wrong network degree")
        return 0
    output = None
    if S == 0:
        output = []
    elif shape=="cycle":
        tmp = [[(z-1)%POP] + [_%POP for _ in range(z+1,z+S)] for z in range(POP)]
        output = tmp
    else:
        print(f"Unrecognized network shape '{shape}'")
    return(output)
###############################################################################

def get_neighbours(vec,count):
    """Generates binary vectors that are 1-bit away (a unit Hamming distance)

    Args:
        vec (list or numpy.ndarray): An input vector
        count (int): Number of neighbours to generate

    Returns:
        list: A list of 1-bit neighbours of vec
        list: A list of decimal equivalents of the above
    """

    tmpv = []
    tmpi = []
    subbset = np.random.choice(np.arange(len(vec)),count,replace=False)
    for i in subbset:
        y = vec.copy()
        y[i] = 1 - y[i]
        tmpv.append(y)
        tmpi.append(binx(y))
    return(tmpv, tmpi)

def random_neighbour(vec,myid,n):
    """Generates a random binary vector that is 1-bit away (a unit Hamming distance)

    Args:
        vec (list or numpy.ndarray): An input vector
        myid (int): An id of an agent of interest
        n (int): Number of tasks allocated to a single agent

    Returns:
        list: A vector with one bit flipped for agent myid
    """

    tmp = vec.copy()
    rnd = np.random.choice(range(myid*n,(myid+1)*n))
    tmp[rnd] = 1- tmp[rnd]
    output = tmp
    return output

def get_index(vec,myid,n):
    """Gets a decimal equivalent for the bitstring for agent myid

    Args:
        vec (list or numpy.ndarray): An input vector
        myid (int): An id of an agent of interest
        n (int): Number of tasks allocated to a single agent

    Returns:
        int: A decimal value of vec for myid
    """

    tmp = vec.copy()
    tmp = binx(tmp[myid*n:(myid+1)*n])
    output = tmp
    return output

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

def hamming_distance(x,y):
    """Calculates the Hamming distance (count of different bits) between two bitstrings
    
    Args:
        x (list): first list
        y (list): second list

    Returns:
        int: An integer value

    """
    tmp = np.sum(np.abs(np.array(x) - np.array(y)))
    output = tmp
    return output
###############################################################################

def beta_mean(x,y):
    """Calculates the mean of the Beta(x,y) distribution

    Args:
        x (float): alpha (shape) parameter
        y (float): beta (scale) parameter

    Returns:
        float: A float that returns x/(x+y)
    """
    
    output = x / (x+y)
    return output

def flatten(x):
    """Converts a list of lists into a flat list

    Args:
        x (list): An input list

    Returns:
        list: A flat list
    """
    
    output = [_ for z in x for _ in z]
    return output

#def calculate_freq(x,vec):
#    tmp = flatten(vec)
#    if len(tmp)==0:
#        tmp = 0
#    else:
#        tmp = tmp.count(x)/len(tmp)
#    output = tmp
#    return output

def calculate_freq(x,vec):
    """Calclates frequency of vec in x

    Args:
        x (numpy.ndarray): A 2d matrix
        vec (numpy.ndarray): A 1d array

    Returns:
        float: Frequency of vec in x
    """

    if x is None or vec.size==0 or vec[0,0]==-1:
        return 0.0
    freq = np.mean(vec,axis=0)
    tmp1 = np.multiply(x,freq)
    tmp2 = np.multiply(1-x,1-freq)
    tmp = tmp1 + tmp2
    output = np.mean(tmp)
    return output

def extract_soc(x,myid,n,nsoc):
    """Extracts social bits from a bitstring

    Args:
        x (numpy.ndarray): An input vector
        myid (int): An id of an agent of interest
        n (int): Number of tasks allocated to a single agent
        nsoc (int): Number of social tasks (exogeneous)

    Returns:
        numpy.ndarray: A vector of size nsoc
    """

    tmp = x[(myid+1)*n-nsoc:(myid+1)*n]
    #tmp = str(tmp).replace(" ","").replace("[","").replace("]","")
    output = tmp
    return output

#def extract_pub(x,myid,n,p,npub):
#    tmp = np.reshape(x,(p,n))
#    tmp = tmp[:,-npub:n]
#    output = tmp
#    return output

###############################################################################

def pick(vec,count):
    """Picks random elements form a vector

    Args:
        vec (numpy.ndarray): An input vector
        count (int): Number of elements to pick

    Returns:
        numpy.ndarray: A vector of size count
    """

    tmp = np.random.choice(range(vec),count,replace=False)
    output = np.sort(tmp)
    return output

def artify(n,p,r):
    """depreciated"""

    tmp = np.arange(n*p)
    tmp = tmp.reshape(p,n)
    fnc = lambda z: np.random.choice(z,r,replace=False)
    tmp = np.apply_along_axis(fnc,1,tmp)
    output = tmp
    return output

def calculate_match(x,art):
    """depreciated"""

    if art==[]:
        return 0.0
    tmp = [x[z[0]]==z[1] for z in art]
    output = sum(tmp) / len(tmp) 
    return output

###############################################################################

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

    cond = (x >= u)
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

def goal_prog(perf1,perf2,u,p1,p2):
    """A goal programming

    Args:
        perf1 (float): first performance
        perf2 (float): second performance
        u (list): goals
        p1 (float): first weigh
        p2 (float): second weight

    Returns:
        float: A GP output
    """

    d1 = np.max((u[0]-perf1,0))
    d2 = np.max((u[1]-perf2,0))
    tmp = p1*d1 + p2*d2
    output = -tmp
    return output

def schism(perf1,perf2,social):
    """depreciated"""

    tmp = None
    if social is True:
        tmp = perf2
    else:
        tmp = perf1
    output = tmp
    return output

def similarity(x,p,n,nsoc):
    """Calculates the similarity measure of the bitstring

    Args:
        x (list): the bitstring of interest
        p (int): number of agents
        n (int): number of tasks per agent
        nsoc (int): number of imitated tasks per agent

    Returns:
        float: The float between 0 and 1
    """
    if p<2:
        print('Need at least 2 agents for similarity measure')
        return

    tmp = np.reshape(x, (p,n))[:,(n-nsoc):]
    summ = 0
    for i in range(p):
        for j in range(i,p):
            summ += hamming_distance(tmp[i,:],tmp[j,:])

    max_summ = nsoc*(p/2)**2 if p%2==0 else nsoc*((p-1)/2)**2 + (p-1)*(nsoc/2)
    output = 1-summ/max_summ
    return output
