import numpy as np
from numba import jit,njit
from scipy.stats import norm
from itertools import combinations as comb

###############################################################################

def interaction_matrix(N,K,shape="roll"):
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
    tmp = x
    if type(tmp) is int:
        tmp = np.binary_repr(tmp,size)
        tmp = [int(z) for z in tmp]
    elif type(tmp) is str:
        tmp = int(tmp,2)
    elif type(tmp) is np.ndarray:
        lenx = len(tmp)
        tmp = np.sum(np.flip(tmp) * 2 ** (np.arange(len(tmp))))
    elif type(tmp) is list:
        tmp = np.sum(np.flip(tmp) * 2 ** (np.arange(len(tmp))))
    else:
        print("incorrect input for function binx")
    return tmp

def contrib_define(p,n,k,c,s,rho):
    #output = np.random.uniform(0,1,(2**(1+k+c*s),n*p))
    corrmat = np.repeat(rho,p*p).reshape(p,p) + (1-rho) * np.eye(p)
    corrmat = 2*np.sin((np.pi / 6 ) * corrmat)
    tmp = np.random.multivariate_normal(mean=[0]*p,cov=corrmat,size=(n*2**(1+k+c*s)))
    tmp = norm.cdf(tmp)
    tmp = np.reshape(tmp.T,(n*p,(2**(1+k+c*s)))).T
    output = tmp
    return output

@njit
def xcontrib_solve(x,imat,cmat,n,p):
    n_p = n*p
    phi = [0.0]*n_p
    for i in range(n_p):
        tmq = np.arange(n_p)#imat[:,i].copy()
        q = int(i/n)
        m = i % n
        tmq = np.concatenate((tmq[q*n:(q+1)*n],tmq[0:q*n],tmq[(q+1)*n:n_p]))
        tmq = np.concatenate((tmq[m:n],tmq[0:m],tmq[n:n_p]))
        tmx = np.where(imat[:,i]>0)[0]
        tmpx = []
        for j in tmq:
            a = False
            for jj in tmx:
                a = a or (j == jj)
            tmpx.append(a)
        tmpx = np.array(tmpx)
        # used loops, because jit does not support list comprehensions
        #tmpx = [z in tmx for z in tmq]
        #tmpx = np.isin(tmq,tmx)
        #tmpx = np.in1d(tmq,tmx)

        indices = tmq[tmpx]

        tmp = x[indices]
        tmp_loc = np.sum(np.flip(tmp) * 2 ** (np.arange(len(tmp)))) #binx(tmp)
        phi[i] = cmat[tmp_loc,i]
    return phi

def contrib_solve(x,imat,cmat,n,p):
    phi = xcontrib_solve(x,imat,cmat,n,p)
    phi = np.reshape(phi,(p,n)).mean(axis=1)
    return phi

def contrib_full(imat,cmat,n,p):
    n_p = n*p
    perfmat = np.zeros((2**n_p,p),dtype=float)
    for i in range(2**n_p):
        bstring = np.array(binx(i,n_p))
        bval = contrib_solve(bstring,imat,cmat,n,p) # !!! processing heavy !!!
        perfmat[i,] = bval
    perfmax = np.max(perfmat,axis=0)
    
    perfmean = np.mean(perfmat,axis=1)
    perfglobalmax = perfmean.max()
    #perfargmax = perfmean.argmax() 

    output1 = perfmat / perfmax
    #output2 = perfargmax
    #output3 = perfglobalmax
    output3 = perfmax
    return output1, output3#, output2, output3

#def getglobalmax2(imat,cmat,n,p):
#    n_p = n*p
#    perfmax = np.zeros(p,dtype=float)
#    perfargmax = 0
#    for i in range(2**n_p):
#        print(i)
#        bstring = np.array(binx(i,n_p))
#        bval = contrib_solve(bstring,imat,cmat,n,p) # !!! processing heavy !!!
#        perfmax = np.maximum(perfmax, bval)
#
#    output = perfmax
#    return output

def get_globalmax(imat,cmat,n,p,t0=1,t1=0.1,alpha=0.0001,k=1):
    n_p = n*p
    
    state = np.array(binx(0,n_p))
    value = contrib_solve(state,imat,cmat,n,p)
    
    t = t0
    while t>t1:
        state_ = random_neighbour(state,0,n_p)
        value_ = contrib_solve(state_,imat,cmat,n,p)
        
        d_sum = np.sum(value_) - np.sum(value)
        d_sep = value_ - value

        if (d_sep > 0).any() or (np.exp(d_sep/t) > np.random.uniform()).any():
        #if (d_sum>0) or (np.exp(d_sum/t) > np.random.uniform()):
            state = state_
            value = np.maximum(value_,value)
        t -= alpha

    output = value
    return output
###############################################################################

def assign_tasks(N,POP,shape="solo"):
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

def generate_network(POP,S=[2],shape="cycle",absval=False,weights=[1.0]):
    if type(S) is int:
        print("Error: wrap S in a list")
        return 0
    elif max(S)>=POP:
        print("Error: wrong network degree")
        return 0
    output = None
    if S == 0:
        output = []
    elif shape=="cycle":
        tmp = [[z-1] + [_%POP for _ in range(z+1,z+S[0])] for z in range(POP)]
        if absval==True:
            tmp = [[(z-1)%POP] + [_%POP for _ in range(z+1,z+S[0])] for z in range(POP)]
        output = tmp
    elif shape == "random":
        tmp = []
        for ss in S:
            tmp.append(random_binary_matrix(POP,ss,0))
        output = np.average(tmp,0,weights)
        #output = np.where(tmp>0)[1]
        #output = np.array_split(output,POP)
    else:
        print(f"Unrecognized network shape '{shape}'")

    return(output)

def generate_couples(POP,S=2,shape="cycle"):
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
    tmp = vec.copy()
    rnd = np.random.choice(range(myid*n,(myid+1)*n))
    tmp[rnd] = 1- tmp[rnd]
    output = tmp
    return output

def get_index(vec,myid,n):
    tmp = vec.copy()
    tmp = binx(tmp[myid*n:(myid+1)*n])
    output = tmp
    return output

def with_noise(vec,pb=0):
    if vec.size==0:
        return vec
    tmp = vec
    nvec = len(vec)
    noise = np.random.choice(2,p=[1-pb, pb])
    if noise:
        tmp = random_neighbour(vec,0,nvec)
    output = tmp
    return output
###############################################################################

def beta_mean(x,y):
    output = x / (x+y)
    return output

def flatten(x):
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
    if x is None or vec.size==0 or vec[0,0]==-1:
        return 0.0
    freq = np.mean(vec,axis=0)
    tmp1 = np.multiply(x,freq)
    tmp2 = np.multiply(1-x,1-freq)
    tmp = tmp1 + tmp2
    output = np.mean(tmp)
    return output

def extract_soc(x,myid,n,nsoc):
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
    tmp = np.random.choice(range(vec),count,replace=False)
    output = np.sort(tmp)
    return output

def artify(n,p,r):
    tmp = np.arange(n*p)
    tmp = tmp.reshape(p,n)
    fnc = lambda z: np.random.choice(z,r,replace=False)
    tmp = np.apply_along_axis(fnc,1,tmp)
    output = tmp
    return output

def calculate_match(x,art):
    if art==[]:
        return 0.0
    tmp = [x[z[0]]==z[1] for z in art]
    output = sum(tmp) / len(tmp) 
    return output

###############################################################################

def cobb_douglas(vec1,vec2):
    x = [z+1 for z in vec2]
    w = vec1
    tmp = np.power(x,w).prod()
    output = tmp
    return output

def satisfice(x,y,u):
    cond = (x >= u)
    tmp = cond * (y + u) + (1-cond) * x
    output = tmp
    return output

def weighted(x,y,p1,p2):
    tmp = p1*x + p2*y
    output = tmp
    return output

def goal_prog(perf1,perf2,u,p1,p2):
    d1 = np.max((u[0]-perf1,0))
    d2 = np.max((u[1]-perf2,0))
    tmp = p1*d1 + p2*d2
    output = -tmp
    return output

def schism(perf1,perf2,social):
    tmp = None
    if social is True:
        tmp = perf2
    else:
        tmp = perf1
    output = tmp
    return output
