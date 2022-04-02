import numpy as np
import nkpack as nk

p = 8
n = 4
nsoc = 3
breakpoint()

tmp = 0
for i in range(2**(n*p)):
    ibin = nk.binx(i,n*p)
    kj = nk.similarity(ibin,p,n,nsoc)
    if kj < tmp:
        print(kj)
        print(ibin)
        tmp = kj
