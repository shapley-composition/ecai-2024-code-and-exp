import numpy as np
from math import sqrt
from composition_stats import ilr

def mahalanobis_matrix(model, X, Y):
    N = len(np.unique(Y))
    P = ilr(model(X))
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1,N):
            Pi = P[Y == i]
            Pj = P[Y == j]
            mi = Pi.mean(axis=0)
            mj = Pj.mean(axis=0)
            co = ((len(Pi)-1)*np.cov(Pi, rowvar=False) + (len(Pj)-1)*np.cov(Pj, rowvar=False))/(len(Pi)+len(Pj)-2)
            d  = sqrt(np.dot(np.dot(mi - mj , np.linalg.inv(co)),mi - mj))
            M[i,j] = d
            M[j,i] = d
    return M
