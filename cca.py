import numpy as np
from numpy import dot
import scipy.linalg as la

class EigenValueVector:
    def __init__(self, value, vector):
        self.value = value
        self.vector = vector
               
def CleanAndSortEigenvalues(eigenvalues, eigenvectors):
    evs = []
    for i in range(0, len(eigenvalues)):
        eva = eigenvalues[i]
        if eva.imag == 0:
            ev = EigenValueVector(eigenvalues[i], eigenvectors[:,i])
            evs.append(ev)
            
    evs.sort(key=lambda evv: evv.value, reverse=True)
    
    sevals = np.zeros(len(evs))
    sevecs = np.zeros((eigenvectors.shape[0], len(evs)))
        
    for i in range(0, len(evs)):
            sevals[i] = evs[i].value.real
            sevecs[:,i] = evs[i].vector
            
    return (sevals, sevecs)

def cca(X,Y):
    """Canonical Correlation Analysis
    
    :param X: observation matrix in X space, every column is one data point
    :param Y: observation matrix in Y space, every column is one data point
    
    :returns: (basis in X space, basis in Y space, correlation)
    """
    
    N = X.shape[1]
    Sxx = 1.0/N * dot(X, X.transpose())
    Sxy = 1.0/N * dot(X, Y.transpose())
    Syy = 1.0/N * dot(Y, Y.transpose())  
  
    L = dot(Sxy, dot(la.pinv2(Syy), Sxy.transpose()))
    lambda2s,A = la.eig(L, Sxx)
    lambdas = np.sqrt(lambda2s)  
    clambdas, cA = CleanAndSortEigenvalues(lambdas, A)         
    B = dot(dot(dot(la.pinv2(Syy), Sxy.transpose()), cA), np.diag(1.0 / clambdas))
         
    return (cA, B, clambdas)



# test
if __name__ == "__main__":
    nobs = 10
    nxdim = 3
    nydim = 3
    x = np.arange(nxdim*nobs).reshape(nxdim,nobs)
    y = np.arange(nydim*nobs).reshape(nydim,nobs)
    
    (A,B,lambdas) = cca(x,y)
    
    print "lambdas=\n",lambdas
    print "A=\n",A  
    print "B=\n",B

