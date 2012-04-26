import numpy as np
from numpy import dot
import scipy.linalg as la
              
def clean_and_sort_eigenvalues(eigenvalues, eigenvectors):  
    evs = [(va,ve) for va,ve in zip(eigenvalues,eigenvectors.T) if va.imag == 0]           
    evs.sort(key=lambda evv: evv[0], reverse=True)   
    sevals = np.array([va.real for va,_ in evs])
    sevecs = np.array([ve for _,ve in evs]).T                   
    return sevals, sevecs

def cca(X,Y):
    """Canonical Correlation Analysis
    
    :param X: observation matrix in X space, every column is one data point
    :param Y: observation matrix in Y space, every column is one data point
    
    :returns: (basis in X space, basis in Y space, correlation)
    """
    
    N = X.shape[1]
    Sxx = 1.0/N * dot(X, X.T)
    Sxy = 1.0/N * dot(X, Y.T)
    Syy = 1.0/N * dot(Y, Y.T)  
    
    epsilon = 1e-6
    rSyy = Syy + epsilon * np.eye(Syy.shape[0])
    rSxx = Sxx + epsilon * np.eye(Sxx.shape[0])   
    irSyy = la.inv(rSyy)
    
    L = dot(Sxy, dot(irSyy, Sxy.T))
    lambda2s,A = la.eig(L, rSxx)
    lambdas = np.sqrt(lambda2s)  
    clambdas, cA = clean_and_sort_eigenvalues(lambdas, A)         
    B = dot(irSyy, dot(Sxy.T, dot(cA, np.diag(1.0 / clambdas))))
         
    return (cA, B, clambdas)


# test
if __name__ == "__main__":
    
    if True:
        # 3d test    
        baseA = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1]]).T
        baseB = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0]]).T
        latent = np.random.random((3,1000))
    else:
        # 1d test
        baseA = np.array([[0],
                          [1],
                          [2]])
        baseB = np.array([[1],
                          [0],
                          [1]])
        latent = np.random.random((1,1000))   
    
    x = dot(baseA, latent)
    y = dot(baseB, latent)
    
    (A,B,lambdas) = cca(x,y)
    
    #print "latent=\n",latent
    #print "x=\n",x
    #print "y=\n",y
    print "lambdas=\n",lambdas
    print "A=\n",A  
    print "B=\n",B    
    atx = dot(A.T,x[:,0:5])
    aty = dot(B.T,y[:,0:5])
    diff = la.norm(atx-aty,'fro')
    print "A^T * x=\n",atx
    print "B^T * y=\n",aty
    print "diff=",diff
    assert diff <= 1e-10, 'Test failed'
