

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.model_selection import LeaveOneOut
#---------------------------------------------------
#
def mpPDF(var,q,pts):
    # Marcenko-Pastur pdf
    # q=T/N
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf,index=eVal)
    return pdf


#---------------------------------------------------
def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal,eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec

#---------------------------------------------------
def fitKDE(obs,bWidth=.25,kernel='gaussian',x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:
        obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None:
        x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:
        x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf


def findOptimalBWidth(eigenvalues):
    bandwidths = 10 ** np.linspace(-2, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(eigenvalues[:, None]);

    # Now we can find the choice of bandwidth which maximizes the score (which in this case defaults to the log-likelihood):

    return grid.best_params_
#---------------------------------------------------
def getRndCov(nCols,nFacts):
    w=np.random.normal(size=(nCols,nFacts))
    cov=np.dot(w,w.T) # random cov matrix, however not full rank
    cov+=np.diag(np.random.uniform(size=nCols)) # full rank cov
    return cov

#---------------------------------------------------
def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(cov))
    corr=cov/np.outer(std,std)
    corr[corr<-1],corr[corr>1]=-1,1 # numerical error
    return corr


#---------------------------------------------------
def errPDFs(var,eVal,q,bWidth,pts=1000):
    # Fit error
    var = var[0]
    pdf0=mpPDF(var,q,pts) # theoretical pdf
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse
#---------------------------------------------------
def findMaxEval(eVal,q,bWidth):
    # Find max random eVal by fitting Marcenko’s dist
    out=minimize(lambda *x:errPDFs(*x),
                 x0=np.array(.5),
                 args=(eVal,q,bWidth),
                 bounds=((1E-5,1-1E-5),))
    print("found errPDFs" + str(out['x'][0]))

    if out['success']:
        var=out['x'][0]
    else:
        var=1
    eMax=var*(1+(1./q)**.5)**2
    return eMax,var
#---------------------------------------------------


def detoned_corr(corr, eigenvalues, eigenvectors, market_component=1):
    """
    De-tones the de-noised correlation matrix by removing the market component.
    The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
    of the first eigenvalue that is above the maximum theoretical eigenvalue and the number of
    eigenvectors related to a market component.
    :param corr: (np.array) Correlation matrix to detone.
    :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
    :param eigenvectors: (float) Eigenvectors array.
    :param market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
    :return: (np.array) De-toned correlation matrix.
    """

    # Getting the eigenvalues and eigenvectors related to market component
    eigenvalues_mark = eigenvalues[:market_component, :market_component]
    eigenvectors_mark = eigenvectors[:, :market_component]

    # Calculating the market component correlation
    corr_mark = np.dot(eigenvectors_mark, eigenvalues_mark).dot(eigenvectors_mark.T)

    # Removing the market component from the de-noised correlation matrix
    corr = corr - corr_mark

    # Rescaling the correlation matrix to have 1s on the main diagonal
    corr = cov2corr(corr)

    return corr



def denoisedCorr(eVal,eVec,nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T)
    corr1=cov2corr(corr1)
    return corr1


def denoisedCorr2(eVal,eVec,nFacts,alpha=0):
    # Remove noise from corr through targeted shrinkage
    eValL,eVecL=eVal[:nFacts,:nFacts],eVec[:,:nFacts]
    eValR,eVecR=eVal[nFacts:,nFacts:],eVec[:,nFacts:]
    corr0=np.dot(eVecL,eValL).dot(eVecL.T)
    corr1=np.dot(eVecR,eValR).dot(eVecR.T)
    corr2=corr0+alpha*corr1+(1-alpha)*np.diag(np.diag(corr1))
    return corr2


def findOptimalBWidth(eigenvalues):
    bandwidths = 10 ** np.linspace(-2, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(eigenvalues[:, None]);

    # Now we can find the choice of bandwidth which maximizes the score (which in this case defaults to the log-likelihood):

    return grid.best_params_


def formBlockMatrix(nBlocks,bSize,bCorr):
    block=np.ones((bSize,bSize))*bCorr
    block[range(bSize),range(bSize)]=1
    corr=block_diag(*([block]*nBlocks))
    return corr
#---------------------------------------------------

def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0


def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov
# generating the empirical covariance matrix
def simCovMu(mu0, cov0, nObs, shrink=False):
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size = nObs)
    #print(x.shape)
    mu1 = x.mean(axis = 0).reshape(-1,1) #calc mean of columns of rand matrix
    #print(mu1.shape)
    if shrink: cov1 = LedoitWolf().fit(x).covariance_
    else: cov1 = np.cov(x, rowvar=0)
    return mu1, cov1

# Denoising of the empirical covariance matrix
# by constant residual eigenvalue method
def deNoiseCov(cov0, q, bWidth):
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0) #denoising by constant residual eigenvalue method
    cov1 = corr2cov(corr1, np.diag(cov0)**.5)
    return cov1


# Derive minimum-variance-portfolio
# Returns a column vector of percentage allocations
# should be subject to lagrangian constraints:
# 1. lambda_1*(sum(expectation(x_i)*x_i) - d = 0
# 2. lambda_2*(sum(x_i - 1))=0
# where d is expected rate of return
# w*=C^−1*μ/I.T*C^−1*μ - is minimum-variance-portfolio
# short sales are allowed
def optPort(cov, mu=None):
    inv = np.linalg.inv(
        cov)  # The precision matrix: contains information about the partial correlation between variables,
    #  the covariance between pairs i and j, conditioned on all other variables
    #  (https://www.mn.uio.no/math/english/research/projects/focustat/publications_2/shatthik_barua_master2017.pdf)
    ones = np.ones(shape=(inv.shape[0], 1))  # column vector 1's
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)  # def: w = w / sum(w) ~ w is column vector

    return w


# optPort with long only curtesy of Brady Preston
# requires: import cvxpy as cp
'''def optPort(cov,mu=None):
    n = cov.shape[0]
    if mu is None:mu = np.abs(np.random.randn(n, 1))
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov)
    ret =  mu.T @ w
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(risk),constraints)
    prob.solve(verbose=True)
    return np.array(w.value.flat).round(4)'''


# According to the question 'Tangent portfolio weights without short sales?'
# there is no analytical solution to the GMV problem with no short-sales constraints
# So - set the negative weights in WGV to 0, and make w sum up to 1
def optPortLongOnly(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))  # column vector 1's
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)  # def: w = w / sum(w) ~ w is column vector
    w = w.flatten()
    threshold = w < 0
    wpluss = w.copy()
    wpluss[threshold] = 0
    wpluss = wpluss / np.sum(wpluss)

    return wpluss