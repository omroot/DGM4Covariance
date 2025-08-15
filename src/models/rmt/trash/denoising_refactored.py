

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.model_selection import LeaveOneOut, KFold



def MarcenkoPasturPDF(variance: float ,
                      q : float,
                      pts: int)-> pd.Series():
    """ Compute the probability density function of the Marcenko-Pastur distribution

    Args:
        variance: variance parameter of the Marcenko-Pastur distribution
        q: the ratio T/N where T is the number of rows and N is the number of columns of the data matrix
        In Asset management, the data matrix is
        pts: the number of the data points in the grid the pdf is evaluated on

    Returns:
        pdf: the probability density function of the Marcenko-Pastur distribution with parameters
        variance and q on pts points
    
    """ 

    min_eigenvalue = variance*(1 - (1./q)**.5 )**2
    max_eigenvalue = variance*(1  + (1./q)**.5 )**2

    eigenvalues_grid = np.linspace(min_eigenvalue,max_eigenvalue,pts)

    pdf=q/(2*np.pi*variance*eigenvalues_grid)*((max_eigenvalue-eigenvalues_grid)*(eigenvalues_grid-min_eigenvalue))**.5

    pdf=pd.Series(pdf,index=eigenvalues_grid)

    return pdf



def getPCA(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the principal components of an input matrix.
    Args:
        matrix: input matrix, typically a covariance or correlation matrix, of dimension NxN

    returns:
        eigenvalues: the eigenvalues , or the variances of the learned principal components,  it is an NxN matrix and
         the eigenvalues are on the diagonal
        eigenvectors: NxN matrix of the eigen vectors, or principal components, each column [:,i] is a normalized
        principal component corresponding to the eigenvalue [i,i]

    """
    # Get eigenvalues,eigenvectors from a Hermitian matrix
    eigenvalues,eigenvectors=np.linalg.eigh(matrix)

    indices=eigenvalues.argsort()[::-1] # arguments for sorting eVal desc
    eigenvalues =eigenvalues[indices]
    eigenvalues=np.diagflat(eigenvalues)

    eigenvectors = eigenvectors[:,indices]
    return eigenvalues,eigenvectors


def FitEvaluateKernelDensityEstimator(observations: np.ndarray,
                               kernel_bandwidth: float=.25,
                               kernel: str='gaussian',
                               evaluation_observations=None) -> pd.Series:
    """Fits a kernel density estimator to an array of input observations
    Args:
        observations: input data
        kernel_bandwidth: the bandwidth of the kernel
        kernel: the kernel to be used
        evaluation_observations: optional array of observation the fitted pdf will be evaluated on, input training
        data will be used instead
    return:
        pdf: probability density function evaluated on the either the training observations or the supplied
        evaluation observations
    """
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(observations.shape)==1:
        observations=observations.reshape(-1,1)
    KernelDensityEstimator = KernelDensity(kernel=kernel,bandwidth=kernel_bandwidth).fit(observations)

    if evaluation_observations is None:
        evaluation_observations=np.unique(observations).reshape(-1,1)
    if len(evaluation_observations.shape)==1:
        evaluation_observations=evaluation_observations.reshape(-1,1)
    logProb=KernelDensityEstimator.score_samples(evaluation_observations) # log(density)
    pdf=pd.Series(np.exp(logProb),index=evaluation_observations.flatten())

    return pdf



def find_optimal_kernel_bandwidth(eigenvalues: np.ndarray) -> dict:
    """This finds the best kernel bandwidth value that fits the best the Kernel Density Estimator
    Args:
        eigenvalues: the eigenvalues the Kernel Density Estimator is fitted on
    Return:
        best_params: a dictionary with the best parameters
    """

    bandwidths = 10 ** np.linspace(-3, 1, 250)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=KFold(100))
    grid.fit(eigenvalues[:, None])


    return grid.best_params_

def generate_random_covariance(number_total_factors :int,
                               number_signal_factors: int)-> np.ndarray:
    """Generates a random covariance matrix with non random/signal coming from a predefined fixed number of
    factors
    Args:
        number_total_factors:
        number_signal_factors:
    Return:
        covariance_matrix: the generated covariance matrix

    """
    w=np.random.normal(size=(number_total_factors,number_signal_factors))
    covariance_matrix=np.dot(w,w.T) # random cov matrix, however not full rank
    covariance_matrix+=np.diag(np.random.uniform(size=number_total_factors)) # full rank cov
    return covariance_matrix

def covariance2correlation(covariance_matrix: np.ndarray)-> np.ndarray:
    """Converts a covariance matrix into a correlation matrix
    Args:
        covariance_matrix: input covariance matrix
    Returns:
        correlation_matrix: output correlation matrix
    """
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(covariance_matrix))
    correlation_matrix=covariance_matrix/np.outer(std,std)
    correlation_matrix[correlation_matrix<-1] = -1 # numerical error
    correlation_matrix[correlation_matrix>1]  = 1 # numerical error
    return correlation_matrix

def empirical_vs_theoretical_pdf_error(variance: np.ndarray,
                                        eigenvalues: np.ndarray,
                                        q:float,
                                        kernel_bandwidth:float,
                                        pts:int=1000)->float:
    """ Computes the sum of squared errors between the theoretical MarcenKo-Pastur distribution
    and the empirical distribution computed using Kernel Density Estimation
    Args:
        variance: parameter for the MarcenKo-Pastur distribution
        eigenvalues: input observation
        q: MP distribution parameter
        kernel_bandwidth: kernel bandwith for the Kernel Density Estimator
        pts: number of data points on the grid for the theoretical distribution pdf
    return:
        sse: sum of squared errors between theoretical and empirical distribution
    """

    var = variance[0]
    theoretical_pdf=MarcenkoPasturPDF(var,q,pts) # theoretical pdf


    empirical_pdf=FitEvaluateKernelDensityEstimator(observations=eigenvalues,
                                                    kernel_bandwidth=kernel_bandwidth,
                                                    evaluation_observations=theoretical_pdf.index.values) # empirical pdf
    sse=np.sum((theoretical_pdf-empirical_pdf)**2)
    return sse


def find_max_random_eigenvalue(eigenvalues,
                                q,
                                kernel_bandwidth):
    """

    :param eigenvalues:
    :param q:
    :param kernel_bandwidth:
    Returns:
        max_random_eigenvalue:
        variance:

    """
    # Find max random eVal by fitting Marcenko’s distribution
    out=minimize(lambda *x:empirical_vs_theoretical_pdf_error(*x),
                 x0=np.array(.5),
                 args=(eigenvalues,q,kernel_bandwidth),
                 bounds=((1E-5,1-1E-5),))
    print(out)
    print(f"found errPDFs {out['x'][0]}"  )

    if out['success']:
        variance=out['x'][0]
    else:
        variance=1
    max_random_eigenvalue = variance*(1+(1./q)**.5)**2
    implied_variance = variance*(1-max_random_eigenvalue/len(eigenvalues))
    return max_random_eigenvalue,implied_variance



def denoise_correlation_matrix(eigenvalues: np.ndarray,
                               eigenvectors: np.ndarray,
                               number_signal_factors: int) -> np.ndarray:
    """Produce a denoised correlation matrix from the set of eigenvalues and eigenvectors and the number of
    signal factors (that can be learned from the Marcenko-Pastur distribution
    Args:
        eigenvalues: the eigen values of the  correlation matrix to be denoised
        eigenvectors: the eigen vectors of the  correlation matrix to be denoised
        number_signal_factors: the number of factors with signal
    Returns:
        correlation_matrix: denoised correlation matrix

    """
    eigenvalues_corrected=np.diag(eigenvalues).copy()
    eigenvalues_corrected[number_signal_factors:]=eigenvalues_corrected[number_signal_factors:].sum()/float(eigenvalues_corrected.shape[0]-number_signal_factors)
    eigenvalues_corrected=np.diag(eigenvalues_corrected)
    correlation_matrix=np.dot(eigenvectors,eigenvalues_corrected).dot(eigenvectors.T)
    correlation_matrix=covariance2correlation(correlation_matrix)
    return correlation_matrix


def shrinkage_denoise_correlation_matrix(eigenvalues,
                                          eigenvectors,
                                          number_signal_factors,
                                          alpha=0):
    """ Denoise the correlation matrix through targeted shrinkage
    Args:
        eigenvalues:  the eigen values of the  correlation matrix to be denoised
        eigenvectors: the eigen vectors of the  correlation matrix to be denoised
        number_signal_factors: the number of factors with signal
        alpha: regularization parameter (zero for total shrinkage)

    Returns:
        denoised_correlation_matrix: denoised correlation matrix

    """


    signal_eigenvalues     = eigenvalues[:number_signal_factors,:number_signal_factors]
    signal_eigenvectors    = eigenvectors[:,:number_signal_factors]


    random_eigenvalues     = eigenvalues[number_signal_factors:, number_signal_factors: ]
    random_eigenvectors    = eigenvectors[:,number_signal_factors:]


    signal_correlation_matrix=np.dot(signal_eigenvectors,signal_eigenvalues).dot(signal_eigenvectors.T)

    random_correlation_matrix=np.dot(random_eigenvectors,random_eigenvalues).dot(random_eigenvectors.T)

    denoised_correlation_matrix=signal_correlation_matrix+alpha*random_correlation_matrix+(1-alpha)*np.diag(np.diag(random_correlation_matrix))

    return denoised_correlation_matrix


def detone_correlation_matrix(correlation_matrix,
                              eigenvalues,
                              eigenvectors,
                              market_components_max_index=1):
    """Detones the denoised correlation matrix from its eigen values and vectors by removing the market component.
    Args:
        correlation_matrix: input correlation matrix
        eigenvalues:the eigen values of the  correlation matrix to be detoned
        eigenvectors:the eigen vectors of the  correlation matrix to be detoned
        market_components_max_index: maximum index of the market component (they could be more than one)
    Returns:
        detoned_correlation_matrix: detoned correlation matrix

    """

    # Getting the eigenvalues and eigenvectors related to market component
    eigenvalues_market = eigenvalues[:market_components_max_index, :market_components_max_index]
    eigenvectors_market = eigenvectors[:, :market_components_max_index]

    # Calculating the market component correlation
    correlation_market_component = np.dot(eigenvectors_market, eigenvalues_market).dot(eigenvectors_market.T)

    # Removing the market component from the de-noised correlation matrix
    unscaled_detoned_correlation_matrix = correlation_matrix - correlation_market_component

    # Rescaling the correlation matrix to have 1s on the main diagonal
    detoned_correlation_matrix = covariance2correlation(unscaled_detoned_correlation_matrix)

    return detoned_correlation_matrix





def denoisedCorr(eVal,eVec,nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T)
    corr1=cov2corr(corr1)
    return corr1





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