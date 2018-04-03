from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import numpy as np
import os, fnmatch
import random
import math

dataDir = '/u/cs401/A3/data/'
#dataDir = 'C:/Users/Admin/401a3/a3401/data'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    #Check if there's a precomputation
    if(len(preComputedForM) == 0):
        preCom = precompute(myTheta.mu[m], myTheta.Sigma[m])
    else:
        preCom = preComputedForM[m]
        
    #Calculate the first term
    sumPart = np.zeros((len(x),len(x[0])))
    s = myTheta.Sigma[m]
    mu = myTheta.mu[m]
    
    #Vectorize calculations...
    sumPart = (0.5 * (x ** 2) * (1 / s)) - (mu * x * (1 / s))
    sumPart = sumPart.sum(axis=1)
        
    #TODO: assume that a Txd matrix is given instead so each row represents a vector
    return (-1 * sumPart) - preCom



def precompute(mu, sigma):
    """Performs the precomputation for log_b_m_x given the mu and sigma of
       the model
    """    
    sumPart = 0
    d = len(mu)
    for index in range(0, d):
        sumPart = sumPart + ((mu[index] ** 2) / (2 * sigma[index]))
    
    productPart = 1
    for index in range(0, d):
        productPart = productPart * sigma[index]
        
    return sumPart + ((d / 2) * math.log(2 * math.pi)) + (0.5) * math.log(productPart)

    
def log_p_m_x( m, x, myTheta, precomBMX=[]):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    #Compute the log b values if not given...
    if(len(precomBMX) == 0):
        for component in range(0, len(myTheta.omega)):
            precomBMX.append(log_b_m_x(m, x, myTheta))

    #Calculate log of the numerator
    bm = []
    for component in range(0, len(myTheta.omega)):
        bm.append(log_b_m_x(m, x, myTheta))
    numerator = logsumexp([bm[m]], [myTheta.omega[m]])
    
    #Calculate log of the denominator
    #Take the column vector of omegas and repeat each column T times
    #to make a MxT omega matrix
    thetaMatrix = np.repeat(myTheta.omega, len(precomBMX[0]), axis=1)
    #Each bk is multiplied by wk then they are summed by rows
    denom = logsumexp(bm, axis=0, b=myTheta.omega)
    
    return numerator - denom

def log_p_m_x_vectorized(m, myTheta, precomBMX):
    """
    Returns vector of log_p_m_x values for corresponding component.
    vectorized so it's faster.
    """

    #Calculate log of the numerator
    #This takes the vector of bs for component m and multiplies it with
    #the omega for component m. exp(b) is taken before multiplying
    omega_vector = np.repeat([myTheta.omega[m]], len(precomBMX[0]), axis=1)
    numerator = logsumexp(precomBMX[m], axis=0, b=omega_vector)
    
    #Calculate log of the denominator
    #Take the column vector of omegas and repeat each column T times
    #to make a MxT omega matrix
    thetaMatrix = np.repeat(myTheta.omega, len(precomBMX[0]), axis=1)
    #Each bk is multiplied by wk then they are summed by rows
    denom = logsumexp(precomBMX, axis=0, b=thetaMatrix)
    
    return numerator - denom    
    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    #Do nothing if no log_Bs given
    if(len(log_Bs) == 0):
        return
    
    #Take the column vector of omegas and repeat each column T times
    #to make a MxT omega matrix
    thetaMatrix = np.repeat(myTheta.omega, len(log_Bs[0]), axis=1)
    
    #This gets log(SUM(wm * bm))
    logsum = logsumexp(log_Bs, axis=0, b=thetaMatrix)
    
    return sum(logsum)

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta( speaker, M, X.shape[1] )
    
    #Initialize values of myTheta
    #Random vector for each mu
    for i in range(0, len(myTheta.mu)):
        rand_v = random.randint(0, X.shape[0] - 1)
        myTheta.mu[i] = X[rand_v]
    #Initialize all sigmas to 1
    for i in range(0, len(myTheta.Sigma)):
        for j in range(0, len(myTheta.Sigma[i])):
            myTheta.Sigma[i][j] = 1
    #Initialize all omegas to 1 / M
    for i in range(0, len(myTheta.omega)):
        myTheta.omega[i] = 1 / M
    
    #Now do the iterations
    prev_L = float('-inf')
    improvement = float('inf')
    i = 0
    log_bs = np.zeros((M,len(X)))
    log_ps = np.zeros((M,len(X)))
    while(i < maxIter and improvement >= epsilon):
        print("Iteration", i, improvement)
        #Precompute values
        precom = []
        for comp in range(0, M):
            precom.append(precompute(myTheta.mu[comp], myTheta.Sigma[comp]))
            
        #Get b values
        for comp in range(0, M):
            log_bs[comp] = log_b_m_x(comp, X, myTheta, precom)
        #Get p values
        for comp in range(0, M):
            log_ps[comp] = log_p_m_x_vectorized(comp, myTheta, log_bs)
        
        #Get log like
        new_L = logLik(log_bs, myTheta)
        improvement = new_L - prev_L
        prev_L = new_L
        
        #New omegas
        sumlogps = np.exp(log_ps).sum(axis=0)
        for component in range(0, M):
            myTheta.omega[component][0] = sumlogps[component] / len(X)
        
        #New means and sigmas, could not figure out how to vectorize
        for component in range(0, M):
            newMu = X[0]
            newSigma = X[0]
            
            #Get numerators
            newMu = (np.exp(log_ps[component]).reshape(len(X), 1) * X).sum(axis=0)
            newSigma = (np.exp(log_ps[component]).reshape(len(X), 1) * (X ** 2)).sum(axis=0)
                
            #Divide by denominators
            newMu = newMu / (myTheta.omega[component] * len(X))
            newSigma = (newSigma / (myTheta.omega[component] * len(X))) - (newMu ** 2)

            """for z in range(0, len(newSigma)):
                if newSigma[z] < 0:
                    print(newSigma[z])
                    newSigma[z] = 1"""
            
            myTheta.mu[component] = newMu
            myTheta.Sigma[component] = newSigma
        
        i = i + 1
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    #First iterate through all models and within that iterate through all components
    #This is to find the log_bs
    #Then use that to get all the log likelihoods
    likelihoods = []
    m = 0
    for model in models:
        log_bs = []
        for component in range(0, len(model.omega)):
            log_bs.append(log_b_m_x(component, mfcc, model))
        likelihoods.append((m, logLik(log_bs, model)))
        m = m + 1
        
    top = sorted(likelihoods, reverse=True, key=lambda x: x[1])
    
    bestModel = top[0][0]
    
    #Print out to stdout the k best
    print("[", correctID, "]")
    for count in range(0, k):
        if count >= len(top):
            break
        print("[", top[count][0], "]", " [", top[count][1], "]")
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)
                
            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k )
        print("\n")
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print("Accuracy:", accuracy)