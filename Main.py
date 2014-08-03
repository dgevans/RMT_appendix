__author__ = 'dgevans'
from parameters import parameters
import numpy as np
import bellman
import initialize
from parameters import UCES
from parameters import UQL
import matplotlib.pyplot as plt

    
"""SETUP: This section creates an instance of the parameter class that holds the
primitives of the model - Preferences, Technology and details about the approximation.
"""

Para = parameters() # Creates an instance of the parameter class. For details see module parameter.py

Para.g = [.1,.15] # Grid for expenditure shocks
Para.theta = 1. # labor productivity
Para.P = np.ones((2,2))/2 #Transition matrix for the Markov process on g 
Para.U = UQL # utility specification. For other choices see parameter.py 
Para.beta = 0.95 # Discount factor
Para.sigma = 0. # risk aversion. This is redundant for quasi-linear preferences
Para.gamma = 2.# inverse Frish elasticity of labor
Para.nx = 100 # number of grid points on the grid for x
Para.transfers = True#Flag that indicates whether to solve the model with or without Transfers#



"""INITILIZATION: Here we initialize the continuation value function V(x,s). 
A good initial guess is obtained using the Lucas Stokey allocation associated with every (x,s) in the state space"""

#Setup grid and initialize value function. 
Para.xmax=None
Para.xmin=None
Para = initialize.setupGrid(Para)
# Initialize with LS solution. This also creates the basis for splines that will store the approximations for V(x,s) and policy rules
Vf,c_policy,xprime_policy = initialize.initializeFunctions(Para)



"""VALUE FUNCTION ITERATION: Here we iterate on the bellman equation that defines V(x,s). 
The module bellman.py has methods that can solve for the optimal policies at every point on the grid (x,s) for a given guess of V(x,s). 
Using these policies we recompute the values and apprximate it again with splines."""

#Iterate until convergence
S = Para.P.shape[0]
coef_old = np.zeros((Para.nx,S))
for s in range(0,S):
    coef_old[:,s] = Vf[s].getCoeffs()

Nmax = 150

diff = []
for i in range(0,Nmax):
    Vf,c_policy,xprime_policy = bellman.iterateBellman(Vf,c_policy,xprime_policy,Para)
    diff.append(0)
    for s_ in range(0,S):
        diff[i] = max(diff[i],np.max(np.abs(coef_old[:,s_]-Vf[s_].getCoeffs())))
        coef_old[:,s_] = Vf[s_].getCoeffs()
    print 'Iteration number {0} error: {1}'.format(i,diff[i])



"""SIMULATIONS: Here we use the solution for V(x,s) and associated policy rules to generate a sample path for 
taxes, debt and state variables (x_t,s_t)"""

T=100000
xHist,cHist,sHist = bellman.simulate(0.,T,xprime_policy,c_policy,Para)
lHist = (cHist+[Para.g[sHist[i]] for i in range(T)])/(Para.theta)
ucHist = Para.U.uc(cHist,lHist,Para)
ulHist = Para.U.ul(cHist,lHist,Para)

tauHist=1+ulHist/(Para.theta*ucHist)
bHist=xHist/ucHist

plt.figure()
plt.plot(tauHist,'k')
plt.title('Tax rates')
plt.xlabel('t')
plt.ylabel(r'$\tau$')
plt.savefig('taxes.png',dpi=300)
    
plt.figure()
plt.plot(bHist,'k')
plt.title('debt')
plt.xlabel('t')
plt.ylabel(r'$b$')
plt.savefig('debt.png',dpi=300)
