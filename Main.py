__author__ = 'dgevans'
from parameters import parameters
import numpy as np
import bellman
import initialize
from parameters import UCES
import matplotlib.pyplot as plt
import cPickle as pickle

    
"""SETUP: This section creates an instance of the parameter class that holds the
primitives of the model - Preferences, Technology and details about the approximation.
"""

Para = parameters() # Creates an instance of the parameter class. For details see module parameter.py
Para.g = [.05,.15] # Grid for expenditure shocks
Para.theta = 1. # labor productivity
Para.P = np.ones((2,2))/2 #Transition matrix for the Markov process on g 
#Para.P==[[.8,.2],[.2,.8]]
Para.U = UCES# utility specification. For other choices see parameter.py 
Para.beta = 0.96 # Discount factor
Para.sigma = 0 # risk aversion. This is redundant for quasi-linear preferences
Para.gamma = 2.# inverse Frish elasticity of labor
Para.nx = 200# number of grid points on the grid for x
Para.transfers = True#Flag that indicates whether to solve the model with or without Transfers#



"""INITILIZATION: Here we initialize the continuation value function V(x,s). 
A good initial guess is obtained using the Lucas Stokey allocation associated with every (x,s) in the state space"""

#Setup grid and initialize value function. 
Para.xmax=2
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

Nmax = 250

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

s_=1
plt.figure()
plt.plot(Para.xgrid, map(xprime_policy[s_,0],Para.xgrid)-Para.xgrid,color='k',linewidth=2)
plt.hold    
plt.plot(Para.xgrid, map(xprime_policy[s_,1],Para.xgrid)-Para.xgrid,color='k',linewidth=2,linestyle='--')
#plt.hold
#plt.plot(Para.xgrid, map(xprime_policy[s_,2],Para.xgrid)-Para.xgrid,'-k')
#plt.title('x_prime')
plt.xlabel(r'$x$')
plt.ylabel(r'$x\'(s)-x$')
plt.legend(['g(s)=g_l','g(s)=g_h'])
plt.savefig('x_prime_policy_ql.png',dpi=300)



b0=.8*np.mean((Para.cFB+Para.g))
s0=0
c0,x0=bellman.solveTime0Problem(b0,s0,Vf,Para)
T=20000
xHist,cHist,sHist = bellman.simulate(x0,T,xprime_policy,c_policy,Para)
cHist[0]=c0
lHist = (cHist+[Para.g[sHist[i]] for i in range(T)])/(Para.theta)
ucHist = Para.U.uc(cHist,lHist,Para)
ulHist = Para.U.ul(cHist,lHist,Para)
t=1
T=20000
tauHist=1+ulHist/(Para.theta*ucHist)
bHist=xHist/ucHist

plt.figure()
plt.plot(tauHist[t:T],'k')
plt.title('Tax rates')
plt.xlabel('t')
plt.ylabel(r'$\tau$')
plt.savefig('taxes.png',dpi=300)
    
plt.figure()
plt.plot(bHist[t:T],'k')
plt.title('debt')
plt.xlabel('t')
plt.ylabel(r'$b$')
plt.savefig('debt.png',dpi=300)


plt.figure()
plt.plot(xHist[t:T],'k')
plt.title('x')
plt.xlabel('t')
plt.ylabel(r'$x$')
plt.savefig('x.png',dpi=300)




dataSimulation=xHist,cHist,sHist,tauHist,bHist,Para
with open('dataSimulation_ql.dat', 'wb') as f:
    pickle.dump(dataSimulation , f)
