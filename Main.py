__author__ = 'dgevans'
from parameters import parameters
import numpy as np
import bellman
import initialize
import LucasStockey as LS
from parameters import UCES
from parameters import UQL

    

Para = parameters()
Para.g = [.1,.15]
Para.theta = 1.
Para.theta = np.array([1.,1.1])
Para.P = np.ones((2,2))/2
S = len(Para.P)
#Para.P = np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
Para.U = UCES
#Para.P = np.array([[.6,.4],[.4,.6]])
Para.beta = np.array([.95])
Para.sigma = 2.
Para.gamma = 2.
Para.nx = 100
S = Para.P.shape[0]
Para.xmax = 2.5
Para.xmin = -2.
Para.transfers = False


##Setup grid and initialize value function
#Setup
Para = initialize.setupGrid(Para)

Para.bounds = [(0,10)]*S+[(Para.xmin,Para.xmax)]*S
Vf,c_policy,xprime_policy = initialize.initializeWithCM(Para)

#Iterate until convergence
coef_old = np.zeros((Para.nx,S))
for s in range(0,S):
    coef_old[:,s] = Vf[s].getCoeffs()

Nmax = 150

diff = []
for i in range(0,Nmax):
    Vf,c_policy,xprime_policy = bellman.iterateBellmanLocally(Vf,c_policy,xprime_policy,Para)
    diff.append(0)
    for s_ in range(0,S):
        diff[i] = max(diff[i],np.max(np.abs(coef_old[:,s_]-Vf[s_].getCoeffs())))
        coef_old[:,s_] = Vf[s_].getCoeffs()
    print diff[i]

xHist,sHist = bellman.simulate(0.,1000,xprime_policy,Para)

