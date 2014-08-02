__author__ = 'dgevans'
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def findBondSteadyState(Para):
    def xBondResidual(mu):
        c,l = solveLSmu(mu,Para)
        uc = Para.U.uc(c,l,Para)
        I = Para.I(c,l)
        Euc = Para.P[0,:].dot(uc)
        x= I/(uc/Euc-Para.beta)
        return x
    S = len(Para.P)
    if S == 2:
        def f(mu):
            x = xBondResidual(mu)
            return x[1]-x[0]
    else:
        def f(mu):
            x = xBondResidual(mu)
            xbar = Para.P[0,:].dot(x)
            return np.linalg.norm(x-xbar)
    muSS = root(f,0.0).x
    cSS,lSS = solveLSmu(muSS,Para)
    ucSS = Para.U.uc(cSS,lSS,Para)
    ISS = Para.I(cSS,lSS)
    EucSS = Para.P[0,:].dot(ucSS)
    xSS= ISS/(ucSS/EucSS-Para.beta)
    return muSS,cSS,lSS,xSS
    
def solveLucasStockey(x,s_,Para):
    S = Para.P.shape[0]
    def x_mu(mu):
        c,l = solveLSmu(mu,Para)
        I = c*Para.U.uc(c,l,Para)+l*Para.U.ul(c,l,Para)
        return Para.P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,I))

    mu_SL = root(lambda mu: x_mu(mu)-x,0).x

    return solveLSmu(mu_SL,Para)
    
    
def findSteadyStateb0(Para):
    muSS,cSS,lSS,xSS = findBondSteadyState(Para)
    
    def f(b0):
        cSS0,lSS0 = solveLSmuTime0(muSS,Para,b0)
        uc0 = Para.U.uc(cSS0,lSS0,Para)
        ISS0 = Para.I(cSS0,lSS0)
        ISS = Para.I(cSS,lSS)
        EISS = Para.P[0,:].dot(ISS)
        return b0 - (ISS0[0]+Para.beta*EISS/(1-Para.beta))/uc0[0]
        
    b0 = root(f,0.0).x
    cSS0,lSS0 = solveLSmuTime0(muSS,Para,b0)
    return b0,muSS,cSS0,lSS0,cSS,lSS


def LSResiduals(z,mu,Para):
    S = Para.P.shape[0]
    c = z[0:S]
    l = z[S:2*S]
    uc = Para.U.uc(c,l,Para)
    ucc = Para.U.ucc(c,l,Para)
    ul = Para.U.ul(c,l,Para)
    ull = Para.U.ull(c,l,Para)

    res = Para.theta*l-c-Para.g
    foc_c = uc -(uc+ucc*c)*mu
    foc_l = (ul-(ul+ull*l)*mu)/Para.theta

    return np.hstack((res,foc_c+foc_l))
    
def LSResidualsTime0(z,mu,Para,b0):
    S = Para.P.shape[0]
    c = z[0:S]
    l = z[S:2*S]
    uc = Para.U.uc(c,l,Para)
    ucc = Para.U.ucc(c,l,Para)
    ul = Para.U.ul(c,l,Para)
    ull = Para.U.ull(c,l,Para)

    res = Para.theta*l-c-Para.g
    foc_c = uc -(uc+ucc*c)*mu+b0*mu*ucc
    foc_l = (ul-(ul+ull*l)*mu)/Para.theta

    return np.hstack((res,foc_c+foc_l))

def solveLSmu(mu,Para):
    S = Para.P.shape[0]
    f = lambda z: LSResiduals(z,mu,Para)

    z_mu = root(f,0.5*np.ones(2*S)).x

    return z_mu[0:S],z_mu[S:2*S]
    
def solveLSmuTime0(mu,Para,b0):
    S = Para.P.shape[0]
    f = lambda z: LSResidualsTime0(z,mu,Para,b0)

    z_mu = root(f,0.5*np.ones(2*S)).x

    return z_mu[0:S],z_mu[S:2*S]


    
def solveForLSmu(x,s_,Para):
    S = Para.P.shape[0]
    def x_mu(mu):
        c,l = solveLSmu(mu,Para)
        I = c*Para.U.uc(c,l,Para)+l*Para.U.ul(c,l,Para)
        return Para.P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,I))

    return root(lambda mu: x_mu(mu)-x,0).x

def solveLucasStockey_alt(x,Para):
    S = Para.P.shape[0]
    def LSres(z):
        beta = Para.beta
        c = z[0:S]
        mu = z[S:2*S]
        xi = z[2*S:3*S]
        l = (c+Para.g)/Para.theta
        xprime = np.zeros(S)
        for s in range(0,S):
            [cprime,lprime] = solveLSmu(mu[s],Para)
            Iprime = c*Para.U.uc(cprime,lprime,Para)+lprime*Para.U.ul(cprime,lprime,Para)
            xprime[s] = Para.P[0,:].dot( np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,Iprime))
        uc = Para.U.uc(c,l,Para)
        ucc = Para.U.ucc(c,l,Para)
        ul = Para.U.ul(c,l,Para)
        ull = Para.U.ull(c,l,Para)
        res = np.zeros(3*S)
        Euc = Para.P[0,:].dot(uc)
        mu_ = Para.P[0,:].dot(uc*mu)/Euc

        res[0:S] = c*uc+l*ul+xprime-x*uc/(beta*Euc)
        res[S:2*S] = uc-mu*( c*ucc+uc )+x*ucc/(beta*Euc) * (mu-mu_)-xi
        res[2*S:3*S] = ul - mu*( l*ull+ul ) + Para.theta * xi
        return res
    z0 = [0.5]*S+[0]*2*S
    z_SL = root(LSres,z0).x

    cLS = z_SL[0:S]
    lLS = (cLS+Para.g)/Para.theta

    return cLS,lLS,Para.I(cLS,lLS)
    
def LSxmu(x,mu,Para):
    c,l = solveLSmu(mu,Para)
    I = Para.I(c,l)
    uc = Para.U.uc(c,l,Para)
    Euc= Para.P[0,:].dot(uc)
    return x*uc/(Para.beta*Euc)-I
    
def doPortfolioPlot(Para):
    Uc = Para.U.uc
    xvec = np.linspace(Para.xmin,Para.xmax,1000)
    S = len(Para.P)    
    def getPortfolioDiff(x):
        c,l = solveLucasStockey(x,0,Para)            
        uc = Uc(c,l,Para)
        I = Para.I(c,l)
        aprime = np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,I)/uc
        price_aprime = Para.P[0,:].dot(aprime*uc)
        price_bond = Para.P[0,:].dot(np.ones(S)*uc)
        return aprime/price_aprime - np.ones(S)/price_bond
    f = lambda x: np.linalg.norm(getPortfolioDiff(x))
    plt.plot(xvec,map(f,xvec))
    _,_,_,xSS = findBondSteadyState(Para)
    xSS = [max(xSS),min(xSS)]
    plt.plot(xSS,map(f,xSS),'kx')
    
    plt.show()
    
def getPortfolio(x,Para):
    c,l = solveLucasStockey(x,0,Para)            
    uc = Para.U.uc(c,l,Para)
    I = Para.I(c,l)
    S = len(Para.P)
    aprime = np.linalg.solve(np.eye(S)-(Para.beta*Para.P.T).T,I)/uc
    if aprime[0] > 0:
        return aprime/np.linalg.norm(aprime)
    else:
        return -aprime/np.linalg.norm(aprime)
        
def CMPolicy(state,Para):
    '''
    Computes the complete markets policy
    '''
    mu,s_ = state
    c,l = solveLSmu(mu,Para)
    Uc = Para.U.uc(c,l,Para)
    Ucc = Para.U.ucc(c,l,Para)
    I = Para.I(c,l)
    P = Para.P
    S = len(P)
    x = Para.beta*Para.P[s_,:].dot(np.linalg.solve(np.eye(S)-Para.beta*P,I))
    xi = Uc-mu*( Ucc*c+Uc )
    return np.hstack((c,l,mu*np.ones(S),xi,x))