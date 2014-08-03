__author__ = 'dgevans'
from scipy.optimize import root
import numpy as np
from bellman import ValueFunctionSpline
from bellman import PolicyRulesSpline
from scipy.optimize import brentq
import parameters
import LucasStockey as LS

def computeFB(Para):
    ###FIND FIRST BEST
    S = Para.P.shape[0]

    #different methods for solving for the FB depending on if we have CES or BGP preferences
    if Para.U == parameters.UCES_AMSS:
        def firstBestRoot(c,Para,s):
            l = (c+Para.g[s])/Para.theta
            return Para.U.uc(c,l,Para)+Para.U.ul(c,l,Para)/Para.theta
        cFB = np.zeros(S)
        for s in range(0,S):
            cFB[s] = brentq(firstBestRoot,0.01,Para.theta-Para.g[s]-.01,args=(Para,s))
    else:
        def firstBestRoot(c,Para):
            l = (c+Para.g)/Para.theta
            return Para.U.uc(c,l,Para)+Para.U.ul(c,l,Para)/Para.theta
        cFB = root(firstBestRoot,0.5*np.ones(S),Para).x

    lFB = (cFB+Para.g)/Para.theta
    ucFB = Para.U.uc(cFB,lFB,Para)
    ulFB = Para.U.ul(cFB,lFB,Para)

    IFB = cFB*ucFB+lFB*ulFB

    EucFB = Para.P.dot(ucFB)
    xFB = IFB/(ucFB/EucFB-Para.beta) #compute x needed to attain FB while keeping x constant
    return cFB,xFB


def setupGrid(Para):
    cFB,xFB = computeFB(Para)
    if Para.xmin == None:
        Para.xmin = min(xFB) #below xmin FB can be acheived

    if Para.xmax == None:
        Para.xmax = -min(xFB) #symmetric grid

    Para.xgrid = np.linspace(Para.xmin,Para.xmax,Para.nx) #linear grid points
    S = Para.P.shape[0]
    xDomain = np.kron(np.ones(S),Para.xgrid) #stack Para.xgrid S times
    s_Domain = np.kron(range(0,S),np.ones(Para.nx,dtype=np.int)) #s assciated with each grid
    Para.domain = zip(xDomain,s_Domain)#zip them together so we have something that looks like
    Para.bounds = [(0,10)]*S+[(Para.xmin,Para.xmax)]*S
    #[(x1,1),(x2,1),...,(xn,1),(x1,2),...]

    return Para

def initializeFunctions(Para):
    #Initializing using deterministic stationary equilibrium but using interest rates comming from FB
    S = Para.P.shape[0]
    cFB,_ = computeFB(Para)

    Q = np.zeros((S*S,S*S)) #full (s_,s) transition matrix
    for s_ in range(0,S):
        for s in range(0,S):
            Q[s_:S*S:S,s_*S+s] = Para.P[s_,s]
    c = np.zeros((Para.nx,S,S))
    xprime = np.zeros((Para.nx,S,S))
    V = np.zeros((Para.nx,S))

    for i in range(0,Para.nx):
        u = np.zeros((S,S))
        for s_ in range(0,S):
            x = Para.xgrid[i]            
            def stationaryRoot(c):
                l = (c+Para.g)/Para.theta
                Uc = Para.U.uc(c,l,Para)
                EUc = Para.P[s_,:].dot(Uc)
                return c*Para.U.uc(c,l,Para)+l*Para.U.ul(c,l,Para)+(Para.beta-Uc/(EUc))*x
            res = root(stationaryRoot,cFB) #find root that holds x constant
            if not res.success:
                raise Exception(res.message)#find root that holds x constant
            c[i,s_,:] =res.x
            xprime[i,:] = x*np.ones(S)
            if Para.transfers:
                for s in range(0,S):
                    c[i,s_,s] = min(c[i,s_,s],cFB[s])#if you can achieve FB do it
                    
                #Compute xprime from implementability constraint
                l = (c[i,s_,:]+Para.g)/Para.theta
                Uc = Para.U.uc(c[i,s_,:],l,Para)
                EUc = Para.P[s_,:].dot(Uc)
                xprime[i,s_,:] = (c[i,s_,:]*Para.U.uc(c[i,s_,:],l,Para)+l*Para.U.ul(c[i,s_,:],l,Para)-x*Uc/EUc)/Para.beta
                
            l = (c[i,s_,:]+Para.g)/Para.theta
            u[s_,:] = Para.U.u(c[i,s_,:],l,Para)
        for s_ in range(0,S):
            beta = Para.beta
            #compute Value using transition matricies.  Gives rough guess on value function
            v = np.linalg.solve(np.eye(S*S) - beta*Q,u.reshape(S*S))
            V[i,s_] = Para.P[s_,:].dot(v[s_*S:(s_+1)*S])

    #Fit functions using splines.  Linear for policies as they can be wonky
    Vf = []
    c_policy = {}
    xprime_policy = {}
    for s_ in range(0,S):
        beta = Para.beta
        Vf.append(ValueFunctionSpline(Para.xgrid,V[:,s_],[2],Para.sigma,beta))
        for s in range(0,S):
            c_policy[(s_,s)] = PolicyRulesSpline(Para.xgrid,c[:,s_,s],[2])
            xprime_policy[(s_,s)] = PolicyRulesSpline(Para.xgrid,xprime[:,s_,s],[2])

    return Vf,c_policy,xprime_policy
    
    
def initializeWithCM(Para):
    '''
    Initialize value function and policies with the complete markets solution
    '''
    #Initializing using deterministic stationary equilibrium but using interest rates comming from FB
    S = Para.P.shape[0]
    beta = Para.beta
    P = Para.P
    k=Para.k    
    cFB,_ = computeFB(Para)
    lFB = (cFB+Para.g)/Para.theta  
    Para.cFB=cFB
    Para.lFB=lFB
    c = np.zeros((Para.nx,S,S))
    xprime = np.zeros((Para.nx,S,S))
    V = np.zeros((Para.nx,S))
    for s_ in range(0,S):
        for i in range(0,Para.nx):
            x = Para.xgrid[i]
            cLS,lLS = LS.solveLucasStockey(x,s_,Para)
            uLS = Para.U.u(cLS,lLS,Para)
            V[i,s_] = P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*P.T).T,uLS))
            V[i,s_]=min(V[i,s_],P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*P.T).T,Para.U.u(cFB,lFB,Para))))
            xprime[i,s_,:] = x
            c[i,s_,:] = cLS
            if Para.transfers:
                for s in range(0,S):
                    c[i,s_,s] = min(c[i,s_,s],cFB[s])#if you can achieve FB do it                    
                    #Compute xprime from implementability constraint
                    l = (c[i,s_,:]+Para.g)/Para.theta
                    Uc = Para.U.uc(c[i,s_,:],l,Para)
                    EUc = Para.P[s_,:].dot(Uc)
                    xprime[i,s_,:] = (c[i,s_,:]*Para.U.uc(c[i,s_,:],l,Para)+l*Para.U.ul(c[i,s_,:],l,Para)-x*Uc/EUc)/Para.beta
                    l = (c[i,s_,:]+Para.g)/Para.theta                    
                    V[i,s_] = P[s_,:].dot(np.linalg.solve(np.eye(S)-(Para.beta*P.T).T,Para.U.u(c[i,s_,:],l,Para)))                   
    #Fit functions using splines.  Linear for policies as they can be wonky
    Vf = []
    c_policy = {}
    xprime_policy = {}
    for s_ in range(0,S):
        beta = (Para.P[s_,:]*Para.beta).sum() # Anmol: why s this necessary?
        Vf.append(ValueFunctionSpline(Para.xgrid,V[:,s_],k,Para.sigma,beta))
        for s in range(0,S):
            c_policy[(s_,s)] = PolicyRulesSpline(Para.xgrid,c[:,s_,s],k)
            xprime_policy[(s_,s)] = PolicyRulesSpline(Para.xgrid,xprime[:,s_,s],k)

    return Vf,c_policy,xprime_policy