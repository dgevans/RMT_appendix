# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 17:19:11 2014

@author: anmol
"""

import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import math
data_riskaversion_3shocks = pickle.load( open( 'dataSimulation_riskaversion_3shocks.dat', "rb" ) )    
data_riskaversion_2shocks = pickle.load( open( 'dataSimulation_riskaversion_2shocks.dat', "rb" ) )    
data_ql_2shocks = pickle.load( open( 'dataSimulation_ql_2shocks.dat', "rb" ) )    



_,_,_,tauHist_riskaversion_2shocks,bHist_riskaversion_2shocks=data_riskaversion_2shocks
_,_,_,tauHist_riskaversion_3shocks,bHist_riskaversion_3shocks=data_riskaversion_3shocks
_,_,_,tauHist_ql_2shocks,bHist_ql_2shocks=data_ql_2shocks

t=100
T=75000
freq=100
plot_seq=np.array(map(math.modf,(np.linspace(t,T,freq))),dtype='int')[:,1]


f,(ax1,ax2) =plt.subplots(2,1,sharex='col')
lines_taxes=ax1.plot(np.vstack((tauHist_ql_2shocks[plot_seq],tauHist_riskaversion_2shocks[plot_seq],tauHist_riskaversion_3shocks[plot_seq])).T)
lines_debt=ax2.plot(np.vstack((bHist_ql_2shocks[plot_seq],bHist_riskaversion_2shocks[plot_seq],bHist_riskaversion_3shocks[plot_seq])).T)
#lines_debt=ax3. plot(np.array([debt_long_sample_no_agg_shock,debt_long_sample_with_agg_shock]).T)

    
plt.setp(lines_taxes[1],color='k',linewidth=2)
plt.setp(lines_taxes[0],color='k',linewidth=2,linestyle='--')
plt.setp(lines_taxes[2],color='r',linewidth=2,linestyle='-.')


plt.setp(lines_debt[1],color='k',linewidth=2)
plt.setp(lines_debt[0],color='k',linewidth=2,linestyle='--')
plt.setp(lines_debt[2],color='r',linewidth=2,linestyle='-.')

    
ax1.set_title(r'Taxes')
ax2.set_title(r'debt')
#ax3.set_title(r'Debt')


#ax1.legend(('no agg shocks','with agg shocks'),loc=0)
plt.xlabel('t')    
plt.savefig('policy_long_sample.png',dpi=300)
   

