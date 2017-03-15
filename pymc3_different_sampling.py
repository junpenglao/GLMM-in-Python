# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 07:44:43 2016

@author: jlao
"""

import numpy as np, pymc3 as pm, theano.tensor as T, matplotlib.pyplot as plt

M    = 6  # number of columns in X - fixed effect
N    = 10 # number of columns in L - random effect
nobs = 10

# generate design matrix using patsy
from patsy import dmatrices
import pandas as pd
predictors = []
for s1 in range(N):
    for c1 in range(2):
        for c2 in range(3):
            for i in range(nobs):
                predictors.append(np.asarray([c1+1,c2+1,s1+1]))
tbltest             = pd.DataFrame(predictors, columns=['Condi1', 'Condi2', 'subj'])
tbltest['Condi1']   = tbltest['Condi1'].astype('category')
tbltest['Condi2']   = tbltest['Condi2'].astype('category')
tbltest['subj']     = tbltest['subj'].astype('category')
tbltest['tempresp'] = np.random.normal(size=(nobs*M*N,1))

Y, X    = dmatrices("tempresp ~ Condi1*Condi2", data=tbltest, return_type='matrix')
Terms   = X.design_info.column_names
_, L    = dmatrices('tempresp ~ -1+subj', data=tbltest, return_type='matrix')
X       = np.asarray(X) # fixed effect
L       = np.asarray(L) # mixed effect
Y       = np.asarray(Y) 
# generate data
w0 = [5.,1.,2.,3.,1.,1.]
z0 = np.random.normal(size=(N,))
Pheno   = np.dot(X,w0) + np.dot(L,z0) + Y.flatten()

with pm.Model() as mixedEffect:
    ### hyperpriors
    h2     = pm.Uniform('h2')
    sigma2 = pm.HalfCauchy('sigma2', 5)
    #beta_0 = pm.Uniform('beta_0', lower=-1000, upper=1000)   # a replacement for improper prior
    w = pm.Normal('w', mu = 0, sd = 100, shape=M)
    z = pm.Normal('z', mu = 0, sd= (h2*sigma2)**0.5 , shape=N)
    g = T.dot(L,z)
    y = pm.Normal('y', mu = g + T.dot(X,w), 
                  sd= ((1-h2)*sigma2)**0.5 , observed=Pheno )
#    like = pm.Potential('like', pm.Normal.dist(mu = g + T.dot(X,w), 
#                  sd= ((1-h2)*sigma2)**0.5).logp(Pheno))
#%% advi
with mixedEffect:
    means, sds, elbos = pm.variational.advi(n=100000)
#%% Metropolis
with mixedEffect:
    trace = pm.sample(50000,step=pm.Metropolis())
#%% NUTS
with mixedEffect:
    trace = pm.sample(3000)
#%% atmcmc
test_folder = ('ATMIP_TEST')
n_chains = 500
n_steps = 100
tune_interval = 25
njobs = 1

from pymc3.step_methods import ATMCMC as atmcmc
with pm.Model() as mixedEffect2:
    ### hyperpriors
    h2     = pm.Uniform('h2', transform=None)
    sigma2 = pm.HalfCauchy('eps', 5, transform=None)
    #beta_0 = pm.Uniform('beta_0', lower=-1000, upper=1000)   # a replacement for improper prior
    w = pm.Normal('w', mu = 0, sd = 100, shape=M)
    z = pm.Normal('z', mu = 0, sd= (h2*sigma2)**0.5 , shape=N)
    g = T.dot(L,z)
    y = pm.Normal('y', mu = g + T.dot(X,w), 
                  sd= ((1-h2)*sigma2)**0.5 , observed=Pheno )
    like = pm.Deterministic('like', h2.logpt + sigma2.logpt + w.logpt + z.logpt + y.logpt)

    step=atmcmc.ATMCMC(n_chains=n_chains, tune_interval=tune_interval, likelihood_name=mixedEffect2.deterministics[0].name)
    
trace = atmcmc.ATMIP_sample(n_steps=n_steps, step=step, njobs=njobs, progressbar=True, model=mixedEffect2, trace='Test')

# trcs = pm.backends.text.load('Test/stage_30',    model=mixedEffect2)
# trcs = pm.backends.text.load('Test/stage_final', model=mixedEffect2)
# Pltr = pm.traceplot(trcs, combined=True)
# plt.show(Pltr[0][0])
#%% plot advi and NUTS (copy from pymc3 example)
burnin = 1000
from scipy import stats
import seaborn as sns
varnames = means.keys()
fig, axs = plt.subplots(nrows=len(varnames), figsize=(12, 18))
for var, ax in zip(varnames, axs):
    mu_arr = means[var]
    sigma_arr = sds[var]
    ax.set_title(var)
    for i, (mu, sigma) in enumerate(zip(mu_arr.flatten(), sigma_arr.flatten())):
        sd3 = (-4*sigma + mu, 4*sigma + mu)
        x = np.linspace(sd3[0], sd3[1], 300)
        y = stats.norm(mu, sigma).pdf(x)
        ax.plot(x, y)
        if trace[var].ndim > 1:
            t = trace[burnin:][var][i]
        else:
            t = trace[burnin:][var]
        sns.distplot(t, kde=True, norm_hist=True, ax=ax)
fig.tight_layout()
#%%
pm.traceplot(trace, combined=True)
plt.show()

df_summary1 = pm.df_summary(trace[burnin:],varnames=['w'])
wpymc = np.asarray(df_summary1['mean'])
df_summary2 = pm.df_summary(trace[burnin:],varnames=['z'])
zpymc = np.asarray(df_summary2['mean'])

w_vi = means['w']
z_vi = means['z']

import statsmodels.formula.api as smf
tbltest['Pheno'] = Pheno
md  = smf.mixedlm("Pheno ~ Condi1*Condi2", tbltest, groups=tbltest["subj"])
mdf = md.fit()
fe_params = pd.DataFrame(mdf.fe_params,columns=['LMM'])
random_effects = pd.DataFrame(mdf.random_effects)
random_effects = random_effects.transpose()
random_effects = random_effects.rename(index=str, columns={'groups': 'LMM'})

fe_params['PyMC'] = pd.Series(wpymc, index=fe_params.index)
random_effects['PyMC'] = pd.Series(zpymc, index=random_effects.index)

fe_params['PyMC_vi'] = pd.Series(w_vi, index=fe_params.index)
random_effects['PyMC_vi'] = pd.Series(z_vi, index=random_effects.index)

# ploting function 
def plotfitted(fe_params,random_effects,X,Z,Y):
    plt.figure(figsize=(18,9))
    ax1 = plt.subplot2grid((2,2), (0, 0))
    ax2 = plt.subplot2grid((2,2), (0, 1))
    ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)
    
    fe_params.plot(ax=ax1)
    random_effects.plot(ax=ax2)
    
    ax3.plot(Y.flatten(),'o',color='k',label = 'Observed', alpha=.25)
    for iname in fe_params.columns.get_values():
        fitted = np.dot(X,fe_params[iname])+np.dot(Z,random_effects[iname]).flatten()
        print("The MSE of "+iname+ " is " + str(np.mean(np.square(Y.flatten()-fitted))))
        ax3.plot(fitted,lw=1,label = iname, alpha=.5)
    ax3.legend(loc=0)
    #plt.ylim([0,5])
    plt.show()

plotfitted(fe_params=fe_params,random_effects=random_effects,X=X,Z=L,Y=Pheno)