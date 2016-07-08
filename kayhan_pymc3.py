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
w0 = [5,1,2,3,1,1]
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
            t = trace[var][i]
        else:
            t = trace[var]
        sns.distplot(t, kde=False, norm_hist=True, ax=ax)
fig.tight_layout()
#%%
pm.traceplot(trace, combined=True)
plt.show()

burnin = 0
df_summary1 = pm.df_summary(trace[burnin:],varnames=['w'])
wpymc = np.asarray(df_summary1['mean'])
df_summary2 = pm.df_summary(trace[burnin:],varnames=['z'])
zpymc = np.asarray(df_summary2['mean'])

import statsmodels.formula.api as smf
tbltest['Pheno'] = Pheno
md  = smf.mixedlm("Pheno ~ Condi1*Condi2", tbltest, groups=tbltest["subj"])
mdf = md.fit()
fixed = np.asarray(mdf.fe_params).flatten()

plt.figure()
plt.plot(w0,'r')
plt.plot(wpymc,'b')
plt.plot(fixed,'g')
plt.legend(['real','PyMC','LME'])


plt.figure()
plt.plot(Pheno,'r')
fitted1=np.dot(X,wpymc).flatten()+np.dot(L,zpymc).flatten()
plt.plot(fitted1,'b')
fitted2=np.dot(X,mdf.fe_params)+np.dot(L,mdf.random_effects).flatten()
plt.plot(fitted2,'g')
plt.legend(['Observed','PyMC','LME'])
