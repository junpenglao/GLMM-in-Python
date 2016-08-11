# -*- coding: utf-8 -*-
"""
example from pymc-dev
"""
import pymc3 as pm
import theano.tensor as T
from numpy import random, sum as nsum, ones, concatenate, newaxis, dot, arange
import numpy as np

random.seed(1)

n_groups = 10
no_pergroup = 30
n_observed = no_pergroup * n_groups
n_group_predictors = 1
n_predictors = 3

group = concatenate([[i] * no_pergroup for i in range(n_groups)])
group_predictors = random.normal(size=(n_groups, n_group_predictors))  # random.normal(size = (n_groups, n_group_predictors))
predictors = random.normal(size=(n_observed, n_predictors))

group_effects_a = random.normal(size=(n_group_predictors, n_predictors))
effects_a = random.normal(
    size=(n_groups, n_predictors)) + dot(group_predictors, group_effects_a)

y = nsum(
    effects_a[group, :] * predictors, 1) + random.normal(size=(n_observed))


model = pm.Model()
with model:

    # m_g ~ N(0, .1)
    group_effects = pm.Normal(
        "group_effects", 0, .1, shape=(n_group_predictors, n_predictors))
    gp = pm.Normal("gp", 0, .1, shape=(n_groups,1))
    # gp = group_predictors
    # sg ~ Uniform(.05, 10)
    sg = pm.Uniform("sg", .05, 10, testval=2.)
    

    # m ~ N(mg * pg, sg)
    effects = pm.Normal("effects",
                     T.dot(gp, group_effects), sg ** -2,
                     shape=(n_groups, n_predictors))

    s = pm.Uniform("s", .01, 10, shape=n_groups)

    g = T.constant(group)

    # y ~ Normal(m[g] * p, s)
    mu_est = pm.Deterministic("mu_est",T.sum(effects[g] * predictors, 1))
    yd = pm.Normal('y',mu_est , s[g] ** -2, observed=y)

    start = pm.find_MAP()
    #h = find_hessian(start)

    step = pm.NUTS(model.vars, scaling=start)

with model:
    trace = pm.sample(3000, step, start)
        
#%%
pm.traceplot(trace)
dftmp = pm.df_summary(trace,varnames=['group_effects'])
print(dftmp['mean'])
import statsmodels.formula.api as smf
# from patsy import dmatrices
import pandas as pd
tbl = pd.DataFrame(predictors,columns=['C1','C2','C3'])
tbl['group'] = pd.Series(group, dtype="category")
tbl['yd']    = y
md2 = smf.mixedlm("yd ~ -1 + C1 + C2 + C3", tbl, groups=tbl["group"])
mdf2= md2.fit()
print(mdf2.summary())
#%%
X      = np.tile(group_predictors[group],(1,3)) * predictors
beta0  = np.linalg.lstsq(X,y)
fitted = np.dot(X,beta0[0])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(y,'k')
plt.plot(fitted,'g')


dftmp = pm.df_summary(trace[1000:],varnames=['mu_est'])
testdf = np.asarray(dftmp['mean'])
plt.plot(testdf,'r')
plt.legend(['observed',str(np.mean(np.square(y-fitted))),str(np.mean(np.square(y-testdf)))])
