# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 07:44:43 2016

@author: jlao
"""

import numpy as np, pymc3 as pm, theano.tensor as T

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
# generate data
Pheno   = np.dot(X,w0)

mixedEffect_model = pm.Model()

with pm.Model() as mixedEffect_model:
    ### hyperpriors
    h2     = pm.Uniform('h2')
    sigma2 = pm.HalfCauchy('eps', 5)
    #beta_0 = pm.Uniform('beta_0', lower=-1000, upper=1000)   # a replacement for improper prior
    w = pm.Normal('w', mu = 0, sd = 100, shape=(M,1))
    z = pm.Normal('z', mu = 0, sd= (h2*sigma2)**0.5 , shape=(N,1))
    g = T.dot(L,z)
    y = pm.Normal('y', mu = g + T.dot(X,w), 
                  sd= ((1-h2)*sigma2)**0.5 , observed=Pheno )