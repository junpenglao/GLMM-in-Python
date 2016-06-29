# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:41:42 2016

@author: jlao
"""
#%%
import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from theano import tensor as tt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# load data
import pandas as pd
Tbl_beh  = pd.read_csv('./behavioral_data.txt', delimiter='\t')
Tbl_beh["subj"]  = Tbl_beh["subj"].astype('category')
#%% visualized data
# Draw a nested violinplot and split the violins for easier comparison
sns.set(style="ticks", palette="muted", color_codes=True)
plt.figure()
Tbl_beh['cbcond']  = pd.Series(Tbl_beh['chimera'] + '-' + Tbl_beh['identity'], index=Tbl_beh.index)
sns.violinplot(y="cbcond", x="rt", hue="group", data=Tbl_beh, split=True,
               inner="quart", palette={"cp": "b", "cg": "y"})
sns.despine(left=True)
#%%
tbltmp  = Tbl_beh.copy(deep=True)
sizetbl = tbltmp.shape
tbltmp['trialcount']  = pd.Series(np.ones((sizetbl[0],),dtype=np.int8), index=tbltmp.index)

tbltmp.set_index(['subj', 'chimera','identity','orientation','group'], inplace=True)
tblmean = tbltmp.mean(level=[0,1,2,3,4])
tblmean = tblmean.reset_index()
tblmean["subj"] = tblmean["subj"].astype('category')
tblsum  = tbltmp.sum(level=[0,1,2,3,4])
tblsum  = tblsum.reset_index()
tblsum["subj"]  = tblsum["subj"].astype('category')
#%%
tbltest = Tbl_beh
tbltest['cbcond']  = pd.Series(tbltest['chimera'] + '-' + tbltest['identity'], 
                index=tbltest.index)

## boxplot + scatter plot to show accuracy
#ax = sns.boxplot(y="cbcond", x="acc", data=tbltest,
#                 whis=np.inf, color="c")
## Add in points to show each observation
#sns.stripplot(y="cbcond", x="acc", data=tbltest,
#              jitter=True, size=3, color=".3", linewidth=0)
plt.figure()
g1 = sns.violinplot(y="cbcond", x="rt", hue="group", data=tbltest, split=True,
               inner="quart", palette={"cp": "b", "cg": "y"})
g1.set(xlim=(0, 3000))
# Make the quantitative axis logarithmic
sns.despine(trim=True)

#%% using statmodels to perform a linear mixed model of reaction time
import statsmodels.formula.api as smf
from patsy import dmatrices
formula = "rt ~ group*orientation*identity"
#formula = "rt ~ -1 + cbcond"
md  = smf.mixedlm(formula, tbltest, groups=tbltest["subj"])
mdf = md.fit()
print(mdf.summary())
#%% GLMM with pymc3 for reaction time data
Y, X   = dmatrices(formula, data=tbltest, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('rt ~ -1+subj', data=tbltest, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)

# In order to convert the upper triangular correlation values to a complete
# correlation matrix, we need to construct an index matrix:
n_var     = nfixed[1] # 4 within group condition that should be correlated
n_elem    = int(n_var * (n_var - 1) / 2)
tri_index = np.zeros([n_var, n_var], dtype=int)
tri_index[np.triu_indices(n_var, k=1)]       = np.arange(n_elem)
tri_index[np.triu_indices(n_var, k=1)[::-1]] = np.arange(n_elem)
beta0     = np.linalg.lstsq(X,Y)

fixedpred = np.argmax(X,axis=1)
randmpred = np.argmax(Z,axis=1)

con  = tt.constant(fixedpred)
sbj  = tt.constant(randmpred)
import pymc3 as pm
with pm.Model() as glmm1:
    # Fixed effect
    beta = pm.Normal('beta', mu = 0, sd = 100, shape=(nfixed[1]))
    # random effect
    s    = pm.HalfCauchy('s',50,shape=(nrandm[1]))
    b    = pm.Normal('b', mu = 0, sd = s, shape=(nrandm[1]))
    eps  = pm.HalfCauchy('eps', 5)
    
    #mu_est = pm.Deterministic('mu_est',beta[con] + b[sbj])
    mu_est = pm.Deterministic('mu_est',tt.dot(X,beta)+tt.dot(Z,b))
    RT = pm.Normal('RT', mu_est, eps, observed = Y)
    
    # start = pm.find_MAP()
    # h = find_hessian(start)

with glmm1:
    # means, sds, elbos = pm.variational.advi(n=100000)
    trace = pm.sample(50000,step=pm.Metropolis())
    
pm.traceplot(trace,varnames=['beta','b','s']) # 
plt.show()
burnin1 = 20000
df_summary = pm.df_summary(trace[burnin1:],varnames=['beta'])
df_summary.index=Terms
print(df_summary)
#%%
Y, X   = dmatrices(formula, data=tbltest, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('rt ~ -1+subj', data=tbltest, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)

X = X.astype(np.float32)
Z = Z.astype(np.float32)
Y = Y.astype(np.float32)

import theano
theano.config.compute_test_value = 'ignore'

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, beta, Z, b):
    return tt.sum(tt.dot(X, beta) + tt.dot(Z, b),axis=1)

def sgd(cost, params, lr=0.001):
    grads = tt.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates
    
Xtf   = tt.fmatrix()
Ztf   = tt.fmatrix()
y     = tt.vector()
tbeta = init_weights([nfixed[1], 1])
tb    = init_weights([nrandm[1], 1])
eps   = init_weights([0])
y_    = model(Xtf, tbeta, Ztf,tb)

cost    = tt.mean(tt.sqr(y - y_))
params  = [tbeta, tb]
updates = sgd(cost, params)

train = theano.function(inputs=[Xtf, Ztf, y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(50000):
    sel = np.random.randint(0,nfixed[0],size=int(nfixed[0]/2))
    batch_xs, batch_zs, batch_ys = X[sel,:],Z[sel,:],Y[sel]    
    train(batch_xs, batch_zs, batch_ys)
        
print (tbeta.get_value())
print (tb.get_value())
#%%
Y, X   = dmatrices(formula, data=tbltest, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('rt ~ -1+subj', data=tbltest, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y)
nfixed = np.shape(X)
nrandm = np.shape(Z)

X = X.astype(np.float32)
Z = Z.astype(np.float32)
Y = Y.astype(np.float32)

import tensorflow as tf
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, beta, Z, b):
    y_pred = tf.matmul(X, beta) + tf.matmul(Z, b)
    #randcoef = tf.matmul(Z, b)
    #Xnew     = tf.transpose(X) * tf.transpose(randcoef)
    #y_pred   = tf.matmul(tf.transpose(Xnew), beta)
    return y_pred # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

Xtf   = tf.placeholder("float32", [None, nfixed[1]]) # create symbolic variables
Ztf   = tf.placeholder("float32", [None, nrandm[1]])
y     = tf.placeholder("float32", [None, 1])
beta  = init_weights([nfixed[1], 1])
b     = init_weights([nrandm[1], 1])
eps   = init_weights([0])
y_    = model(Xtf, beta, Ztf, b)
# y_    = tf.nn.softmax(model(Xtf, beta) + model(Ztf, b) + eps)

# cost  = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
cost  = tf.square(y - y_) # use square error for cost function

train_step    = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

init          = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver         = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        sel = np.random.randint(0,nfixed[0],size=int(nfixed[0]/2))
        batch_xs, batch_zs, batch_ys = X[sel,:],Z[sel,:],Y[sel]
        sess.run(train_step, feed_dict={Xtf: batch_xs, Ztf: batch_zs, y: batch_ys})
        
        accuracy = tf.reduce_mean(tf.cast(cost, tf.float32))
        if i % 1000 == 0:        
            print(i,sess.run(accuracy, feed_dict={Xtf: X, Ztf: Z, y: Y}))

    betaend = sess.run(beta)
    bend    = sess.run(b)
    
outdf = pd.DataFrame(data=betaend,columns=['beta'],index=Terms)
print(outdf)
#%%
yd = np.asarray(tbltest.rt)
plt.figure()
plt.plot(yd,'k')

fitted=np.dot(X,mdf.fe_params)+np.dot(Z,mdf.random_effects).flatten()
plt.plot(fitted,'r')

df_summary1 = pm.df_summary(trace[burnin1:],varnames=['beta'])
betapymc = np.asarray(df_summary1['mean'])
df_summary2 = pm.df_summary(trace[burnin1:],varnames=['b'])
bpymc = np.asarray(df_summary2['mean'])
fitted1=np.dot(X,betapymc).flatten()+np.dot(Z,bpymc).flatten()
plt.plot(fitted1,'b')

fitted2=np.dot(X,betaend).flatten()+np.dot(Z,bend).flatten()
plt.plot(fitted2,'g')

fitted3=np.dot(X,tbeta.get_value()).flatten()+np.dot(Z,tb.get_value()).flatten()
plt.plot(fitted3,'y')

plt.legend(['observed',
            'Mixed model'+'-'+str(np.mean(np.square(yd-fitted))),
            'PyMC3'+'-'+str(np.mean(np.square(yd-fitted1))),
            'Tensorflow'+'-'+str(np.mean(np.square(yd-fitted2))),
            'Theano'+'-'+str(np.mean(np.square(yd-fitted3)))])
# plt.legend(['observed','Mixed model','PyMC3','Tensorflow'])

fixedCoef = pd.DataFrame(mdf.fe_params,columns=['LMM'])
fixedCoef['PyMC3']  = pd.Series(betapymc, index=fixedCoef.index)
fixedCoef['Tensorflow']  = pd.Series(betaend.flatten(), index=fixedCoef.index)
fixedCoef['Theano']  = pd.Series(tbeta.get_value().flatten(), index=fixedCoef.index)
#%%
plt.figure()
fixed=np.asarray(mdf.fe_params).flatten()
plt.plot(fixed,'k')
plt.plot(betaend,'r')
plt.plot(tbeta.get_value(),'y')
df_summary = pm.df_summary(trace[20000:],varnames=['beta'])
testdf = np.asarray(df_summary['mean'])
plt.plot(testdf,'b')


ydcnt  = np.copy(yd)
sbjidx = np.argmax(Z,axis=1)
b      = np.zeros(np.shape(np.unique(sbjidx)))
for i in np.unique(sbjidx):
    b[i] = np.mean(ydcnt[sbjidx == i])
    ydcnt[sbjidx == i] += -b[i]
beta1  = np.linalg.lstsq(X,ydcnt)
plt.plot(beta1[0],'g')

Xstack    = np.hstack((X,Z))
betastack = np.linalg.lstsq(Xstack,yd)
beta1s    = betastack[0]
plt.legend(['Mixed model','Tensorflow','Theano','PyMC3','HLM'])

plt.figure()
fixed=np.asarray(mdf.random_effects).flatten()
plt.plot(fixed,'k')
plt.plot(bend,'r')
plt.plot(tb.get_value(),'y')
df_summary = pm.df_summary(trace[burnin1:],varnames=['b'])
testdf = np.asarray(df_summary['mean'])
plt.plot(testdf,'b')
plt.plot(b,'g')
plt.legend(['Mixed model','Tensorflow','Theano','PyMC3','HLM'])

#%% GLMM with pymc3 for reaction time data
# model as in brms
Y, X   = dmatrices(formula, data=tbltest, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('rt ~ -1+subj', data=tbltest, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)

# In order to convert the upper triangular correlation values to a complete
# correlation matrix, we need to construct an index matrix:
n_var     = nfixed[1] # 4 within group condition that should be correlated
n_elem    = int(n_var * (n_var - 1) / 2)
tri_index = np.zeros([n_var, n_var], dtype=int)
tri_index[np.triu_indices(n_var, k=1)]       = np.arange(n_elem)
tri_index[np.triu_indices(n_var, k=1)[::-1]] = np.arange(n_elem)
beta0     = np.linalg.lstsq(X,Y)

fixedpred = np.argmax(X,axis=1)
randmpred = np.argmax(Z,axis=1)

con  = tt.constant(fixedpred)
sbj  = tt.constant(randmpred)
import pymc3 as pm

X1     = X[0:,1:]
X_mean = np.mean(X1,axis=0) # column means of X before centering 
X_cent = X1 - X_mean

with pm.Model() as glmm2:
    # Fixed effect
    beta = pm.Normal('beta', mu = 0, sd = 100, shape=(nfixed[1]-1))
    # temporary Intercept 
    temp_Intercept = pm.Normal('temp_Intercept', mu = 0, sd = 100)
    # random effect
    # group-specific standard deviation
    s      = pm.HalfCauchy('s', 5)
    b      = pm.Normal('b', mu = 0, sd = 1, shape=(nrandm[1]))
    r_1    = pm.Deterministic('r_1',s*b)
    sigma  = pm.HalfCauchy('eps', 5)
    
    # compute linear predictor 
    mu_est = tt.dot(X1,beta) + temp_Intercept + r_1[sbj]
    
    RT = pm.Normal('RT', mu_est, sigma, observed = Y)
    b_Intercept = pm.Deterministic("b_Intercept", temp_Intercept - tt.sum(X_mean * beta))
    # start = pm.find_MAP()
    # h = find_hessian(start)
    
with glmm2:
    # means, sds, elbos = pm.variational.advi(n=100000)
    trace2 = pm.sample(100000,step=pm.Metropolis())

    # start  = pm.find_MAP()
    # start['beta'] = beta0[0][1:]
    # step   = pm.NUTS(scaling=start)
    # trace2 = pm.sample(3000, step=pm.NUTS())

pm.traceplot(trace2[0::2]) # 
plt.show()
#%%
burnin2    = 20000
df_summary = pm.df_summary(trace2[burnin2::2],varnames=['beta'])
df_stmp    = pm.df_summary(trace2[burnin2::2],varnames=['b_Intercept'])
df_new     = df_stmp.append(df_summary, ignore_index=True)
df_new.index=Terms
print(df_new)

yd = np.asarray(tbltest.rt)
plt.figure()
plt.plot(yd,'k')

fitted=np.dot(X,mdf.fe_params)+np.dot(Z,mdf.random_effects).flatten()
plt.plot(fitted,'r')

df_summary1 = pm.df_summary(trace[burnin1:],varnames=['beta'])
betapymc = np.asarray(df_summary1['mean'])
df_summary2 = pm.df_summary(trace[burnin1:],varnames=['b'])
bpymc = np.asarray(df_summary2['mean'])
fitted1=np.dot(X,betapymc).flatten()+np.dot(Z,bpymc).flatten()
plt.plot(fitted1,'b')

betapymc2 = np.asarray(df_new['mean'])
dftmp1 = pm.df_summary(trace2[burnin2::2],varnames=['r_1'])
bpymc2 = np.asarray(dftmp1['mean'])
dftmp2 = pm.df_summary(trace2[burnin2::2],varnames=['temp_Intercept'])
tIncpt = np.asarray(dftmp2['mean'])
fitted2=np.dot(X,betapymc2).flatten()+np.dot(Z,bpymc2).flatten()
plt.plot(fitted2,'g')

plt.legend(['observed',
            'Mixed model'+'-'+str(np.mean(np.square(yd-fitted))),
            'Model1'+'-'+str(np.mean(np.square(yd-fitted1))),
            'Model2'+'-'+str(np.mean(np.square(yd-fitted2)))])

plt.figure()
fixed=np.asarray(mdf.fe_params).flatten()
plt.plot(fixed,'r')
plt.plot(betapymc,'b')
plt.plot(betapymc2,'g')
plt.legend(['Mixed model','Model1','Model2'])

plt.figure()
fixed=np.asarray(mdf.random_effects).flatten()
plt.plot(fixed,'r')
plt.plot(bpymc,'b')
plt.plot(bpymc2,'g')
plt.legend(['Mixed model','Model1','Model2'])

fixedCoef2 = pd.DataFrame(mdf.fe_params,columns=['LMM'])
fixedCoef2['Model1']  = pd.Series(betapymc, index=fixedCoef2.index)
fixedCoef2['Model2']  = pd.Series(betapymc2, index=fixedCoef2.index)
print(fixedCoef2)