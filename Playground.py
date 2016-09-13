# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:55:33 2016

@author: laoj

Experiment with different package
"""
#%% simulation data
import numpy as np
M    = 6  # number of columns in X - fixed effect
N    = 10 # number of columns in L - random effect
nobs = 10

# generate design matrix using patsy
import statsmodels.formula.api as smf
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
tbltest['tempresp'] = np.random.normal(size=(nobs*M*N,1))*10

Y, X    = dmatrices("tempresp ~ Condi1*Condi2", data=tbltest, return_type='matrix')
Terms   = X.design_info.column_names
_, Z    = dmatrices('tempresp ~ -1+subj', data=tbltest, return_type='matrix')
X       = np.asarray(X) # fixed effect
Z       = np.asarray(Z) # mixed effect
Y       = np.asarray(Y) 
nfixed  = np.shape(X)
nrandm  = np.shape(Z)
# generate data
w0 = [5.0, 1.0, 2.0, 13.0, 1.0, 1.0]
#w0 -= np.mean(w0)
#w0 = np.random.normal(size=(M,))
z0 = np.random.normal(size=(N,))*10

Pheno     = np.dot(X,w0) + np.dot(Z,z0) + Y.flatten()
beta0     = np.linalg.lstsq(X,Pheno)

fixedpred = np.argmax(X,axis=1)
randmpred = np.argmax(Z,axis=1)

tbltest['Pheno'] = Pheno
md  = smf.mixedlm("Pheno ~ Condi1*Condi2", tbltest, groups=tbltest["subj"])
mdf = md.fit()
Y   = Pheno

fitted=np.dot(X,mdf.fe_params)+np.dot(Z,mdf.random_effects).flatten()

fe_params = pd.DataFrame(mdf.fe_params,columns=['LMM'])
fe_params['real'] = pd.Series(w0, index=fe_params.index)
random_effects = pd.DataFrame(mdf.random_effects)
random_effects['real'] = pd.Series(z0, index=random_effects.index)
#%% Real data
Tbl_beh  = pd.read_csv('./behavioral_data.txt', delimiter='\t')
Tbl_beh["subj"]  = Tbl_beh["subj"].astype('category')
tbltest = Tbl_beh
formula = "rt ~ group*orientation*identity"
#formula = "rt ~ -1 + cbcond"
md  = smf.mixedlm(formula, tbltest, groups=tbltest["subj"])
mdf = md.fit()
Y, X   = dmatrices(formula, data=tbltest, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('rt ~ -1+subj', data=tbltest, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)
fe_params = pd.DataFrame(mdf.fe_params,columns=['LMM'])
random_effects = pd.DataFrame(mdf.random_effects)

fitted=np.dot(X,mdf.fe_params)+np.dot(Z,mdf.random_effects).flatten()
#%% edward
import edward as ed
import tensorflow as tf
from edward.models import Normal, InverseGamma
from edward.stats import norm, studentt
#ed.set_seed(42)
#%%
class MixedModel:
    def __init__(self, lik_std=0.1, prior_std=1.0):
        self.lik_std = lik_std
        self.prior_std = prior_std
        
    def mixedformula(self, X, Z, zs):
        beta, b, Intercept = zs['beta'], zs['b'], zs['Intercept']
        fixedpredi = tf.matmul(X, beta)
        randmpredi = tf.matmul(Z, b)
        h = fixedpredi + randmpredi + Intercept
        return tf.squeeze(h)  # n_minibatch x 1 to n_minibatch
    
    def log_prob(self, xs, zs):
        """Return scalar, the log joint density log p(xs, zs)."""
        X, Z, y = xs['X'], xs['Z'], xs['y']
        beta, b, Intercept = zs['beta'], zs['b'], zs['Intercept']
        log_prior = 0.0
        log_prior += tf.reduce_sum(norm.logpdf(beta, 0.0, self.prior_std))
        log_prior += tf.reduce_sum(norm.logpdf(b, 0.0, self.prior_std))
        log_prior += tf.reduce_sum(norm.logpdf(Intercept, 0.0, self.prior_std))
        mu = self.mixedformula(X, Z, zs)
        log_lik = tf.reduce_sum(norm.logpdf(y, mu, self.lik_std))
        return log_lik + log_prior

X1     = X[:,1:]
#X_mean = np.mean(X1,axis=0) # column means of X before centering 
#X_cent = X1 - X_mean
#x_train, y_train = X_cent , Y
x_train, y_train = X1 , Y
D  = x_train.shape[1]  # num features
Db = Z.shape[1]

qi_mu = tf.Variable(tf.random_normal([]))
qi_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qi = Normal(mu=qi_mu, sigma=qi_sigma)

#qw_mu = tf.expand_dims(tf.convert_to_tensor(beta0[0].astype(np.float32)),1)
qw_mu = tf.Variable(tf.random_normal([D,1]))
qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([D,1])))
qw = Normal(mu=qw_mu, sigma=qw_sigma)

qb_mu = tf.Variable(tf.random_normal([Db,1]))
qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([Db,1])))
qb = Normal(mu=qb_mu, sigma=qb_sigma)

zs   = {'beta': qw, 'b': qb, 'Intercept': qi}
#%%
model = MixedModel(lik_std=.1,prior_std=100.0)

sess = ed.get_session()
data = {'X': x_train, 'y': y_train, 'Z': Z}
inference = ed.MFVI(zs, data, model)
inference.run(n_iter=50000, n_print=10000)

"""
Observation: if the ratio between the fixed effect and the random effect is large,
The model needs more iteration to converge.
Also, if either the fixed effect or the random effect is not normal distributed,
the fitting is not going to perform well
"""
#%%
Xnew = ed.placeholder(tf.float32, shape=(None, D))
Znew = ed.placeholder(tf.float32, shape=(None, Db))
ynew = ed.placeholder(tf.float32, shape=(None))

data = {'X': Xnew, 'y': ynew, 'Z': Znew}

model = MixedModel(lik_std=.1,prior_std=100.0)

sess = ed.get_session()
inference = ed.MFVI(zs, data, model)
#inference.initialize()

#optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
optimizer = tf.train.RMSPropOptimizer(0.1, epsilon=1.0)
#optimizer = tf.train.GradientDescentOptimizer(0.0001)
inference.initialize(optimizer=optimizer)

NEPOCH = 10000
train_loss = np.zeros(NEPOCH)
test_loss = np.zeros(NEPOCH)
batch_xs, batch_zs, batch_ys = x_train,Z,y_train
for i in range(NEPOCH):
    #sel = np.random.randint(0,nfixed[0],size=int(nfixed[0]/2))
    #batch_xs, batch_zs, batch_ys = x_train[sel,:],Z[sel,:],y_train[sel]
    _, train_loss[i] = sess.run([inference.train, inference.loss],
                              feed_dict={Xnew: batch_xs, ynew: batch_ys, Znew: batch_zs})
    test_loss[i] = sess.run(inference.loss, feed_dict={Xnew: batch_xs, ynew: batch_ys, Znew: batch_zs})
#%%
i_mean, i_std, w_mean, w_std, b_mean, b_std = sess.run([qi.mu, qi.sigma, qw.mu,
                                                        qw.sigma,qb.mu, qb.sigma])

fixed_ed  = np.hstack([i_mean+b_mean.mean(),w_mean.flatten()])
randm_ed  = b_mean-b_mean.mean()
fitted_ed = np.dot(X,fixed_ed)+np.dot(Z,randm_ed).flatten()

fe_params['edward'] = pd.Series(fixed_ed, index=fe_params.index)
random_effects['edward'] = pd.Series(randm_ed.flatten(), index=random_effects.index)

print(fe_params)
print(random_effects)

print("The MSE of LMM is " + str(np.mean(np.square(Y-fitted))))
print("The MSE of Edward is " + str(np.mean(np.square(Y-fitted_ed))))

import matplotlib.pylab as plt
plt.figure()
plt.plot(Y)
plt.plot(fitted)
plt.plot(fitted_ed)
plt.show()