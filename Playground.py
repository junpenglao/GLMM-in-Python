# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:55:33 2016

@author: laoj

Experiment with different package
"""
#%% simulation data
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
M1   = 6  # number of columns in X - fixed effect
N1   = 10 # number of columns in L - random effect
nobs = 20

# generate design matrix using patsy
import statsmodels.formula.api as smf
from patsy import dmatrices
import pandas as pd
predictors = []
for s1 in range(N1):
    for c1 in range(2):
        for c2 in range(3):
            for i in range(nobs):
                predictors.append(np.asarray([c1+1,c2+1,s1+1]))
tbltest             = pd.DataFrame(predictors, columns=['Condi1', 'Condi2', 'subj'])
tbltest['Condi1']   = tbltest['Condi1'].astype('category')
tbltest['Condi2']   = tbltest['Condi2'].astype('category')
tbltest['subj']     = tbltest['subj'].astype('category')
tbltest['tempresp'] = np.random.normal(size=(nobs*M1*N1,1))*10

Y, X    = dmatrices("tempresp ~ Condi1*Condi2", data=tbltest, return_type='matrix')
Terms   = X.design_info.column_names
_, Z    = dmatrices('tempresp ~ -1+subj', data=tbltest, return_type='matrix')
X       = np.asarray(X) # fixed effect
Z       = np.asarray(Z) # mixed effect
Y       = np.asarray(Y) 
N,nfixed = np.shape(X)
_,nrandm = np.shape(Z)
# generate data
w0 = [5.0, 1.0, 2.0, 8.0, 1.0, 1.0] + np.random.randn(6)
#w0 -= np.mean(w0)
#w0 = np.random.normal(size=(M,))
z0 = np.random.normal(size=(N1,))*10

Pheno     = np.dot(X,w0) + np.dot(Z,z0) + Y.flatten()
beta0     = np.linalg.lstsq(X,Pheno)

fixedpred = np.argmax(X,axis=1)
randmpred = np.argmax(Z,axis=1)

tbltest['Pheno'] = Pheno
md  = smf.mixedlm("Pheno ~ Condi1*Condi2", tbltest, groups=tbltest["subj"])
mdf = md.fit()
Y   = np.expand_dims(Pheno,axis=1)

fitted=mdf.fittedvalues

fe_params = pd.DataFrame(mdf.fe_params,columns=['LMM'])
fe_params.index=Terms
random_effects = pd.DataFrame(mdf.random_effects)
random_effects = random_effects.transpose()
random_effects = random_effects.rename(index=str, columns={'groups': 'LMM'})
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
Y      = np.asarray(Y)
N,nfixed = np.shape(X)
_,nrandm = np.shape(Z)
fe_params = pd.DataFrame(mdf.fe_params,columns=['LMM'])
random_effects = pd.DataFrame(mdf.random_effects)
random_effects = random_effects.transpose()
random_effects = random_effects.rename(index=str, columns={'groups': 'LMM'})

fitted=mdf.fittedvalues
#%% ploting function 
def plotfitted(fe_params=fe_params,random_effects=random_effects,X=X,Z=Z,Y=Y):
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

plotfitted(fe_params=fe_params,random_effects=random_effects,X=X,Z=Z,Y=Y)
#%% Bambi (tested on version 0.1.0)
from bambi import Model
from scipy.stats import norm

# Assume we already have our data loaded
model = Model(tbltest)
model.add(formula)
model.add(random=['1|subj'], 
          categorical=['group','orientation','identity','subj'])
model.build(backend='pymc')

# Plot prior
#p = len(model.terms)
#fig, axes = plt.subplots(int(np.ceil(p/2)), 2, figsize=(12,np.ceil(p/2)*2))
#
#for i, t in enumerate(model.terms.values()):
#    m = t.prior.args['mu']
#    sd = t.prior.args['sd']
#    x = np.linspace(m - 3*sd, m + 3*sd, 100)
#    y = norm.pdf(x, loc=m, scale=sd)
#    axes[divmod(i,2)[0], divmod(i,2)[1]].plot(x,y)
#    axes[divmod(i,2)[0], divmod(i,2)[1]].set_title(t.name)
#plt.subplots_adjust(wspace=.25, hspace=.5)

model.plot_priors(varnames=['Intercept','group','orientation',
                           'identity','group:orientation','group:identity',
                           'orientation:identity','group:orientation:identity'])
plt.show()

results = model.fit(formula, random=['1|subj'], 
                    categorical=['group','orientation','identity','subj'],
                    samples=2000, chains=2)
_ = results.plot(varnames=['Intercept','group','orientation',
                           'identity','group:orientation','group:identity',
                           'orientation:identity','group:orientation:identity'])
_ = results.plot(varnames=['1|subj'])

burn_in=1000

summary = results[burn_in:].summary(ranefs=True)
print(summary)
# tracedf = results[burn_in:].to_df(ranefs=True)

fe_params['Bambi'] = summary[summary.index.isin(fe_params.index)]['mean']
random_effects['Bambi'] = summary.loc[['1|subj['+ind_re+']'
                                       for ind_re in random_effects.index]]['mean'].values

#%% Tensorflow
import tensorflow as tf

tf.reset_default_graph()
def tfmixedmodel(X, beta, Z, b):
    with tf.name_scope("fixedEffect"):
        fe = tf.matmul(X, beta)
    with tf.name_scope("randomEffect"):
        re = tf.matmul(Z, b)
    #randcoef = tf.matmul(Z, b)
    #Xnew     = tf.transpose(X) * tf.transpose(randcoef)
    #y_pred   = tf.matmul(tf.transpose(Xnew), beta)
    return tf.add(fe,re) # notice we use the same model as linear regression, 
                  # this is because there is a baked in cost function which performs softmax and cross entropy

Xtf      = tf.placeholder("float32", [None, nfixed]) # create symbolic variables
Ztf      = tf.placeholder("float32", [None, nrandm])
y        = tf.placeholder("float32", [None, 1])
beta_tf  = tf.Variable(tf.random_normal([nfixed, 1], stddev=1, name="fixed_beta"))
b_tf     = tf.Variable(tf.random_normal([nrandm, 1], stddev=1, name="random_b"))
b_tf     = b_tf - tf.reduce_mean(b_tf)
eps_tf   = tf.Variable(tf.random_normal([0], stddev=1, name="eps"))
y_       = tfmixedmodel(Xtf, beta_tf, Ztf, b_tf)
#y_    = tf.nn.softmax(tf.matmul(Xtf, beta) + tf.matmul(Ztf, b) + eps)

# Add histogram summaries for weights
tf.summary.histogram("fixed", beta_tf)
tf.summary.histogram("random", b_tf)

nb_epoch   = 5000
batch_size = 100
with tf.name_scope("cost"):
    #cost = tf.reduce_sum(tf.pow(y - y_, 2))
    #train_step    = tf.train.RMSPropOptimizer(0.01, epsilon=1.0).minimize(cost)
    cost        = tf.reduce_mean(tf.square(y - y_)) # use square error for cost function
    train_step  = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    # Add scalar summary for cost
    tf.summary.scalar("cost", cost)

with tf.name_scope("SSE"):
    sse = tf.reduce_mean(tf.cast(cost, tf.float32))
    # Add scalar summary for SSE
    tf.summary.scalar("SSE", sse)

with tf.Session() as sess:
    # create a log writer. run 'tensorboard --logdir=/tmp/GLMMtest'
    writer = tf.summary.FileWriter("/tmp/GLMMtest", sess.graph) # for 0.8
    merged = tf.summary.merge_all()

    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for i in range(nb_epoch):
        shuffleidx = np.random.permutation(N)
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
            batch_xs, batch_zs, batch_ys = X[shuffleidx[start:end]],Z[shuffleidx[start:end]],Y[shuffleidx[start:end]]
            sess.run(train_step, feed_dict={Xtf: batch_xs, Ztf: batch_zs, y: batch_ys})
        summary, acc = sess.run([merged, sse], 
                                feed_dict={Xtf: X, Ztf: Z, y: Y})
        writer.add_summary(summary, i)  # Write summary
        if (i % 100 == 0):
            print(i, acc)
    
    betatf = sess.run(beta_tf)
    btf    = sess.run(b_tf)    
    
fe_params['TF'] = pd.Series(betatf.flatten(), index=fe_params.index)
random_effects['TF'] = pd.Series(btf.flatten(), index=random_effects.index)
#%% variational inference (using tensorflow)
"""
Using mean field variational inference on ELBO as explained in
https://github.com/blei-lab/edward/blob/master/edward/inferences/klqp.py
WARNING: SLOW
"""
tf.reset_default_graph()
Xtf      = tf.placeholder("float32", [None, nfixed]) # create symbolic variables
Ztf      = tf.placeholder("float32", [None, nrandm])
y        = tf.placeholder("float32", [None, 1])

priorstd = 1
from tensorflow.contrib.distributions import Normal

#fixed effect
eps_fe   = tf.random_normal([nfixed, 1], name='eps_fe')
beta_mu  = tf.Variable(tf.random_normal([nfixed, 1], stddev=priorstd), name="fixed_mu")

##diag cov
beta_logvar  = tf.Variable(tf.random_normal([nfixed, 1], stddev=priorstd), name="fixed_logvar")
std_encoder1 = tf.exp(0.5 * beta_logvar)
beta_tf      = Normal(loc=beta_mu, scale=std_encoder1)

#random effect
eps_rd   = tf.random_normal([nrandm, 1], name='eps_rd')
b_mu     = tf.Variable(tf.random_normal([nrandm, 1], stddev=priorstd), name="randm_mu")
b_mu     = b_mu - tf.reduce_mean(b_mu)

b_logvar     = tf.Variable(tf.random_normal([nrandm, 1], stddev=priorstd), name="randm_logvar")
std_encoder2 = tf.exp(0.5 * b_logvar)
b_tf         = Normal(loc=b_mu, scale=std_encoder2)

# MixedModel
y_mu     = tfmixedmodel(Xtf, beta_mu, Ztf, b_mu)

# Add histogram summaries for weights
tf.summary.histogram("fixed",  beta_mu)
tf.summary.histogram("random", b_mu)

nb_epoch   = 1000
batch_size = 100

priormu, priorsigma, priorliksigma= 0.0, 100.0, 10.0
n_samples = 5 #5-10 might be enough
with tf.name_scope("cost"):
    #mean_squared_error
    RSEcost  = tf.reduce_mean(tf.square(y - y_mu)) # use square error for cost function
    
#    #negative log-likelihood (same as maximum-likelihood)
#    y_sigma  = tf.sqrt(tfmixedmodel(Xtf, tf.square(std_encoder1), Ztf, tf.square(std_encoder2)))
#    NLLcost  = - tf.reduce_sum(-0.5 * tf.log(2. * np.pi) - tf.log(y_sigma)
#                               -0.5 * tf.square((y - y_mu)/y_sigma))

    #Mean-field Variational inference using ELBO
    p_log_prob = [0.0] * n_samples
    q_log_prob = [0.0] * n_samples
    for s in range(n_samples):
        beta_tf_copy = Normal(loc=beta_mu, scale=std_encoder1)
        beta_sample  = beta_tf_copy.sample()
        q_log_prob[s] += tf.reduce_sum(beta_tf.log_prob(beta_sample))
        b_tf_copy    = Normal(loc=b_mu, scale=std_encoder2)
        b_sample     = b_tf_copy.sample()
        q_log_prob[s] += tf.reduce_sum(b_tf.log_prob(b_sample))
        
        priormodel    = Normal(loc=priormu, scale=priorsigma)
        y_sample      = tf.matmul(Xtf, beta_sample) + tf.matmul(Ztf, b_sample)
        p_log_prob[s] += tf.reduce_sum(priormodel.log_prob(beta_sample))
        p_log_prob[s] += tf.reduce_sum(priormodel.log_prob(b_sample))
        modelcopy     = Normal(loc=y_sample, scale=priorliksigma)
        p_log_prob[s] += tf.reduce_sum(modelcopy.log_prob(y))
        
    p_log_prob = tf.stack(p_log_prob)
    q_log_prob = tf.stack(q_log_prob)
    ELBO = -tf.reduce_mean(p_log_prob - q_log_prob)
    
    #train_step    = tf.train.AdamOptimizer(0.01).minimize(NLLcost)
    #train_step    = tf.train.AdagradOptimizer(0.1).minimize(ELBO)
    #train_step    = tf.train.RMSPropOptimizer(0.01, epsilon=0.1).minimize(NLLcost)
    train_step    = tf.train.GradientDescentOptimizer(0.01).minimize(ELBO)
    
    # Add scalar summary for cost
    tf.summary.scalar("cost", ELBO)

with tf.name_scope("MSE"):
    sse = tf.reduce_mean(tf.cast(RSEcost, tf.float32))
    # Add scalar summary for SSE
    tf.summary.scalar("MSE", sse)

with tf.Session() as sess:
    # create a log writer. run 'tensorboard --logdir=/tmp/GLMMtest'
    writer = tf.summary.FileWriter("/tmp/GLMMtest", sess.graph) # for 0.8
    merged = tf.summary.merge_all()

    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for i in range(nb_epoch):
        shuffleidx = np.random.permutation(N)
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
            batch_xs, batch_zs, batch_ys = X[shuffleidx[start:end]],Z[shuffleidx[start:end]],Y[shuffleidx[start:end]]
            sess.run(train_step, feed_dict={Xtf: batch_xs, Ztf: batch_zs, y: batch_ys})
        summary, acc = sess.run([merged, sse], 
                                feed_dict={Xtf: X, Ztf: Z, y: Y})
        writer.add_summary(summary, i)  # Write summary
        if (i % 1000 == 0):
            print(i, acc)
    
    betatf = sess.run(beta_mu)
    btf    = sess.run(b_mu)
    betatf_std = sess.run(std_encoder1)
    btf_std    = sess.run(std_encoder2)
    
fe_params['TF_VA'] = pd.Series(betatf.flatten(), index=fe_params.index)
random_effects['TF_VA'] = pd.Series(btf.flatten(), index=random_effects.index)
sess.close()
#%% variational inference (using Edward)
"""
https://github.com/blei-lab/edward/blob/master/notebooks/linear_mixed_effects_models.ipynb
"""
import edward as ed
import tensorflow as tf
from edward.models import Normal
        
#DATA
X1     = X[:,1:]
#X_mean = np.mean(X1,axis=0) # column means of X before centering 
#X_cent = X1 - X_mean
#x_train, y_train = X_cent , Y
x_train,z_train,y_train = X1.astype('float32'), Z.astype('float32'), Y.flatten()
N, D = x_train.shape  # num features
Db = z_train.shape[1]

# Set up placeholders for the data inputs.
Xnew = tf.placeholder(tf.float32, shape=(None, D))
Znew = tf.placeholder(tf.float32, shape=(None, Db))

# MODEL
Wf = Normal(loc=tf.zeros([D]), scale=tf.ones([D]))
Wb = Normal(loc=tf.zeros([Db]), scale=tf.ones([Db]))
Ib = Normal(loc=tf.zeros([1]), scale=tf.ones(1))

# INFERENCE
qi_mu = tf.Variable(tf.random_normal([1]))
qi_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([1])))
qi = Normal(loc=qi_mu, scale=qi_sigma)

#qw_mu = tf.expand_dims(tf.convert_to_tensor(beta0[0].astype(np.float32)),1)
qw_mu = tf.Variable(tf.random_normal([D]))
qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([D])))
qw = Normal(loc=qw_mu, scale=qw_sigma)

#qb_mu = tf.Variable(tf.random_normal([Db,1]))
qb_mu = tf.Variable(tf.random_normal([Db])) #force the random coeff to be zero-distributed
#qb_mu = qb_mu - tf.reduce_mean(qb_mu)
qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([Db])))
qb = Normal(loc=qb_mu, scale=qb_sigma)
yhat = ed.dot(Xnew, Wf)+ed.dot(Znew, Wb)+Ib
y  = Normal(loc=yhat, scale=tf.ones(N))

sess = ed.get_session()
inference = ed.KLqp({Wf: qw, Wb: qb, Ib: qi},
                    data={y: y_train, Xnew: x_train, Znew:z_train})

#optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
optimizer = tf.train.RMSPropOptimizer(1., epsilon=1.0)
#optimizer = tf.train.GradientDescentOptimizer(0.01)
inference.run(optimizer=optimizer, n_samples=20, n_iter=10000)

#inference.run(n_samples=20, n_iter=20000)

i_mean, i_std, w_mean, w_std, b_mean, b_std = sess.run([qi.loc, qi.scale, qw.loc,
                                                        qw.scale,qb.loc, qb.scale])

#fixed_ed  = np.hstack([i_mean+b_mean.mean(),w_mean.flatten()])
#randm_ed  = b_mean-b_mean.mean()
fixed_ed  = np.hstack([i_mean,w_mean.flatten()])
randm_ed  = b_mean
fixed_ed_std  = np.hstack([i_std, w_std.flatten()])
randm_ed_std  = b_std

fitted_ed = np.dot(X,fixed_ed)+np.dot(Z,randm_ed).flatten()

fe_params['edward'] = pd.Series(fixed_ed, index=fe_params.index)
random_effects['edward'] = pd.Series(randm_ed.flatten(), index=random_effects.index)
#%% variational inference (using Edward, with sigma also modelled)
import edward as ed
import tensorflow as tf
from edward.models import Normal, Empirical, InverseGamma

#DATA
X1     = X[:,1:]
#X_mean = np.mean(X1,axis=0) # column means of X before centering 
#X_cent = X1 - X_mean
#x_train, y_train = X_cent , Y
x_train,z_train,y_train = X1.astype('float32'), Z.astype('float32'), Y.flatten()
D  = x_train.shape[1]  # num features
Db = z_train.shape[1]

# MODEL
Wf = Normal(loc=tf.zeros([D]), scale=tf.ones([D]))
Wb = Normal(loc=tf.zeros([Db]), scale=tf.ones([Db]))
Ib = Normal(loc=tf.zeros(1), scale=tf.ones(1))

Xnew = tf.placeholder(tf.float32, shape=(None, D))
Znew = tf.placeholder(tf.float32, shape=(None, Db))
ynew = tf.placeholder(tf.float32, shape=(None, ))

sigma2 = InverseGamma(concentration=tf.ones(1)*.1, rate=tf.ones(1)*.1)
#sigma2 = Normal(loc=tf.zeros([1]), scale=tf.ones([1])*100)
y  = Normal(loc=ed.dot(x_train, Wf)+ed.dot(z_train, Wb)+Ib, scale=tf.log(sigma2))

# INFERENCE
sess = ed.get_session()
T = 10000
qi = Empirical(params=tf.Variable(tf.zeros([T, 1])))
qw = Empirical(params=tf.Variable(tf.zeros([T, D])))
qb = Empirical(params=tf.Variable(tf.zeros([T, Db])))
qsigma2 = Empirical(params=tf.Variable(tf.ones([T,1])))

inference = ed.SGHMC({Wf: qw, Wb: qb, Ib: qi, sigma2: qsigma2}, data={y: y_train})
inference.run(step_size=.0005)

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(qi.get_variables()[0].eval())
ax2.plot(qw.get_variables()[0].eval())
ax3.plot(qb.get_variables()[0].eval())
ax4.plot(qsigma2.get_variables()[0].eval())

burnin = int(T/2)
qi_post = qi.get_variables()[0].eval()[burnin:].mean(axis=0)
qw_post = qw.get_variables()[0].eval()[burnin:].mean(axis=0)
qb_post =qb.get_variables()[0].eval()[burnin:].mean(axis=0)

#fixed_ed  = np.hstack([i_mean+b_mean.mean(),w_mean.flatten()])
#randm_ed  = b_mean-b_mean.mean()
fixed_ed  = np.hstack([qi_post,qw_post])
randm_ed  = qb_post
fixed_ed_std  = np.hstack([i_std, w_std.flatten()])
randm_ed_std  = b_std

fitted_ed = np.dot(X,fixed_ed)+np.dot(Z,randm_ed).flatten()

fe_params['edward2'] = pd.Series(fixed_ed, index=fe_params.index)
random_effects['edward2'] = pd.Series(randm_ed.flatten(), index=random_effects.index)
#%% PyTorch
import torch
from torch.autograd import Variable
import torch.optim as optim
dtype = torch.FloatTensor

x_train,z_train,y_train = X.astype('float32'), Z.astype('float32'), Y.astype('float32')
# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
Xt = Variable(torch.from_numpy(x_train), requires_grad=False)
Zt = Variable(torch.from_numpy(z_train), requires_grad=False)
y  = Variable(torch.from_numpy(y_train),  requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(nfixed,1).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(nrandm,1).type(dtype), requires_grad=True)

learning_rate = 1e-2
params = [w1,w2]

solver = optim.SGD(params, lr=learning_rate)
for t in range(10000):
    # Forward pass: compute predicted y using operations on Variables; we compute
    # ReLU using our custom autograd operation.
    y_pred = Xt.mm(w1) + Zt.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).mean()
    if (t % 1000 == 0):
        print(t, loss.data[0])

#    # Manually zero the gradients before running the backward pass
#    w1.grad.data.zero_()
#    w2.grad.data.zero_()
#
#    # Use autograd to compute the backward pass.
#    loss.backward()
#
#    # Update weights using gradient descent
#    w1.data -= learning_rate * w1.grad.data
#    w2.data -= learning_rate * w2.grad.data
    
    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    solver.zero_grad()
#    for p in params:
#        p.grad.data.zero_()
        
fe_params['PyTorch'] = pd.Series(w1.data.numpy().flatten(), index=fe_params.index)
random_effects['PyTorch'] = pd.Series(w2.data.numpy().flatten(), index=random_effects.index)

#%% ploting
plotfitted(fe_params=fe_params,random_effects=random_effects,X=X,Z=Z,Y=Y)
