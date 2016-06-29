// This Stan code was generated with the R package 'brms'. 
// We recommend generating the data with the 'make_standata' function. 
functions { 
} 
data { 
  int<lower=1> N;  // total number of observations 
  vector[N] Y;  // response variable 
  int<lower=1> K;  // number of population-level effects 
  matrix[N, K] X;  // centered population-level design matrix 
  vector[K] X_means;  // column means of X before centering 
  // data for group-specific effects of subj 
  int<lower=1> J_1[N]; 
  int<lower=1> N_1; 
  int<lower=1> K_1; 
  vector[N] Z_1; 
  int prior_only;  // should the likelihood be ignored? 
} 
transformed data { 
} 
parameters { 
  vector[K] b;  // population-level effects 
  real temp_Intercept;  // temporary Intercept 
  real<lower=0> sd_1;  // group-specific standard deviation 
  vector[N_1] z_1;  // unscaled group-specific effects 
  real<lower=0> sigma;  // residual SD 
} 
transformed parameters { 
  // group-specific effects 
  vector[N_1] r_1; 
  vector[N] eta;  // linear predictor 
  r_1 <- sd_1 * (z_1);
  // compute linear predictor 
  eta <- X * b + temp_Intercept; 
  for (n in 1:N) { 
    eta[n] <- eta[n] + r_1[J_1[n]] * Z_1[n]; 
  } 
} 
model { 
  // prior specifications 
  sd_1 ~ student_t(3, 0, 186); 
  z_1 ~ normal(0, 1); 
  sigma ~ student_t(3, 0, 186); 
  // likelihood contribution 
  if (!prior_only) { 
    Y ~ normal(eta, sigma); 
  } 
} 
generated quantities { 
  real b_Intercept;  // population-level intercept 
  b_Intercept <- temp_Intercept - dot_product(X_means, b); 
} 