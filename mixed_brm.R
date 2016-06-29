# from the example ----
library(brms)
bayesfit <- brm(count ~ log_Age_c + log_Base4_c * Trt_c + (1|patient) + (1|visit) + (1|obs), 
           data = epilepsy, family = "poisson")
# get model
stanmodel<- make_stancode(count ~ log_Age_c + log_Base4_c * Trt_c + (1|patient) + (1|visit) + (1|obs), 
                 data = epilepsy, family = "poisson")
summary(bayesfit, waic = TRUE)
library(lme4)
glmefit <- glmer(count ~ log_Age_c + log_Base4_c * Trt_c + (1|patient) + (1|visit) + (1|obs), 
               data = epilepsy, family = "poisson")
summary(glmefit)

# Beh data ----
library(lme4)
library(brms)

Tbl_beh <- read.csv("behavioral_data.txt",sep = "\t")
Tbl_beh$subj <- factor(Tbl_beh$subj)
Tbl_beh$trial <- factor(Tbl_beh$trial)
stanmodel <- make_stancode(rt ~ group*orientation*identity + (1|subj),
                data = Tbl_beh,family = "normal")
standata  <- make_standata(rt ~ group*orientation*identity + (1|subj),
                          data = Tbl_beh,family = "normal")
bayesfit <- brm(rt ~ group*orientation*identity + (1|subj),
                data = Tbl_beh,family = "normal")
bayesfit$model
summary(bayesfit, waic = TRUE)

lmefit <- lmer(rt ~ group*orientation*identity + (1|subj),
                 data = Tbl_beh)
summary(lmefit)