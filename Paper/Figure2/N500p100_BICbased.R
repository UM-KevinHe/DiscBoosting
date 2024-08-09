rm(list = ls())


# load("Boosting_NR_result.RData")
library(mvtnorm)
library(matrixStats)
library(survival)
library(Rcpp)
library(ggplot2)
library(discSurv)


# setwd("~/Dropbox (University of Michigan)/Lingfeng Research/VaraibleSelection/Discretized/Timevarying/Paper/Boosting_NR_diffIter/figure1_auc_boxplot_TICStop////")
# sourceCpp("~/Dropbox (University of Michigan)/Lingfeng Research/VaraibleSelection/Discretized/Timevarying/Newton's Method/testcode/Discrete_logit_NR_timevarying_interact.cpp")
# sourceCpp("~/Dropbox (University of Michigan)/Lingfeng Research/VaraibleSelection/Discretized/Timevarying/Newton's Method/testcode/Discrete_logit_NR_timevarying_spline.cpp")
# sourceCpp("~/Dropbox (University of Michigan)/Lingfeng Research/VaraibleSelection/Discretized/Timevarying/Discrete_logit_Boosting_arma_Both_constant_timevarying.cpp")


sourceCpp("/home/lfluo/VariableSelection/TimeVarying/source/Discrete_logit_Boosting_arma_Both_constant_timevarying.cpp")

sourceCpp("/home/lfluo/VariableSelection/TimeVarying/source/Discrete_logit_NR_timevarying_interact.cpp")
sourceCpp("/home/lfluo/VariableSelection/TimeVarying/source/Discrete_logit_NR_timevarying_spline.cpp")

seed_tmp <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
# seed_tmp <- 96
seed_index <- seed_tmp

########################################################################################################################################################################################################################

FPFNSeSpLik=function(TrueBeta=TrueBeta,beta=beta){
  FP <- length(which(TrueBeta==0 & beta!=0))
  FN <- length(which(TrueBeta!=0 & beta==0))
  Se <- length(which(TrueBeta!=0 & beta!=0))/length(which(TrueBeta!=0))
  Sp <- length(which(TrueBeta==0 & beta==0))/length(which(TrueBeta==0))
  FDP=FP/max(length(which(beta!=0)),1)
  output=c(FP, FN, Se, Sp, FDP)
  return(output)
}

AR1 <- function(tau, m) {
  if(m==1) {R <- 1}
  if(m > 1) {
    R <- diag(1, m)
    for(i in 1:(m-1)) {
      for(j in (i+1):m) {
        R[i,j] <- R[j,i] <- tau^(abs(i-j))
      }
    }
  }
  return(R)
  
}

simu_z <- function(n, size.groups){
  Sigma_z1=diag(size.groups) # =p
  Corr1<-AR1(0.3,size.groups) #correlation structure 0.5 0.6
  diag(Corr1) <- 1
  Sigma_z1<- Corr1
  pre_z= rmvnorm(n, mean=rep(0,size.groups), sigma=Sigma_z1)
  z_rare=function(x){
    U=runif(1, 0.4, 0.7)
    
    x2=quantile(x,prob=U)
    x3=x
    x3[x<x2]=0
    x3[x>x2]=1
    return(x3)
  }
  z=apply(pre_z, 2, z_rare)
  return(z)
}

p = 100
step_size = 0.5
PenalizeGroup = FALSE
Mstop <- 1000

beta_effect <- function(x, P){
  return (c(1, cos(pi*x/50), -1, sin(3*pi*x/80), (-1+exp(-0.25*x)), rep(0,P-5)))
}
p_true = 5
p_tv_true = 3
########################################################################################################################################################################################################################

pb = txtProgressBar(style = 3)

knot = 7
FPFN <- NULL
FPFN_constant <- NULL
FPFN_tv <- NULL
chosen_constant_index_all <- NULL
chosen_tv_index_all <- NULL
time_used_all <- NULL
BIC_all <- NULL

seq = seq(0.1/(Mstop/100),1,0.1/(Mstop/100))*Mstop
FPFN_tv_mstop_all <- vector(mode = "list", length = length(seq))
FPFN_mstop_all <- vector(mode = "list", length = length(seq))

penalty_all = seq(0,4,0.2)
####################################################################################################################################

AIC_diffstep <- NULL
TIC_diffstep <- NULL
GIC_diffstep <- NULL

BIC_AIC_diffstep <- NULL
BIC_TIC_diffstep <- NULL
BIC_GIC_diffstep <- NULL
BIC_HTIC_diffstep <- NULL

lambda_TIC <- NULL

################
set.seed(seed_index*2022)
if (seed_index == 88 || seed_index == 87) {
  set.seed(seed_index*2022+1)
}

N = 500
Z.char <- paste0('Z', 1:p)
eta <- -1+c(-3,-2.5,-2.2,-2.12,-2.08,-2,-1.95,-1.87,-1.8,-1.69,-1.61,-1.49,-1.39,-1.25,-1.1,-0.9,-0.7,-0.35,-0)
# eta <- rep(-0.1,5)
censor = length(eta)
z <- as.matrix(simu_z(N, p))
# z <- scale(z,center=TRUE,scale=TRUE)
day <- 1 #1
idx.atrisk <- 1:N
days.to.event <- rep(length(eta), N)
status <- rep(0, N)

beta <- beta_effect(day, p)
probs <- plogis(eta[1] + beta[1]*z[,1] + beta[2]*z[,2] + beta[3]*z[,3] + beta[4]*z[,4] + beta[5]*z[,5])
summary(probs)
idx.event <- idx.atrisk[rbinom(length(probs), 1, probs)==1]
status[idx.event] <- 1
days.to.event[idx.event] <- day
idx.out <- idx.event
censoring=runif(N,0,censor)
conTime = data.frame(time=censoring)
censoring_time <- as.numeric(contToDisc(dataShort = conTime, timeColumn = "time", intervalLimits = 1:(censor))$timeDisc)#3
for (x in tail(eta,-1)) {
  day <- day+1
  beta <- beta_effect(day,p)
  idx.atrisk <- c(1:N)[-idx.out]  
  # probs <- plogis(x+(as.matrix(Z[idx.atrisk,])%*%beta))
  probs <- plogis( x + (beta[1]*z[idx.atrisk,1] + beta[2]*z[idx.atrisk,2] + beta[3]*z[idx.atrisk,3] + beta[4]*z[idx.atrisk,4] + beta[5]*z[idx.atrisk,5]))
  idx.event <- idx.atrisk[rbinom(length(probs), 1, probs)==1]
  status[idx.event] <- 1
  days.to.event[idx.event] <- day
  idx.out <- unique(c(idx.out, idx.event))
}

tcens <- as.numeric(censoring<days.to.event) # censoring indicator
delta <- 1-tcens
time <- days.to.event*(delta==1)+censoring_time*(delta==0)
delta[-idx.out] <- 0
data <- as.data.frame(cbind(delta, z, time))
colnames(data) <- c("status", Z.char, "time")

order_t <- order(time, decreasing = F)
z = z[order_t,]
delta = delta[order_t]
time = time[order_t]

########################fitting boosting model:
unique_t = unique(time)[order(unique(time))]
if(length(unique_t) != length(eta)){
  print("error")
}

knot_set=quantile(unique_t,prob=seq(1:(knot-4))/(knot-3))
bs8=splines::bs(unique_t,df=knot, knot=knot_set, intercept=FALSE, degree=3)
bs8=cbind(matrix(1,length(unique_t),1), bs8)
B.spline=bs8
K = dim(bs8)[2]

theta_NR <- matrix(rep(0, ncol(z)*K), ncol=K) # initialization

beta_z   <- rep(0, ncol(z))
beta_t   = rep(0,length(unique_t))

time_used <- proc.time()
system.time(result   <- boosting_logit_expand_both_constant_timevarying_v2_noinv(t = time, z = z, delta = delta,
                                                                                 beta_t_init = beta_t, beta_z_init = beta_z,
                                                                                 unique_t = unique_t,
                                                                                 theta_init = theta_NR,
                                                                                 b_spline = B.spline,
                                                                                 tol=1e-10, Mstop= Mstop,
                                                                                 step_size_day = 1,
                                                                                 step_size_beta = step_size,
                                                                                 StopByInfo = FALSE,
                                                                                 PenalizeGroup = PenalizeGroup) )
time_used <- (proc.time() - time_used)[3]
########################
one_m_sp <- seq(0,1,0.1)

########fitting time-varying effects model

prev_BIC_HTIC_diffstep <- Inf

m_stop_seq <- seq(20,100,10)

model_all_BIC_HTIC <- vector(mode='list', length=length(m_stop_seq))

index_selected_lambda_all <- NULL
lambda_TIC <- NULL


for (m_stop_index in c(1:length(m_stop_seq))) {
  m_stop_tmp <- m_stop_seq[m_stop_index]
  
  
  selected_constant <- unique(result$select_index_constant[1:m_stop_tmp])
  selected_tv <- unique(result$select_index_timevarying[1:m_stop_tmp])
  selected <- unique(c(selected_constant,selected_tv))
  selected <- selected[selected>=0]
  selected <- sort(selected + 1)
  selected
  
  selected_tv <- unique(result$select_index_timevarying[1:m_stop_tmp])
  selected_tv <- selected_tv[selected_tv>=0]
  selected_tv <- sort(selected_tv + 1)
  selected_tv
  
  knot_set=quantile(time[delta==1],prob=seq(1:(knot-4))/(knot-3))
  bs8=splines::bs(unique_t,df=knot, knot=knot_set, intercept=TRUE, degree=3)
  B.spline=bs8
  K = dim(bs8)[2]
  
  
  ti_index <- selected[!selected %in% selected_tv]
  if(length(ti_index)!=0){
    
    # if(length(selected[!selected %in% selected_tv])!=0){
    z_tv = as.matrix(z[,selected_tv])
    z_ti = as.matrix(z[,selected[!selected %in% selected_tv]])
    
    theta_NR <- matrix(rep(0, ncol(z_tv)*K), ncol=K) # initialization
    
    AIC_tmp <- NULL
    TIC_tmp <- NULL
    GIC_tmp <- NULL
    
    BIC_AIC_tmp <- NULL
    BIC_TIC_tmp <- NULL
    BIC_GIC_tmp <- NULL
    BIC_HTIC_tmp <- NULL
    
    for (penalty in penalty_all) {
      res2 <- NR_logit_timevarying_spline_ti(time, z_tv, delta, beta_t_init = beta_t, theta_init = theta_NR, b_spline = B.spline,
                                             z_ti = z_ti, beta_ti_init = rep(0, dim(z_ti)[2]) ,
                                             unique_t = unique_t,
                                             tol = 1e-10,
                                             Mstop = 30,
                                             penalty = penalty,
                                             SmoothMatrix = matrix(0,1,1),
                                             SplineType = "pspline",
                                             IC = TRUE)
      
      AIC_tmp <- c(AIC_tmp, res2$AIC_all)
      TIC_tmp <- c(TIC_tmp, res2$TIC_all)
      GIC_tmp <- c(GIC_tmp, res2$GIC_all)
      
      BIC_AIC_tmp <- c(BIC_AIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_AIC_all)
      BIC_TIC_tmp <- c(BIC_TIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_TIC_all)
      BIC_GIC_tmp <- c(BIC_GIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_GIC_all)
      BIC_HTIC_tmp <- c(BIC_HTIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_HTIC_all)
    }
    
    
    ###BIC_HTIC:##############################################################################
    lambda_TIC <- c(lambda_TIC, penalty_all[which.min(TIC_tmp)])
    res_tmp <- NR_logit_timevarying_spline_ti(time, z_tv, delta, beta_t_init = beta_t, theta_init = theta_NR, b_spline = B.spline,
                                              z_ti = z_ti, beta_ti_init = rep(0, dim(z_ti)[2]) ,
                                              unique_t = unique_t,
                                              tol = 1e-10,
                                              Mstop = 30,
                                              penalty = penalty_all[which.min(TIC_tmp)],
                                              SmoothMatrix = matrix(0,1,1),
                                              SplineType = "pspline",
                                              IC = FALSE)
    
    model_all_BIC_HTIC[[m_stop_index]] <- res_tmp
    
    
  } else{
    z_tv = as.matrix(z[,selected_tv])
    theta_NR <- matrix(rep(0, ncol(z_tv)*K), ncol=K) # initialization
    
    AIC_tmp <- NULL
    TIC_tmp <- NULL
    GIC_tmp <- NULL
    
    BIC_AIC_tmp <- NULL
    BIC_TIC_tmp <- NULL
    BIC_GIC_tmp <- NULL
    BIC_HTIC_tmp <- NULL
    
    for (penalty in penalty_all) {
      res2 <- NR_logit_timevarying_spline(time, z_tv, delta, beta_t_init = beta_t, theta_init = theta_NR, b_spline = B.spline,
                                          unique_t = unique_t,
                                          tol = 1e-10,
                                          Mstop = 30,
                                          penalty = penalty,
                                          SmoothMatrix = matrix(0,1,1),
                                          SplineType = "pspline",
                                          IC = TRUE)
      
      print(res2$Infocrit$df_AIC)
      print(res2$Infocrit$df_TIC)
      print(res2$Infocrit$df_GIC)
      AIC_tmp <- c(AIC_tmp, res2$AIC_all)
      TIC_tmp <- c(TIC_tmp, res2$TIC_all)
      GIC_tmp <- c(GIC_tmp, res2$GIC_all)
      
      BIC_AIC_tmp <- c(BIC_AIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_AIC_all)
      BIC_TIC_tmp <- c(BIC_TIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_TIC_all)
      BIC_GIC_tmp <- c(BIC_GIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_GIC_all)
      BIC_HTIC_tmp <- c(BIC_HTIC_tmp, -2*N*res2$logplkd_vec[length(res2$logplkd_vec)] + log(N)*res2$df_HTIC_all)
    }
    
    ###BIC_HTIC:##############################################################################
    lambda_TIC <- c(lambda_TIC, penalty_all[which.min(TIC_tmp)])
    res_tmp <- NR_logit_timevarying_spline(time, z_tv, delta, beta_t_init = beta_t, theta_init = theta_NR, b_spline = B.spline,
                                           unique_t = unique_t,
                                           tol = 1e-10,
                                           Mstop = 30,
                                           penalty = penalty_all[which.min(TIC_tmp)],
                                           SmoothMatrix = matrix(0,1,1),
                                           SplineType = "pspline",
                                           IC = FALSE)
    
    model_all_BIC_HTIC[[m_stop_index]] <- res_tmp
  }
  
  index_selected_lambda <- which.min(TIC_tmp)
  index_selected_lambda_all <- c(index_selected_lambda_all,index_selected_lambda)
  
  new_BIC_HTIC <- BIC_HTIC_tmp[index_selected_lambda]
  
  BIC_HTIC_diffstep <- c(BIC_HTIC_diffstep, BIC_HTIC_tmp[index_selected_lambda])
  
  if (new_BIC_HTIC > prev_BIC_HTIC_diffstep && m_stop_index >= 5) {
    break
  }
  
  # Update previous values
  prev_BIC_HTIC_diffstep <- new_BIC_HTIC
}
# 
save.image(paste0("N",N,"p",p,"_BoostingVS_NR_BICstop_seed",seed_tmp,".RData"))
