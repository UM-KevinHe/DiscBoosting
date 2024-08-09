rm(list = ls())

# setwd("~/Dropbox (University of Michigan)/Lingfeng Research/VaraibleSelection/Discretized/Timevarying/Paper/figure1_auc_boxplot/boosting//")


library(mvtnorm)
library(discSurv)
library(matrixStats)
library(survival)
library(Rcpp)
library(ggplot2)

sourceCpp("/home/lfluo/VariableSelection/TimeVarying/source/Discrete_logit_Boosting_arma_Both_constant_timevarying.cpp")

# sourceCpp("~/Dropbox (University of Michigan)/Lingfeng Research/VaraibleSelection/Discretized/Timevarying/Discrete_logit_Boosting_arma_Both_constant_timevarying.cpp")
# sourceCpp("~/Dropbox (University of Michigan)/Lingfeng Research/VaraibleSelection/Discretized/Timevarying/Discrete_logit_Boosting_arma_Both_constant_timevarying.cpp")


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

p = 500
step_size = 0.5
PenalizeGroup = FALSE
Mstop <- 5000

beta_effect <- function(x, P){
  return (c(1, cos(pi*x/50), -1, sin(3*pi*x/80), (-1+exp(-0.25*x)), rep(0,P-5)))
}
p_true = 5
p_tv_true = 3

# tt = c(1:length(eta))
# plot_data1 <- data.frame(tt, sin(3*pi*x/80))
# colnames(plot_data1) <- c("time", "beta")
# ggplot(data = plot_data1, aes(x=time, y = beta)) +
#   geom_line() +
#   scale_y_continuous(limits = c(-2,2), breaks = seq(-3,3,1)) +
#   geom_hline(yintercept = 0, linetype = "dashed")



loop = 0
nloop = 100

beta_glm <- NULL
beta_algo <- NULL

theta_all <- NULL

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


one_m_sp <- seq(0,1,0.1)

se_table <- matrix(0, nloop, length(one_m_sp))
sp_table <- matrix(0, nloop, length(one_m_sp))
se_table_tv <- matrix(0, nloop, length(one_m_sp))
sp_table_tv <- matrix(0, nloop, length(one_m_sp))

model_all <- vector(mode = "list", length = nloop)


while(loop < nloop){
  loop = loop + 1
  set.seed(loop*2022)

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
    

    library(survival)
    km_data = data.frame(time, delta)
    fit <- survfit(Surv(time, delta) ~ 1,  data = km_data)
    plot(fit)
    
    ########fitting time-varying effects model
    unique_t = unique(time)[order(unique(time))]
    if(length(unique_t) != length(eta)){
      next
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
    time_used_all <- c(time_used_all, time_used)
    
    theta_all <- rbind(theta_all, result$theta)
    
    model_all[[loop]] <- result
    
    ##################################################################################################################
    ###### record BIC results
    ##################################################################################################################
    
    ##get overall selecetd
    # selected_constant <- unique(result$select_index_constant[1:m_stop])
    # selected_tv <- unique(result$select_index_timevarying[1:m_stop])
    # selected <- unique(c(selected_constant,selected_tv))
    # selected <- selected[selected>=0]
    # selected <- selected + 1
    # 
    # select_index_constant <- result$select_index_constant[1:m_stop]
    # select_index_constant <- select_index_constant[select_index_constant>=0]
    # select_index_constant <- select_index_constant+1
    # select_index_constant_unique <- unique(select_index_constant)
    # select_index_timevarying <- result$select_index_timevarying[1:m_stop]
    # select_index_timevarying <- select_index_timevarying[select_index_timevarying>=0]
    # select_index_timevarying <- select_index_timevarying+1
    # select_index_timevarying_unique <- unique(select_index_timevarying)
    # 
    # 
    # chosen_constant_index <- rep(0,p)
    # chosen_tv_index <- rep(0,p)
    # 
    # ###fpfn-timevarying
    # chosen_tv_index <- NULL
    # tv_table <- table(select_index_timevarying)
    # chosen_tv_index[as.numeric(names(tv_table))] <- tv_table
    # chosen_tv_index_all <- rbind(chosen_tv_index_all, chosen_tv_index)
    # beta_tv = c(0,1,0,1,1,rep(0,p-5))
    # FPFN_tv <- rbind(FPFN_tv, FPFNSeSpLik(TrueBeta = beta_tv, beta =  chosen_tv_index))
    # 
    # ###fpfn-overall
    # beta_estmiate <- rep(0,p)
    # beta_estmiate[selected] <- 1
    # FPFNSeSpLik(TrueBeta = beta, beta =  beta_estmiate)
    # FPFN <- rbind(FPFN, FPFNSeSpLik(TrueBeta = beta, beta =  beta_estmiate))
    # 
    
    
    ##################################################################################################################
    ###### record different stop iteration results
    ##################################################################################################################
    se_all <- NULL
    sp_all <- NULL
    se_all_tv <- NULL
    sp_all_tv <- NULL
    for (i in c(1:length(seq))) {
      stop_tmp = seq[i]
      selected_constant <- unique(result$select_index_constant[1:stop_tmp])
      selected_tv <- unique(result$select_index_timevarying[1:stop_tmp])
      selected <- unique(c(selected_constant,selected_tv))
      selected <- selected[selected>=0]
      selected <- selected + 1
      
      select_index_timevarying <- result$select_index_timevarying[1:stop_tmp]
      select_index_timevarying <- select_index_timevarying[select_index_timevarying>=0]
      select_index_timevarying <- select_index_timevarying+1
      select_index_timevarying_unique <- unique(select_index_timevarying)
      
      chosen_constant_index <- rep(0,p)
      chosen_tv_index <- rep(0,p)
      
      ###fpfn-timevarying
      tv_table <- table(select_index_timevarying)
      chosen_tv_index[as.numeric(names(tv_table))] <- tv_table
      chosen_tv_index_all <- rbind(chosen_tv_index_all, chosen_tv_index)
      beta_tv = c(0,1,0,1,1,rep(0,p-5))
      FPFNtmp <- FPFNSeSpLik(TrueBeta = beta_tv, beta =  chosen_tv_index)
      FPFN_tv_mstop_all[[i]] <- rbind(FPFN_tv_mstop_all[[i]], FPFNSeSpLik(TrueBeta = beta_tv, beta =  chosen_tv_index))
      se_all_tv <- c(se_all_tv, FPFNtmp[3])
      sp_all_tv <- c(sp_all_tv, FPFNtmp[4])
      
      ###fpfn-overall
      beta_estmiate <- rep(0,p)
      beta_estmiate[selected] <- 1
      FPFNtmp_tv <-FPFNSeSpLik(TrueBeta = beta, beta =  beta_estmiate)
      FPFN_mstop_all[[i]] <- rbind(FPFN_mstop_all[[i]], FPFNSeSpLik(TrueBeta = beta, beta =  beta_estmiate))
      se_all <- c(se_all, FPFNtmp[3])
      sp_all <- c(sp_all, FPFNtmp[4])
    }
    
    for (index in 1:length(one_m_sp)) {
      ##overall:
      sp_tmp = one_m_sp[index]
      close_dist <- min(abs(1-sp_all-sp_tmp))
      close_dist_index <- which(abs(1-sp_all-sp_tmp) == close_dist)
      se_tmp = mean(se_all[abs(1-sp_all-sp_tmp) == close_dist])
      if(close_dist < 0.5){
        se_table[loop, index] = se_tmp
        sp_table[loop, index] = (1-sp_all)[close_dist_index][1]
      } else{
        se_table[loop, index] = NA
        sp_table[loop, index] = (1-sp_all)[close_dist_index][1]
      }
      
      
      ##tv:
      close_dist <- min(abs(1-sp_all_tv-sp_tmp))
      close_dist_index <- which(abs(1-sp_all_tv-sp_tmp) == close_dist)
      se_tmp = mean(se_all_tv[abs(1-sp_all_tv-sp_tmp) == close_dist])
      if(close_dist < 0.5){
        se_table_tv[loop, index] = se_tmp
        sp_table_tv[loop, index] = (1-sp_all_tv)[close_dist_index][1]
      } else{
        se_table_tv[loop, index] = NA
        sp_table_tv[loop, index] = (1-sp_all_tv)[close_dist_index][1]
      }
    }
    

    setTxtProgressBar(pb, loop / nloop)

}

#overall variable selection:
# FPFN
# colMeans(FPFN)
# colMeans(FPFN_tv)

# colMeans(chosen_constant_index_all)
# colMeans(chosen_tv_index_all)

save(se_table, sp_table, se_table_tv, sp_table_tv, file = paste0("N_",N,"p_",p,"_boosting_forAUC.RData"))

save.image(file = paste0("N_",N,"p_",p,"_boosting_image.RData"))


