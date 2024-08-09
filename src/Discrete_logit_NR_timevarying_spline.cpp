#include <math.h>
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

List IC_calculate(arma::vec &t, arma::mat &z, arma::vec &delta, arma::mat &b_spline, arma::vec &beta_t, arma::mat &theta,
                  int &max_t, int &P, int &n, int &knot,
                  double &penalty, arma::mat S_matrix);

// [[Rcpp::export]]
double obj_fun(arma::vec &t, arma::mat &z, arma::vec &delta, arma::mat &b_spline, arma::vec &beta_t, arma::mat &theta,
               int &n){


  double loglik = 0;

  arma::mat beta_v = theta*b_spline.t();

  for (int i = 0; i < n; ++i)
  {
    for (int s = 1; s <= t(i); ++s)
    {
      mat z_theta_B_sp = z.row(i)*beta_v.col(s-1);
      double lambda = 1/(1+exp(-beta_t(s-1)-z_theta_B_sp(0,0)));

      if (t(i)==s && delta(i) == 1){
        loglik += log(lambda);
      }
      else{
        loglik += log(1-lambda);
      }

    }
  }
  return loglik/z.n_rows;

}


 

// [[Rcpp::export]]
arma::mat spline_construct(const int knot,
                     const int p,
                     const std::string SplineType = "pspline"){

  arma::mat P_pre  = arma::zeros<arma::mat>(knot,knot);
  P_pre.diag().ones();
  P_pre      = diff(P_pre);
  arma::mat S_pre  = P_pre.t()*P_pre;

  arma::mat S_matrix     = arma::zeros<arma::mat>(p*knot, p*knot);
  for (int i = 0; i < p; ++i)
  {
    S_matrix.submat(i*knot,i*knot,i*knot+knot-1, i*knot+knot-1) = S_pre;
  }

  return S_matrix;
}

// [[Rcpp::export]]
arma::mat spline_construct_ti(const int knot,
                     const int p,
                     const std::string SplineType = "pspline"){

  int knot_tv = knot - 1;

  arma::mat P_pre  = arma::zeros<arma::mat>(knot_tv,knot_tv);
  P_pre.diag().ones();
  P_pre      = diff(P_pre);
  arma::mat S_pre  = P_pre.t()*P_pre;

  arma::mat S_matrix     = arma::zeros<arma::mat>(p*knot, p*knot);
  for (int i = 0; i < p; ++i)
  {
    S_matrix.submat((i+1)+i*knot_tv,(i+1)+i*knot_tv, i*knot_tv+knot_tv+i, i*knot_tv+knot_tv+i) = S_pre;
  }


  return S_matrix;
}


// [[Rcpp::export]]
arma::mat spline_construct2(const int knot,
                      const int p,
                      const std::string SplineType,
                      const arma::mat &SmoothMatrix){

  arma::mat S_matrix     = arma::zeros<arma::mat>(p*knot, p*knot);

  arma::mat S_pre    = SmoothMatrix;
  S_matrix     = arma::zeros<arma::mat>(p*knot, p*knot);
  for (int i = 0; i < p; ++i)
  {
    S_matrix.submat(i*knot,i*knot,i*knot+knot-1, i*knot+knot-1) = S_pre;
  }
  

  return S_matrix;
}

// [[Rcpp::export]]
arma::mat spline_construct2_ti(const int knot,
                      const int p,
                      const std::string SplineType,
                      const arma::mat &SmoothMatrix){

  int knot_tv = knot - 1;
  
  arma::mat S_pre    = SmoothMatrix;
  arma::mat S_matrix     = arma::zeros<arma::mat>(p*knot, p*knot);
  for (int i = 0; i < p; ++i)
  {
    S_matrix.submat((i+1)+i*knot_tv,(i+1)+i*knot_tv, i*knot_tv+knot_tv+i, i*knot_tv+knot_tv+i) = S_pre;
  }


  return S_matrix;
}


List Update_logit(arma::vec &t, arma::mat &z, arma::vec &delta, arma::mat &b_spline, 
                  arma::vec &beta_t, arma::mat &theta,
                  int &max_t, int &P, int &n, int &knot){
  double loglik = 0;

  int new_dim = P*knot;

  vec score_t = arma::zeros<vec>(max_t);
  vec score_v = arma::zeros<vec>(new_dim);
  vec info_t  = arma::zeros<vec>(max_t);
  mat info_v  = arma::zeros<mat>(new_dim, new_dim);
  mat info_tv = arma::zeros<mat>(max_t, new_dim);

  arma::mat beta_v = theta*b_spline.t();

  for (int i = 0; i < n; ++i)
  {
    mat zzT = z.row(i).t()*z.row(i);

    for (int s = 1; s <= t(i); ++s)
    {
      vec B_sp_tmp     = b_spline.row(s-1).t();
      mat z_theta_B_sp = z.row(i)*beta_v.col(s-1);
      mat zB_kron      = kron(z.row(i).t(),B_sp_tmp);

      double lambda = 1/(1+exp(-beta_t(s-1)-z_theta_B_sp(0,0)));
      score_t(s-1) -= lambda;
      score_v      -= lambda*zB_kron;

      info_t(s-1)   += lambda*(1-lambda);
      info_v        += lambda*(1-lambda)*(kron(zzT, B_sp_tmp*B_sp_tmp.t()));
      info_tv.row(s-1) += (lambda*(1-lambda))*zB_kron.t();

      if (t(i)==s && delta(i) == 1){
        loglik += log(lambda);
        score_t(s-1) += 1;
        score_v += zB_kron;
      }
      else{
        loglik += log(1-lambda);
      }
    }
  }

  //new faster code:
  vec info_t_inv = arma::ones<vec>(max_t)/info_t;
  mat info_t_inv_rep = repmat(info_t_inv,1,new_dim);
  mat schur   = arma::zeros<mat>(new_dim,new_dim);

  mat F = (info_t_inv_rep)%(info_tv);
  schur = info_v-(info_tv.t())*F;

  mat schur_inv_Ft = solve(schur, F.t(), solve_opts::allow_ugly);
  mat schur_inv_scorv_v = solve(schur, score_v, solve_opts::allow_ugly);

  vec step_t    = info_t_inv%score_t + F*schur_inv_Ft*score_t - F*schur_inv_scorv_v;
  vec step      = -schur_inv_Ft*score_t + schur_inv_scorv_v;

  ////////////////////////////////////////////////////////////////////////////////
  //end
  double inc = (dot(score_t, step_t) + dot(score_v, step))/z.n_rows;

  return List::create(_["loglik"]=loglik, _["step_t"]=step_t, 
                      _["step_theta"]=reshape(step, size(theta.t())).t(), 
                      _["inc"]=inc,
                      _["info_v"]=info_v);
}



// [[Rcpp::export]]
List NR_logit_timevarying(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init,
                          arma::mat &theta_init,
                          arma::mat &b_spline,
                          arma::vec &unique_t,
                          double &tol, int &Mstop,
                          const string &btr = "dynamic",
                          const string &stop = "ratch",
                          const double &s=1e-2,
                          const double &t_adjust=0.6){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int max_t       = max(t);             // maximum time indicator
  int knot        = b_spline.n_cols;
  arma::mat theta = theta_init;

  vec beta_t  = beta_t_init;

  int new_dim = P*knot;
  mat info_v = arma::zeros<mat>(P, P);

  List result;
  List update;
  double loglik, logplkd_init;
  NumericVector logplkd_vec;

  loglik = obj_fun(t, z, delta, b_spline, beta_t, theta, n);
  logplkd_vec.push_back(loglik);

  unsigned int iter = 0, btr_max = 1000 , btr_ct = 0;
  double crit = 1.0, v = 1.0, inc, diff_logplkd, rhs_btr = 0;

  while (iter < Mstop && crit > tol) {
    ++iter;
    update = Update_logit(t, z, delta, b_spline, beta_t, theta, 
                          max_t, P, n, knot);

    v = 1.0;
    vec step_t =  update["step_t"];
    mat step_theta = update["step_theta"];
    inc = update["inc"];

    vec beta_t_tmp   = beta_t + step_t;
    mat theta_tmp    = theta +  step_theta;

    double logplkd_tmp = obj_fun(t, z, delta, b_spline, beta_t_tmp, theta_tmp, n);
    
    diff_logplkd = logplkd_tmp - loglik;
    if (btr=="dynamic")      rhs_btr = inc;
    else if (btr=="static")  rhs_btr = 1.0;

    while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max){
      ++btr_ct;
      v *= t_adjust;
      beta_t_tmp = beta_t + v * step_t;
      theta_tmp = theta + v * step_theta;
      double logplkd_tmp = obj_fun(t, z, delta, b_spline, beta_t, theta, n);
      diff_logplkd = logplkd_tmp - loglik;
    }


    beta_t  = beta_t_tmp; 
    theta   = theta_tmp; 
    if (iter==1) logplkd_init = loglik;

    if (stop=="relch")
      crit = abs(diff_logplkd/(diff_logplkd+loglik));
    else if (stop=="ratch")
      crit = abs(diff_logplkd/(diff_logplkd+loglik-logplkd_init));


    loglik += diff_logplkd;
    logplkd_vec.push_back(loglik);
    Rcout<<"loglik: "<<loglik<<endl;
    if (crit <= tol){
      arma::mat info_inv_tmp = update["info_v"];
      info_v = info_inv_tmp;
      Rcout<<"algorithm converged after "<<iter<<" iterations"<<endl;
    }

  }
  
  result["info_v"] = info_v;
  result["update"] = update;
  result["logplkd_vec"] = logplkd_vec;
  result["theta"] = theta;
  result["beta_t"] = beta_t;
  return result;

}



List Update_logit_spline(arma::vec &t, arma::mat &z, arma::vec &delta, arma::mat &b_spline, arma::vec &beta_t, arma::mat &theta,
                  int &max_t, int &P, int &n, int &knot,
                  double &penalty, arma::mat S_matrix){
  double loglik = 0;

  int new_dim = P*knot;

  vec score_t = arma::zeros<vec>(max_t);
  vec score_v = arma::zeros<vec>(new_dim);
  vec info_t  = arma::zeros<vec>(max_t);
  mat info_v  = arma::zeros<mat>(new_dim, new_dim);
  mat info_tv = arma::zeros<mat>(max_t, new_dim);

  arma::mat beta_v = theta*b_spline.t();

  for (int i = 0; i < n; ++i)
  {
    mat zzT = z.row(i).t()*z.row(i);

    for (int s = 1; s <= t(i); ++s)
    {
      vec B_sp_tmp     = b_spline.row(s-1).t();
      mat z_theta_B_sp = z.row(i)*beta_v.col(s-1);
      mat zB_kron      = kron(z.row(i).t(),B_sp_tmp);

      double lambda = 1/(1+exp(-beta_t(s-1)-z_theta_B_sp(0,0)));
      score_t(s-1) -= lambda;
      vec score_theta = lambda*zB_kron;
      score_v      -= lambda*zB_kron - (penalty*S_matrix)*vectorise(theta.t(), 0);

      info_t(s-1)   += lambda*(1-lambda);
      info_v        += lambda*(1-lambda)*(kron(zzT, B_sp_tmp*B_sp_tmp.t())) + penalty*S_matrix;

      info_tv.row(s-1) += (lambda*(1-lambda))*zB_kron.t();

      if (t(i)==s && delta(i) == 1){
        loglik       += log(lambda);
        score_t(s-1) += 1;
        score_v      += zB_kron;
      }
      else{
        loglik       += log(1-lambda);
      }

    }
  }

  vec info_t_inv = arma::ones<vec>(max_t)/info_t;
  mat info_t_inv_rep = repmat(info_t_inv,1,new_dim);
  mat schur   = arma::zeros<mat>(new_dim,new_dim);

  mat F = (info_t_inv_rep)%(info_tv);
  schur = info_v-(info_tv.t())*F;
  mat schur_inv_Ft = solve(schur, F.t(), solve_opts::allow_ugly);
  mat schur_inv_scorv_v = solve(schur, score_v, solve_opts::allow_ugly);

  vec step_t    = info_t_inv%score_t + F*schur_inv_Ft*score_t - F*schur_inv_scorv_v;
  vec step      = -schur_inv_Ft*score_t + schur_inv_scorv_v;

  double inc = (dot(score_t, step_t) + dot(score_v, step))/z.n_rows;

  return List::create(_["loglik"]=loglik, _["step_t"]=step_t, 
                      _["step_theta"]=reshape(step, size(theta.t())).t(), 
                      _["inc"]=inc,
                      _["info_v"]=info_v);
}


// [[Rcpp::export]]
List NR_logit_timevarying_spline( arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, 
                                  arma::mat &theta_init,
                                  arma::mat &b_spline,
                                  arma::vec &unique_t,
                                  arma::mat &SmoothMatrix,
                                  double &penalty,
                                  double &tol, int &Mstop,
                                  const string &btr = "dynamic",
                                  const string &stop = "ratch",
                                  const double &s=1e-2,
                                  const double &t_adjust=0.6,
                                  const std::string &SplineType = "pspline",
                                  const bool &IC = false){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int max_t       = max(t);             // maximum time indicator
  int knot        = b_spline.n_cols;
  arma::mat theta = theta_init;

  penalty /= n;


  vec beta_t  = beta_t_init;

  int new_dim = P*knot;

  mat info_t_inv = arma::zeros<mat>(max_t, max_t);
  mat schur   = arma::zeros<mat>(new_dim,new_dim);
  mat info_v = arma::zeros<mat>(P, P);


  arma::mat S_matrix;
  if(SplineType == "pspline") {
    S_matrix        = spline_construct(knot, P, SplineType);  
  }
  else{
    S_matrix        = spline_construct2(knot, P, SplineType, SmoothMatrix);  
  }


  List result;
  List update;
  double loglik, logplkd_init;
  NumericVector logplkd_vec;

  loglik = obj_fun(t, z, delta, b_spline, beta_t, theta, n);
  logplkd_vec.push_back(loglik);

  unsigned int iter = 0, btr_max = 1000, btr_ct = 0;
  double crit = 1.0, v = 1.0, inc, diff_logplkd, rhs_btr = 0;


  while (iter < Mstop && crit > tol) {
    ++iter;
    update = Update_logit_spline(t, z, delta, b_spline, beta_t, theta, 
                                  max_t, P, n, knot, penalty, S_matrix);
    v = 1.0;
    vec step_t =  update["step_t"];
    mat step_theta = update["step_theta"];
    inc = update["inc"];
    vec beta_t_tmp   = beta_t + step_t;
    mat theta_tmp    = theta +  step_theta;

    double logplkd_tmp = obj_fun(t, z, delta, b_spline, beta_t_tmp, theta_tmp, n);
    diff_logplkd = logplkd_tmp - loglik;

    if (btr=="dynamic")      rhs_btr = inc;
    else if (btr=="static")  rhs_btr = 1.0;

    while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max){
      ++btr_ct;
      v *= t_adjust;
      beta_t_tmp = beta_t + v * step_t;
      theta_tmp = theta + v * step_theta;
      double logplkd_tmp = obj_fun(t, z, delta, b_spline, beta_t_tmp, theta_tmp, n);
      diff_logplkd = logplkd_tmp - loglik;
    }
    beta_t  = beta_t_tmp; 
    theta   = theta_tmp; 
    if (iter==1) logplkd_init = loglik;

    if (stop=="relch")
      crit = abs(diff_logplkd/(diff_logplkd+loglik));
    else if (stop=="ratch")
      crit = abs(diff_logplkd/(diff_logplkd+loglik-logplkd_init));

    loglik += diff_logplkd;
    logplkd_vec.push_back(loglik);
    Rcout<<"loglik: "<<loglik<<endl;
    if (crit <= tol)
    {
      arma::mat info_v_tmp = update["info_v"];
      info_v = info_v_tmp;
      Rcout<<"algorithm converged after "<<iter<<" iterations"<<endl;
    }
  }

  List Infocrit;
  NumericVector AIC_all, TIC_all, GIC_all;
  NumericVector df_AIC_all, df_TIC_all, df_GIC_all, df_HTIC_all;

  if (IC){
    Infocrit = IC_calculate(t, z, delta, b_spline, beta_t, theta, 
                            max_t, P, n, knot, penalty, S_matrix);

    double df_AIC = Infocrit["df_AIC"];
    double df_TIC = Infocrit["df_TIC"];
    double df_GIC = Infocrit["df_GIC"];
    double df_HTIC = Infocrit["df_HTIC"];

    //AIC:
    AIC_all.push_back(-2*loglik*n + 2*df_AIC);
    df_AIC_all.push_back(df_AIC);
    //TIC:
    TIC_all.push_back(-2*loglik*n + 2*df_TIC);
    df_TIC_all.push_back(df_TIC);
    //GIC:
    GIC_all.push_back(-2*loglik*n + 2*df_GIC);
    df_GIC_all.push_back(df_GIC);
    //HTIC:
    df_HTIC_all.push_back(df_HTIC);
  }

  result["info_v"] = info_v;
  result["logplkd_vec"] = logplkd_vec;
  result["theta"] = theta;
  result["beta_t"] = beta_t;
  result["iter"]  = iter;
  result["S_matrix"] = S_matrix;
  result["Infocrit"] = Infocrit;
  result["AIC_all"]  = AIC_all;
  result["TIC_all"]  = TIC_all;
  result["GIC_all"]  = GIC_all;
  result["df_AIC_all"]  = df_AIC_all;
  result["df_TIC_all"]  = df_TIC_all;
  result["df_GIC_all"]  = df_GIC_all;
  result["df_HTIC_all"]  = df_HTIC_all;

  return result;
}


// [[Rcpp::export]]
List IC_calculate(arma::vec &t, arma::mat &z, arma::vec &delta, arma::mat &b_spline, arma::vec &beta_t, arma::mat &theta,
                  int &max_t, int &P, int &n, int &knot,
                  double &penalty, arma::mat S_matrix){
  double loglik = 0;

  int new_dim = P*knot;

  vec score_t = arma::zeros<vec>(max_t);
  vec score_v = arma::zeros<vec>(new_dim);
  vec info_t  = arma::zeros<vec>(max_t);
  mat info_v  = arma::zeros<mat>(new_dim, new_dim);
  mat info_v_p  = arma::zeros<mat>(new_dim, new_dim);
  mat info_tv = arma::zeros<mat>(max_t, new_dim);

  vec grad_tmp = arma::zeros<vec>(new_dim);
  vec grad_p_tmp;
  arma::mat grad_all = arma::zeros<arma::mat>(new_dim + max_t, n);
  arma::mat grad_p_all = arma::zeros<arma::mat>(new_dim + max_t, n);

  arma::mat info_J   = arma::zeros<arma::mat>(new_dim + max_t, new_dim + max_t);
  arma::mat info_J_p = arma::zeros<arma::mat>(new_dim + max_t, new_dim + max_t);
  arma::mat info_J_p_gic = arma::zeros<arma::mat>(new_dim + max_t, new_dim + max_t);

  arma::mat beta_v = theta*b_spline.t();

  for (int i = 0; i < n; ++i)
  {
    mat zzT = z.row(i).t()*z.row(i);

    vec grad_t_tmp      = arma::zeros<vec>(max_t);
    for (int s = 1; s <= t(i); ++s)
    {
      vec B_sp_tmp      = b_spline.row(s-1).t();
      mat z_theta_B_sp  = z.row(i)*beta_v.col(s-1);
      mat zB_kron       = kron(z.row(i).t(),B_sp_tmp);

      double lambda     = 1/(1+exp(-beta_t(s-1)-z_theta_B_sp(0,0)));
      grad_t_tmp(s-1)  -= lambda;
      vec score_theta   = -lambda*zB_kron;
      vec score_theta_p = score_theta + (penalty*S_matrix)*vectorise(theta.t(), 0);

      grad_tmp          = score_theta;
      grad_p_tmp        = score_theta_p;

      info_t(s-1)      += lambda*(1-lambda);

      mat info_v11      = lambda*(1-lambda)*(kron(zzT, B_sp_tmp*B_sp_tmp.t()));
      mat info_v11_p    = info_v11 + penalty*S_matrix;

      info_v           += info_v11;
      info_v_p         += info_v11_p;
      info_tv.row(s-1) += (lambda*(1-lambda))*zB_kron.t();

      if (t(i)==s && delta(i) == 1){
        grad_t_tmp(s-1)   += 1;
        grad_tmp       += zB_kron;
        grad_p_tmp     += zB_kron;
      }
    }

    grad_all.col(i)   = join_cols(grad_t_tmp, grad_tmp);
    grad_p_all.col(i) = join_cols(grad_t_tmp, grad_p_tmp);
  }

  for (int i = 0; i < n; ++i)
  {
    vec grad_tmp = grad_all.col(i);
    vec grad_p_tmp = grad_p_all.col(i);

    info_J       += grad_tmp * grad_tmp.t();
    info_J_p     += grad_p_tmp*grad_p_tmp.t();
    info_J_p_gic += grad_p_tmp*grad_tmp.t();
  }

  //AIC:
  mat info_t_diag = arma::zeros<mat>(max_t, max_t);
  info_t_diag.diag() = info_t;

  mat I_0 = join_cols(join_rows(info_t_diag, info_tv), 
                      join_rows(info_tv.t(), info_v));
  mat I_p = join_cols(join_rows(info_t_diag, info_tv), 
                      join_rows(info_tv.t(), info_v_p));

  mat I_p_inv = inv(I_p);

  arma::mat matAIC = I_0*I_p_inv;
  double df_AIC  = trace(matAIC);

  //TIC:
  arma::mat matTIC = matAIC*info_J_p*I_p_inv;
  double df_TIC  = trace(matTIC);

  //GIC:
  arma::mat matGIC = I_p_inv*info_J_p_gic;
  double df_GIC  = trace(matGIC);

  //HTIC:
  arma::mat matHTIC = 2*I_p_inv*I_0 - matAIC*matAIC;
  double df_HTIC = trace(matHTIC);

  return List::create(_["df_AIC"] = df_AIC,
                      _["df_TIC"] = df_TIC,
                      _["df_GIC"] = df_GIC,
                      _["df_HTIC"] = df_HTIC);
}














