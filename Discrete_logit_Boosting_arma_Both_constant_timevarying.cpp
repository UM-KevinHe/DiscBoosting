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

  
List dbeta_logit(arma::vec &t, arma::mat &Z, arma::vec &delta, arma::vec &beta_t, arma::vec &beta_v, int &K, int &p, int &n){
  double loglik=0;               //loglikelihood
  arma::vec score_t = arma::zeros<arma::vec>(K);  //score vector w.r.t time(gamma)
  arma::vec score_v = arma::zeros<arma::vec>(p);
  arma::vec info_t = arma::zeros<arma::vec>(K);

  double delta_prime=0;

  for (int i = 0 ; i < n ; i++){
    for (int s = 1 ; s <= t[i] ; s++){
      vec z_i_betav = Z.row(i)*beta_v;
      double lambda = 1/(1+exp(-beta_t(s-1)-z_i_betav[0]));
      loglik        = loglik + log(1-lambda);
      score_t[s-1] -= lambda;
      score_v       = score_v - lambda * Z.row(i).t();
      info_t[s-1]  += lambda*(1-lambda);
      if (t[i] != s){
        delta_prime = 0;
      }
      if (t[i] == s){
        delta_prime = delta[i];
      }
      loglik       += delta_prime*log(lambda)-delta_prime*log(1-lambda);
      score_t[s-1]  = score_t[s-1] + delta_prime;
      score_v      += delta_prime*Z.row(i).t();
    }
  }

  List result;
  result["loglik"]  = loglik;
  result["score_t"] = score_t;
  result["score_v"] = score_v;
  result["info_t"]  = info_t;
  
  return result;
}

// [[Rcpp::export]]
double obj_function(arma::vec &t, arma::mat &Z, arma::vec &delta, arma::vec &beta_t, arma::mat &theta, arma::mat &b_spline, int &K, int &p, int &n){
  double loglik=0;               //loglikelihood
  arma::vec score_t = arma::zeros<arma::vec>(K);  //score vector w.r.t time(gamma)
  arma::vec score_v = arma::zeros<arma::vec>(p);
  arma::vec info_t = arma::zeros<arma::vec>(K);

  double delta_prime=0;
  arma::mat beta_v = theta*b_spline.t();

  vec z_i_betav;
  for (int i = 0 ; i < n ; i++){
    for (int s = 1 ; s <= t(i) ; s++){
      z_i_betav = Z.row(i)*beta_v.col(s-1 );
      double lambda = 1/(1+exp(-beta_t(s-1)-z_i_betav(0)));
      loglik        = loglik + log(1-lambda);
      if (t(i) != s){
        delta_prime = 0;
      }
      if (t(i) == s){
        delta_prime = delta(i);
      }
      loglik       += delta_prime*log(lambda)-delta_prime*log(1-lambda);
    }
  }

  return loglik;
}

// List score_function(arma::vec &t, arma::mat &Z, arma::vec &delta, arma::vec &beta_t, arma::mat &theta, arma::mat &b_spline, int &K, int &p, int &n){
//   double loglik=0;               //loglikelihood
//   arma::vec score_t = arma::zeros<arma::vec>(K);  //score vector w.r.t time(gamma)
//   arma::vec score_v = arma::zeros<arma::vec>(p);

//   double delta_prime=0;

//   arma::mat beta_v = theta*b_spline.t();

//   for (int i = 0 ; i < n ; i++){
//     for (int s = 1 ; s <= t[i] ; s++){
//       vec z_i_betav = Z.row(i)*beta_v.col(i);
//       double lambda = 1/(1+exp(-beta_t(s-1)-z_i_betav(0)));
//       loglik        = loglik + log(1-lambda);
//       score_t[s-1] -= lambda;
//       score_v       = score_v - lambda * Z.row(i).t();
//       if (t[i] != s){
//         delta_prime = 0;
//       }
//       if (t[i] == s){
//         delta_prime = delta[i];
//       }
//       loglik       += delta_prime*log(lambda)-delta_prime*log(1-lambda);
//       score_t[s-1]  = score_t[s-1] + delta_prime;
//       score_v      += delta_prime*Z.row(i).t();
//     }
//   }

//   List result;
//   result["loglik"]  = loglik;
//   result["score_t"] = score_t;
//   result["score_v"] = score_v;
  
//   return result;
// }




// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                           arma::mat &theta_init,
                           arma::mat &b_spline,
                           arma::vec &unique_t,
                           double tol, int Mstop, 
                           double step_size_day,
                           double step_size_beta,
                           const bool StopByInfo = false,
                           const bool PenalizeGroup = false){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);


  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde,K);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K,K);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_inv_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < P; ++j)
  {
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    // cout<<"tmp row and col: "<<j*knot+1<<" "<<(knot*(j+1)-1)<<endl;
    Z_Tilde_square_inv_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = inv(tmp);
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j)
  {
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  mat Z_Tilde_square_inv = zeros<mat>(knot, knot*P);
  if(StopByInfo){
    for (int j = 0; j < P; ++j)
    {
      mat tmp = Z_Tilde.cols(j*knot,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot,(knot*(j+1)-1));
      Z_Tilde_square_inv.submat(0, j*knot, knot-1, (knot*(j+1)-1)) = inv(tmp);
    }
  }
  cout <<"expanded data sample size :"<< N_tilde<<endl;

  double ll0,ll,ll2;

  //pseudo-outcome related variables:

  List result;
  arma::mat beta_v;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();
    for (int j = 0; j < P; ++j)
    {
      // cout<<"j_index: "<<j*knot<<" "<<knot*(j+1)-1<<endl;
      //Time constant part:
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      // Rcout << "tmp_constant length: "<<tmp_constant.n_elem<<endl;
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      // Rcout << "tmp_tv length: "<<tmp_tv.n_elem<<endl;
      // Rcout << "1, j*knot, knot-1, (knot*(j+1)-1: 0 "<<j*(knot-1)<<" "<<knot-1<<" "<< ((knot-1)*(j+1)-1)<<endl;
      theta_Tilde.submat(1,j,knot-1,j)  = Z_Tilde_square_inv_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1))*tmp_tv;
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    // if (m != 1){
    //   double tmp = min;
    //   cout<<"tmp: "<<tmp<<endl;
    //   int min_lastIter = min;
    //   cout<<"min_lastIter: "<<min_lastIter<<endl;
    // }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();

    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
    }


    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    // if(StopByInfo){
    //   H_matrix  = Z_Tilde.cols(min*knot,(knot*(min+1)-1))*Z_Tilde_square_inv.submat(0, min*knot, knot-1, (knot*(min+1)-1))*Z_Tilde.cols(min*knot,(knot*(min+1)-1)).t();
    //   vec W_tmp = zeros<vec>(N_tilde);
    //   W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
    //   W_matrix  = repmat(W_tmp,1,N_tilde);
    //   if(m==1){
    //     B_matrix = W_matrix%H_matrix;
    //   }
    //   else{
    //     B_matrix = B_matrix + W_matrix%H_matrix*(Identity_mat-B_matrix);
    //   }
    //   double df = trace(B_matrix);
    //   df += beta_t.n_elem;
    //   AIC.push_back(-2*ll2+2*df);
    //   df_all.push_back(df);
    // }
    if(StopByInfo){
      Z_Tilde_tmp = Z_Tilde.cols(min*knot,(knot*(min+1)-1));
      Z_Tilde_square_inv_tmp = Z_Tilde_square_inv.submat(0, min*knot, knot-1, (knot*(min+1)-1));
      vec W_tmp = zeros<vec>(N_tilde);
      W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
      W_matrix  = repmat(W_tmp,1,N_tilde);
      if(m==1){
        mat H_matrix = Z_Tilde_tmp*Z_Tilde_square_inv_tmp*Z_Tilde_tmp.t();
        B_matrix = W_matrix%H_matrix;
      }
      else{
        Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
        inv_Z_I_minus_B = Z_Tilde_square_inv_tmp*Z_I_minus_B;
        Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
        B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
      }
      double df = trace(B_matrix);
      df += beta_t.n_elem;
      AIC.push_back(-2*ll2+2*df);
      df_all.push_back(df);
    }

    // cout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 
  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["m"]=m;
  result["logllk"] = logllk;
  result["n_count_index"] = n_count_index;
  // result["U_k"] = U_k;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  return result;
}


double obj_function_biomial(const int &N_tilde, const arma::mat &Bm, const arma::vec &Y_Tilde, 
                            const double &delta_bound,
                            const bool &truncate = true) {
  vec BmY = Bm*Y_Tilde;
  Rcout << "Bm: " << Bm.n_rows <<" "<<Bm.n_cols <<endl;
  Rcout << "BmY: " << BmY.n_rows <<" "<<BmY.n_cols <<endl;

  int count = 0;
  if (truncate){
    for (int i = 0; i < N_tilde; ++i){
      if (BmY(i) < 0 || BmY(i) >1){
          // Rcout << "BmY: " << BmY(i) <<endl;
          BmY(i) = max(min(BmY(i),1-delta_bound),delta_bound);
          count++;
        }
    }
  }

  // Rcout << "BmY: " << accu(BmY) <<endl;
  // Rcout << "Y_Tilde: " << accu(Y_Tilde) <<endl;
  vec ones = arma::ones(N_tilde);
  vec logBmY = Y_Tilde%log(BmY);
  vec neg_logBmY = (ones-Y_Tilde)%log(ones-BmY);

  Rcout << "logBmY: " << accu(logBmY) <<endl;
  Rcout << "neg_logBmY: " << accu(neg_logBmY) <<endl;
  Rcout << "count: " << count <<endl;
  double res = accu(logBmY) + accu(neg_logBmY);

  return res;
}




// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_v2(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                           arma::mat &theta_init,
                           arma::mat &b_spline,
                           arma::vec &unique_t,
                           double &tol, int &Mstop, 
                           double step_size_day,
                           double step_size_beta,
                           const bool &StopByInfo = false,
                           const bool &PenalizeGroup = false,
                           const bool &truncate = true,
                           const double &delta_bound = 1e-5){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_inv_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < P; ++j)
  {
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    // cout<<"tmp row and col: "<<j*knot+1<<" "<<(knot*(j+1)-1)<<endl;
    Z_Tilde_square_inv_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = inv(tmp);
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j)
  {
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  mat Z_Tilde_square_inv = zeros<mat>(knot, knot*P);
  if(StopByInfo){
    for (int j = 0; j < P; ++j)
    {
      mat tmp = Z_Tilde.cols(j*knot,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot,(knot*(j+1)-1));
      Z_Tilde_square_inv.submat(0, j*knot, knot-1, (knot*(j+1)-1)) = inv(tmp);
    }
  }
  cout <<"expanded data sample size :"<< N_tilde<<endl;
  // // mat H_matrix_all = zeros<mat>(N_tilde,N_tilde*P);
  // if(StopByInfo){
  //   for (int j = 0; j < P; ++j)
  //   {
  //     mat tmp  = Z_Tilde.cols(j*knot,(knot*(j+1)-1))*Z_Tilde_square_inv.submat(0, j*knot, knot-1, (knot*(j+1)-1))*Z_Tilde.cols(j*knot,(knot*(j+1)-1)).t();
  //     // H_matrix_all.submat(0, j*N_tilde, N_tilde-1, (N_tilde*(j+1)-1)) = tmp;
  //   }
  // }





  double ll0,ll,ll2;

  //pseudo-outcome related variables:

  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();
    for (int j = 0; j < P; ++j)
    {
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      theta_Tilde.submat(1,j,knot-1,j)  = Z_Tilde_square_inv_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1))*tmp_tv;
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();

    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }


    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    if(StopByInfo){
      //update only the time-varying parts
      if(!select_constant){
        Z_Tilde_tmp = Z_Tilde.cols(min*knot+1,(knot*(min+1)-1));
        Z_Tilde_square_inv_tmp = Z_Tilde_square_inv_tv.submat(0, min*(knot-1), knot-2, ((knot-1)*(min+1)-1));
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp*Z_Tilde_square_inv_tmp*Z_Tilde_tmp.t();
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
          inv_Z_I_minus_B = Z_Tilde_square_inv_tmp*Z_I_minus_B;
          Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      } 
      //update only the constant parts
      else{
        Z_Tilde_tmp_constant = Z_Tilde.col(min*knot);
        Z_Tilde_square_inv_tmp_constant = Z_Tilde_square_inv_constant(min);
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp_constant*Z_Tilde_tmp_constant.t();
          H_matrix = Z_Tilde_square_inv_tmp_constant*H_matrix;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B_constant = (Z_Tilde_tmp_constant.t())*(Identity_mat-B_matrix);
          Z_I_minus_B_constant = Z_Tilde_square_inv_tmp_constant * Z_I_minus_B_constant;
          Z_inv_Z_I_minus_B_constant = Z_Tilde_tmp_constant*Z_I_minus_B_constant;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B_constant;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      }
    }

    // Rcout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 
  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["m"]=m;
  result["logllk"] = logllk;
  result["n_count_index"] = n_count_index;
  // result["U_k"] = U_k;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  // result["logBinomial_all"] = logBinomial_all;
  return result;
}


// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_v2_noinv(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                           arma::mat &theta_init,
                           arma::mat &b_spline,
                           arma::vec &unique_t,
                           double &tol, int &Mstop, 
                           double step_size_day,
                           double step_size_beta,
                           const bool &StopByInfo = false,
                           const bool &PenalizeGroup = false,
                           const bool &truncate = true,
                           const double &delta_bound = 1e-5){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < P; ++j)
  {
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = tmp;
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j)
  {
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  // mat Z_Tilde_square_inv = zeros<mat>(knot, knot*P);
  // if(StopByInfo){
  //   for (int j = 0; j < P; ++j)
  //   {
  //     mat tmp = Z_Tilde.cols(j*knot,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot,(knot*(j+1)-1));
  //     Z_Tilde_square_inv.submat(0, j*knot, knot-1, (knot*(j+1)-1)) = inv(tmp);
  //   }
  // }
  cout <<"expanded data sample size :"<< N_tilde<<endl;
  // // mat H_matrix_all = zeros<mat>(N_tilde,N_tilde*P);
  // if(StopByInfo){
  //   for (int j = 0; j < P; ++j)
  //   {
  //     mat tmp  = Z_Tilde.cols(j*knot,(knot*(j+1)-1))*Z_Tilde_square_inv.submat(0, j*knot, knot-1, (knot*(j+1)-1))*Z_Tilde.cols(j*knot,(knot*(j+1)-1)).t();
  //     // H_matrix_all.submat(0, j*N_tilde, N_tilde-1, (N_tilde*(j+1)-1)) = tmp;
  //   }
  // }





  double ll0,ll,ll2;

  //pseudo-outcome related variables:

  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();
    for (int j = 0; j < P; ++j)
    {
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      theta_Tilde.submat(1,j,knot-1,j)  = solve(Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)), tmp_tv, solve_opts::fast);
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();

    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }


    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    if(StopByInfo){
      //update only the time-varying parts
      if(!select_constant){
        Z_Tilde_tmp = Z_Tilde.cols(min*knot+1,(knot*(min+1)-1));
        Z_Tilde_square_inv_tmp = Z_Tilde_square_tv.submat(0, min*(knot-1), knot-2, ((knot-1)*(min+1)-1));
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t = solve(Z_Tilde_square_inv_tmp, Z_Tilde_tmp.t(), solve_opts::fast);
          mat H_matrix = Z_Tilde_tmp * Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
          inv_Z_I_minus_B = solve(Z_Tilde_square_inv_tmp, Z_I_minus_B, solve_opts::fast);
          Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      } 
      //update only the constant parts
      else{
        Z_Tilde_tmp_constant = Z_Tilde.col(min*knot);
        Z_Tilde_square_inv_tmp_constant = Z_Tilde_square_inv_constant(min);
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp_constant*Z_Tilde_tmp_constant.t();
          H_matrix = Z_Tilde_square_inv_tmp_constant*H_matrix;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B_constant = (Z_Tilde_tmp_constant.t())*(Identity_mat-B_matrix);
          Z_I_minus_B_constant = Z_Tilde_square_inv_tmp_constant * Z_I_minus_B_constant;
          Z_inv_Z_I_minus_B_constant = Z_Tilde_tmp_constant*Z_I_minus_B_constant;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B_constant;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      }
    }

    // Rcout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 
  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["m"]=m;
  result["logllk"] = logllk;
  result["n_count_index"] = n_count_index;
  // result["U_k"] = U_k;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  // result["logBinomial_all"] = logBinomial_all;
  return result;
}



// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_cvtesting(arma::vec &t, arma::mat &z, arma::vec &delta, 
                          arma::vec &t_tesing, arma::mat &z_tesing, arma::vec &delta_tesing, 
                           arma::vec &beta_t_init, arma::vec &beta_z_init, 
                           arma::mat &theta_init,
                           arma::mat &b_spline,
                           arma::mat &b_spline_testing,
                           arma::vec &unique_t,
                           double &tol, int &Mstop, 
                           double step_size_day,
                           double step_size_beta,
                           const bool &StopByInfo = false,
                           const bool &PenalizeGroup = false,
                           const bool &truncate = true,
                           const double &delta_bound = 1e-5,
                           const int &track = 100){
  //based on boosting_logit_expand_both_constant_timevarying_v2_noinv, just calculate the likelihood on the testing
  //data set.


  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int n_testing = z_tesing.n_rows;               // number of subjects

  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);

  arma::vec llk_diff_all = arma::zeros<arma::vec>(track);

  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, logllk_testing, select_index, AIC, df_all;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < P; ++j)
  {
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = tmp;
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j)
  {
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  cout <<"expanded data sample size :"<< N_tilde<<endl;

  double ll0,ll,ll2,ll_testing;
  double llk_testing_max = -datum::inf;
  //pseudo-outcome related variables:

  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();
    for (int j = 0; j < P; ++j)
    {
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      theta_Tilde.submat(1,j,knot-1,j)  = solve(Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)), tmp_tv, solve_opts::fast);
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();

    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }

    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    ll_testing = obj_function(t_tesing, z_tesing, delta_tesing, beta_t, theta, b_spline_testing, K, P, n_testing);

    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);
    logllk_testing.push_back(ll_testing);
    // Rcout << "diff : " << diff << endl;
    // if (diff<tol) {
    //   break;
    // } 

    if(llk_testing_max < ll_testing){
      llk_testing_max = ll_testing;
    }


    if(m >= (50 + track)) {
      for (int key = 0; key < track; ++key)
      {
        llk_diff_all(key) = logllk_testing[m-key-1];
      }
      double llk_max_track  = llk_diff_all.max();
      
      if(llk_max_track < llk_testing_max) {
        Rcout<<"llk_max_track: "<<llk_max_track<<endl;
        Rcout<<"llk_testing_max: "<<llk_testing_max<<endl;
        break;
      }
    }

  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["m"]=m;
  result["logllk"] = logllk;
  result["logllk_testing"] = logllk_testing;
  result["n_count_index"] = n_count_index;
  // result["U_k"] = U_k;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  // result["logBinomial_all"] = logBinomial_all;
  return result;
}




// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_interact(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                                                              arma::mat &theta_init,
                                                              arma::mat &b_spline,
                                                              arma::vec &unique_t,
                                                              double &tol, int &Mstop, 
                                                              double step_size_day,
                                                              double step_size_beta,
                                                              const int &interact_ind,
                                                              const bool &StopByInfo = false,
                                                              const bool &PenalizeGroup = false,
                                                              const bool &truncate = true,
                                                              const double &delta_bound = 1e-5){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  const int interact_ind_cpp = interact_ind;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_inv_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < interact_ind_cpp; ++j){
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    // cout<<"tmp row and col: "<<j*knot+1<<" "<<(knot*(j+1)-1)<<endl;
    Z_Tilde_square_inv_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = inv(tmp);
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j){
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  // mat Z_Tilde_square_inv = zeros<mat>(knot, knot*P);
  // if(StopByInfo){
  //   for (int j = 0; j < P; ++j)
  //   {
  //     mat tmp = Z_Tilde.cols(j*knot,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot,(knot*(j+1)-1));
  //     Z_Tilde_square_inv.submat(0, j*knot, knot-1, (knot*(j+1)-1)) = inv(tmp);
  //   }
  // }
  Rcout <<"expanded data sample size :"<< N_tilde<<endl;
  // // mat H_matrix_all = zeros<mat>(N_tilde,N_tilde*P);
  // if(StopByInfo){
  //   for (int j = 0; j < P; ++j)
  //   {
  //     mat tmp  = Z_Tilde.cols(j*knot,(knot*(j+1)-1))*Z_Tilde_square_inv.submat(0, j*knot, knot-1, (knot*(j+1)-1))*Z_Tilde.cols(j*knot,(knot*(j+1)-1)).t();
  //     // H_matrix_all.submat(0, j*N_tilde, N_tilde-1, (N_tilde*(j+1)-1)) = tmp;
  //   }
  // }





  double ll0,ll,ll2;

  //pseudo-outcome related variables:

  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();

    //non-interact part
    for (int j = 0; j < interact_ind_cpp; ++j)
    {
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      theta_Tilde.submat(1,j,knot-1,j)  = Z_Tilde_square_inv_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1))*tmp_tv;
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    for (int j = interact_ind_cpp; j < P; ++j)
    {
      Rcout<<"j:"<<j<<" P"<<P<<endl;
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      min_obj_tv(j) = INT_MAX;
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();

    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }


    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    if(StopByInfo){
      //update only the time-varying parts
      if(!select_constant){
        Z_Tilde_tmp = Z_Tilde.cols(min*knot+1,(knot*(min+1)-1));
        Z_Tilde_square_inv_tmp = Z_Tilde_square_inv_tv.submat(0, min*(knot-1), knot-2, ((knot-1)*(min+1)-1));
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp*Z_Tilde_square_inv_tmp*Z_Tilde_tmp.t();
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
          inv_Z_I_minus_B = Z_Tilde_square_inv_tmp*Z_I_minus_B;
          Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      } 
      //update only the constant parts
      else{
        Z_Tilde_tmp_constant = Z_Tilde.col(min*knot);
        Z_Tilde_square_inv_tmp_constant = Z_Tilde_square_inv_constant(min);
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp_constant*Z_Tilde_tmp_constant.t();
          H_matrix = Z_Tilde_square_inv_tmp_constant*H_matrix;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B_constant = (Z_Tilde_tmp_constant.t())*(Identity_mat-B_matrix);
          Z_I_minus_B_constant = Z_Tilde_square_inv_tmp_constant * Z_I_minus_B_constant;
          Z_inv_Z_I_minus_B_constant = Z_Tilde_tmp_constant*Z_I_minus_B_constant;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B_constant;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      }
    }

    // Rcout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 
  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["m"]=m;
  result["logllk"] = logllk;
  result["n_count_index"] = n_count_index;
  // result["U_k"] = U_k;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  // result["logBinomial_all"] = logBinomial_all;
  return result;
}



// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_interact_noinv(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                                                              arma::mat &theta_init,
                                                              arma::mat &b_spline,
                                                              arma::vec &unique_t,
                                                              double &tol, int &Mstop, 
                                                              double step_size_day,
                                                              double step_size_beta,
                                                              const int &interact_ind,
                                                              const bool &StopByInfo = false,
                                                              const bool &PenalizeGroup = false,
                                                              const bool &truncate = true,
                                                              const double &delta_bound = 1e-5){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  const int interact_ind_cpp = interact_ind;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all, BIC;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  Rcout <<"expanded data sample size :"<< N_tilde<<endl;


  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  Rcout <<"allocating Z_Tilde :"<< N_tilde<<endl;


  mat Z_Tilde_square_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < interact_ind_cpp; ++j){
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    // cout<<"tmp row and col: "<<j*knot+1<<" "<<(knot*(j+1)-1)<<endl;
    Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = tmp;
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j){
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  cube theta_all(P, knot, Mstop+1);  




  double ll0,ll,ll2;

  //pseudo-outcome related variables:

  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();

    //non-interact part
    for (int j = 0; j < interact_ind_cpp; ++j)
    {
      //constant part:
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      theta_Tilde.submat(1,j,knot-1,j)  = solve(Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)), tmp_tv, solve_opts::fast);
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    for (int j = interact_ind_cpp; j < P; ++j)
    {
      //constant part:
      // Rcout<<"j:"<<j<<" P"<<P<<endl;
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      min_obj_tv(j) = INT_MAX;
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();

    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }


    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    if(StopByInfo){
      //update only the time-varying parts
      if(!select_constant){
        Z_Tilde_tmp = Z_Tilde.cols(min*knot+1,(knot*(min+1)-1));
        Z_Tilde_square_inv_tmp = Z_Tilde_square_tv.submat(0, min*(knot-1), knot-2, ((knot-1)*(min+1)-1));
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t = solve(Z_Tilde_square_inv_tmp, Z_Tilde_tmp.t(), solve_opts::fast);
          mat H_matrix = Z_Tilde_tmp * Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t;
          B_matrix = W_matrix % H_matrix;
        }
        else{
          Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
          inv_Z_I_minus_B = solve(Z_Tilde_square_inv_tmp, Z_I_minus_B, solve_opts::fast);
          Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        BIC.push_back(-2*ll2 + log(n) * df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      } 
      //update only the constant parts
      else{
        Z_Tilde_tmp_constant = Z_Tilde.col(min*knot);
        Z_Tilde_square_inv_tmp_constant = Z_Tilde_square_inv_constant(min);
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp_constant*Z_Tilde_tmp_constant.t();
          H_matrix = Z_Tilde_square_inv_tmp_constant*H_matrix;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B_constant = (Z_Tilde_tmp_constant.t())*(Identity_mat-B_matrix);
          Z_I_minus_B_constant = Z_Tilde_square_inv_tmp_constant * Z_I_minus_B_constant;
          Z_inv_Z_I_minus_B_constant = Z_Tilde_tmp_constant*Z_I_minus_B_constant;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B_constant;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        BIC.push_back(-2*ll2 + log(n) * df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      }
    }

    theta_all.slice(m-1)    = theta;

    // Rcout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 

  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["theta_all"] = theta_all;
  result["m"]=m;
  result["logllk"] = logllk;
  // result["n_count_index"] = n_count_index;
  // result["delta_ik"] = delta_ik;
  // result["Y_Tilde"] = Y_Tilde;
  // result["U_k"] = U_k;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["BIC"] = BIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  return result;
}


void UpdateInteractSet(unordered_set <int> &interactTermsSet, int &min, 
                       mat &InteractMatrix,
                       unordered_set <int> &mainTermSet,
                       const bool &strong_h = false){

  int p = InteractMatrix.n_cols;

  if (min >= p){
    return;
  }

  if (strong_h == false){
    vec A_rowi = InteractMatrix.col(min);
    for (int i = 0; i < p; ++i){
      interactTermsSet.insert(A_rowi(i));
    }
  }
  else {
    //strong heredity situation:
    std::unordered_set<int>::iterator it = mainTermSet.begin();

    while(it != mainTermSet.end()) {
      if(*it < p & *it != min){
        // cout<< (*it) <<endl;
        int insert_value = InteractMatrix(*it, min);
        if(insert_value != -1){
          interactTermsSet.insert(insert_value);
        }
      }
      ++it;
    }

  }

}





// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_interact_noinv_mainfirst(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                                                              arma::mat &theta_init,
                                                              arma::mat &b_spline,
                                                              arma::vec &unique_t,
                                                              arma::mat &InteractMatrix,
                                                              double &tol, int &Mstop, 
                                                              double step_size_day,
                                                              double step_size_beta,
                                                              const int &interact_ind,
                                                              const bool &StopByInfo = false,
                                                              const bool &PenalizeGroup = false,
                                                              const bool &truncate = true,
                                                              const double &delta_bound = 1e-5,
                                                              const bool &strong_h = true,
                                                              const bool &pure_interaction = false){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  const int interact_ind_cpp = interact_ind;


  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    


  //initialize the sets to indicate for interaction terms selection:
  unordered_set <int> mainTermSet;
  unordered_set <int> interactTermsSet;

  for (int i = 0; i < interact_ind_cpp; ++i)
  {
    interactTermsSet.insert(i);
  }

  if(pure_interaction == true){
    for (int i = 0; i < P; ++i)
      {
        interactTermsSet.insert(i);
      }
  }


  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all, BIC;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);

  //prepare for iteration loop:
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < interact_ind_cpp; ++j){
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    // cout<<"tmp row and col: "<<j*knot+1<<" "<<(knot*(j+1)-1)<<endl;
    Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = tmp;
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j){
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  cout <<"expanded data sample size :"<< N_tilde<<endl;

  cube theta_all(P, knot, Mstop+1);  
  mat beta_t_all(K, Mstop+1);



  double ll0,ll,ll2;

  //start iteration
  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();

    //non-interact part
    for (int j = 0; j < interact_ind_cpp; ++j)
    {
      //constant part:
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      theta_Tilde.submat(1,j,knot-1,j)  = solve(Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)), tmp_tv, solve_opts::fast);
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    for (int j = interact_ind_cpp; j < P; ++j)
    {
      bool tmp = (interactTermsSet.find(j) == interactTermsSet.end());
      if (interactTermsSet.find(j) == interactTermsSet.end()){
        min_obj_constant(j) = INT_MAX;
        min_obj_tv(j) = INT_MAX;
        continue;
      }

      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      min_obj_tv(j) = INT_MAX;
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();


    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }
    //update the interactTermsSet:
    if (min < interact_ind){
      mainTermSet.insert(min);
    }
    UpdateInteractSet(interactTermsSet, min, InteractMatrix, mainTermSet, strong_h);


    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    if(StopByInfo){
      //update only the time-varying parts
      if(!select_constant){
        Z_Tilde_tmp = Z_Tilde.cols(min*knot+1,(knot*(min+1)-1));
        Z_Tilde_square_inv_tmp = Z_Tilde_square_tv.submat(0, min*(knot-1), knot-2, ((knot-1)*(min+1)-1));
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t = solve(Z_Tilde_square_inv_tmp, Z_Tilde_tmp.t(), solve_opts::fast);
          mat H_matrix = Z_Tilde_tmp * Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t;
          B_matrix = W_matrix % H_matrix;
        }
        else{
          Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
          inv_Z_I_minus_B = solve(Z_Tilde_square_inv_tmp, Z_I_minus_B, solve_opts::fast);
          Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        BIC.push_back(-2*ll2 + log(n) * df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      } 
      //update only the constant parts
      else{
        Z_Tilde_tmp_constant = Z_Tilde.col(min*knot);
        Z_Tilde_square_inv_tmp_constant = Z_Tilde_square_inv_constant(min);
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp_constant*Z_Tilde_tmp_constant.t();
          H_matrix = Z_Tilde_square_inv_tmp_constant*H_matrix;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B_constant = (Z_Tilde_tmp_constant.t())*(Identity_mat-B_matrix);
          Z_I_minus_B_constant = Z_Tilde_square_inv_tmp_constant * Z_I_minus_B_constant;
          Z_inv_Z_I_minus_B_constant = Z_Tilde_tmp_constant*Z_I_minus_B_constant;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B_constant;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        BIC.push_back(-2*ll2 + log(n) * df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      }
    }
    beta_t_all.col(m-1) = beta_t;
    theta_all.slice(m-1)    = theta;

    // Rcout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 

  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["beta_t_all"]=beta_t_all;
  result["theta_all"] = theta_all;
  result["m"]=m;
  result["logllk"] = logllk;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["BIC"] = BIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  result["mainTermSet"] = mainTermSet;
  result["interactTermsSet"] = interactTermsSet;
  return result;
}






// [[Rcpp::export]]
List z_expand_timevarying(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                           arma::mat &theta_init,
                           arma::mat &b_spline,
                           arma::vec &unique_t){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  arma::mat theta = theta_init;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }

  Rcout<<"N_tilde: "<<N_tilde<<endl;

  arma::mat theta_Tilde = zeros<mat>(knot, P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec time_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }


  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  Rcout<<"n_count: "<<n_count<<endl;

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i){
    for (int k = 1; k <= max(t); ++k){
      if (t(i) >= k){
        time_Tilde(n_count) = k;
        n_count++;
      }
    }
  }

  List result;

  result["Z_Tilde"] = Z_Tilde;
  result["Y_Tilde"] = Y_Tilde;
  result["time_Tilde"] = time_Tilde;
  return result;


}


// // [[Rcpp::export]]
// List z_expand_timevarying_spmatrix(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
//                            arma::mat &theta_init,
//                            arma::mat &b_spline,
//                            arma::vec &unique_t){
//   int P = z.n_cols;               // number of covariates
//   int n = z.n_rows;               // number of subjects
//   int K = max(t);             // maximum time indicator
//   int knot = b_spline.n_cols;
//   double diff = datum::inf;     
//   arma::mat theta = theta_init;

//   vec beta_t = beta_t_init;
//   vec beta_z = beta_z_init;    

//   int N_tilde = 0, n_count=0;
//   //pseudo-outcome related variables:
//   for (int i = 0; i < n; ++i)
//   {
//     for (int k = 0; k < t(i); k++)
//     {
//       N_tilde += 1;
//     }
//   }

//   Rcout<<"N_tilde: "<<N_tilde<<endl;

//   arma::mat theta_Tilde = zeros<mat>(knot, P);

//   arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
//   arma::vec time_Tilde = arma::zeros<arma::vec>(N_tilde);
//   arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
//   arma::sp_mat Z_Tilde(N_tilde,P*knot);
//   vec eta = zeros<vec>(N_tilde);
//   vec U     = zeros<vec>(N_tilde);
//   vec U_mgamma = zeros<vec>(N_tilde);
//   vec p_hat = zeros<vec>(N_tilde);


//   vec n_count_index = zeros<vec>(K);
//   vec U_k = zeros<vec>(K);  

//   sp_mat Identity_mat(N_tilde,N_tilde);
//   for (int i = 0; i < N_tilde; ++i)
//   {
//     Identity_mat(i,i) = 1;
//   }


//   vec delta_ik  = zeros<mat>(N_tilde);
//   vec delta_ik_mgamma = zeros<mat>(N_tilde);
//   n_count = 0;
//   for (int i = 0; i < n; ++i)
//   {
//     for (int k = 0; k < t(i); k++)
//     {
//       if (t(i) == (k+1)){
//         if(delta(i)==1){
//           delta_ik(n_count) = 1;
//         }
//       }
//       n_count++;
//       n_count_index(k) +=1;
//     }
//   }

//   Rcout<<"n_count: "<<n_count<<endl;

//   n_count = 0;
//   for (int i = 0; i < n; ++i)
//   {
//     for (int k = 1; k <= t(i); k++)
//     {
//       if (t(i) != k){
//         Y_Tilde(n_count) = 0;
//       }
//       if (t(i) == k){
//         Y_Tilde(n_count) = delta(i);
//       }
//       n_count++;
//     }
//   }

//   n_count = 0;
//   for (int i = 0; i < n; ++i)
//   {
//     for (int k = 0; k < t(i); k++)
//     {
//       arma::rowvec z_row_i = z.row(i);
//       arma::rowvec b_spline_row_k = b_spline.row(k);
//       for (int p = 0; p < P; ++p) {
//         for (int j = 0; j < knot; ++j) {
//           Z_Tilde(n_count, p*knot + j) = z_row_i(p) * b_spline_row_k(j);
//         }
//       }
//       n_count++;
//     }
//   }

//   n_count = 0;
//   for (int i = 0; i < n; ++i){
//     for (int k = 1; k <= max(t); ++k){
//       if (t(i) >= k){
//         time_Tilde(n_count) = k;
//         n_count++;
//       }
//     }
//   }

//   List result;

//   result["Z_Tilde"] = Z_Tilde;
//   result["Y_Tilde"] = Y_Tilde;
//   result["time_Tilde"] = time_Tilde;
//   return result;


// }





// [[Rcpp::export]]
List IC_calculate(arma::vec &t, arma::mat &z, arma::vec &delta, arma::mat &b_spline, arma::vec &beta_t, arma::mat &theta,
                  int &max_t, int &P, int &n, int &knot,
                  double &penalty){
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
      // vec score_theta_p = score_theta + (penalty*S_matrix)*vectorise(theta.t(), 0);
      vec score_theta_p = score_theta;

      grad_tmp          = score_theta;
      grad_p_tmp        = score_theta_p;

      info_t(s-1)      += lambda*(1-lambda);

      mat info_v11      = lambda*(1-lambda)*(kron(zzT, B_sp_tmp*B_sp_tmp.t()));
      // mat info_v11_p    = info_v11 + penalty*S_matrix;
      mat info_v11_p    = info_v11;

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

  return List::create(_["df_AIC"] = df_AIC,
                      _["df_TIC"] = df_TIC,
                      _["df_GIC"] = df_GIC);
}




// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_interact_noinv_mainfirst_newIC(arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                                                              arma::mat &theta_init,
                                                              arma::mat &b_spline,
                                                              arma::vec &unique_t,
                                                              arma::mat &InteractMatrix,
                                                              double &tol, int &Mstop, 
                                                              double step_size_day,
                                                              double step_size_beta,
                                                              const int &interact_ind,
                                                              const bool &StopByInfo = false,
                                                              const bool &PenalizeGroup = false,
                                                              const bool &truncate = true,
                                                              const double &delta_bound = 1e-5,
                                                              const bool &strong_h = true){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  const int interact_ind_cpp = interact_ind;


  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    


  //initialize the sets to indicate for interaction terms selection:
  unordered_set <int> mainTermSet;
  unordered_set <int> interactTermsSet;

  for (int i = 0; i < interact_ind_cpp; ++i)
  {
    interactTermsSet.insert(i);
  }


  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all, BIC, df_AIC_all, df_TIC_all, df_GIC_all;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  double df_AIC, df_TIC, df_GIC;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);

  //prepare for iteration loop:
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < interact_ind_cpp; ++j){
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    // cout<<"tmp row and col: "<<j*knot+1<<" "<<(knot*(j+1)-1)<<endl;
    Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = tmp;
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j){
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  cout <<"expanded data sample size :"<< N_tilde<<endl;

  cube theta_all(P, knot, Mstop+1);  
  mat beta_t_all(K, Mstop+1);



  double ll0,ll,ll2;

  //start iteration
  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();

    //non-interact part
    for (int j = 0; j < interact_ind_cpp; ++j)
    {
      //constant part:
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
      theta_Tilde.submat(1,j,knot-1,j)  = solve(Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)), tmp_tv, solve_opts::fast);
      min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
    }

    for (int j = interact_ind_cpp; j < P; ++j)
    {
      bool tmp = (interactTermsSet.find(j) == interactTermsSet.end());
      if (interactTermsSet.find(j) == interactTermsSet.end()){
        min_obj_constant(j) = INT_MAX;
        min_obj_tv(j) = INT_MAX;
        continue;
      }

      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      min_obj_tv(j) = INT_MAX;
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();


    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }
    //update the interactTermsSet:
    if (min < interact_ind){
      mainTermSet.insert(min);
    }
    UpdateInteractSet(interactTermsSet, min, InteractMatrix, mainTermSet, strong_h);


    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    if(StopByInfo){
      //update only the time-varying parts
      if(!select_constant){
        Z_Tilde_tmp = Z_Tilde.cols(min*knot+1,(knot*(min+1)-1));
        Z_Tilde_square_inv_tmp = Z_Tilde_square_tv.submat(0, min*(knot-1), knot-2, ((knot-1)*(min+1)-1));
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t = solve(Z_Tilde_square_inv_tmp, Z_Tilde_tmp.t(), solve_opts::fast);
          mat H_matrix = Z_Tilde_tmp * Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t;
          B_matrix = W_matrix % H_matrix;
        }
        else{
          Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
          inv_Z_I_minus_B = solve(Z_Tilde_square_inv_tmp, Z_I_minus_B, solve_opts::fast);
          Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        BIC.push_back(-2*ll2 + log(n) * df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      } 
      //update only the constant parts
      else{
        Z_Tilde_tmp_constant = Z_Tilde.col(min*knot);
        Z_Tilde_square_inv_tmp_constant = Z_Tilde_square_inv_constant(min);
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp_constant*Z_Tilde_tmp_constant.t();
          H_matrix = Z_Tilde_square_inv_tmp_constant*H_matrix;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B_constant = (Z_Tilde_tmp_constant.t())*(Identity_mat-B_matrix);
          Z_I_minus_B_constant = Z_Tilde_square_inv_tmp_constant * Z_I_minus_B_constant;
          Z_inv_Z_I_minus_B_constant = Z_Tilde_tmp_constant*Z_I_minus_B_constant;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B_constant;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        BIC.push_back(-2*ll2 + log(n) * df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      }


      List Infocrit;

      double penalty = 0;
      //update only the time-varying parts
      Infocrit = IC_calculate(t, z, delta, b_spline, beta_t, theta, 
                              K, P, n, knot, penalty);

      df_AIC = Infocrit["df_AIC"];
      df_TIC = Infocrit["df_TIC"];
      df_GIC = Infocrit["df_GIC"];

      //AIC:
      df_AIC_all.push_back(df_AIC);
      //TIC:
      df_TIC_all.push_back(df_TIC);
      //GIC:
      df_GIC_all.push_back(df_GIC);
    }

    beta_t_all.col(m-1) = beta_t;
    theta_all.slice(m-1)    = theta;

    // Rcout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 

  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["beta_t_all"]=beta_t_all;
  result["theta_all"] = theta_all;
  result["m"]=m;
  result["logllk"] = logllk;
  result["df_all"] = df_all;
  result["df_AIC_all"] = df_AIC_all;
  result["df_TIC_all"] = df_TIC_all;
  result["df_GIC_all"] = df_GIC_all;
  result["AIC"] = AIC;
  result["BIC"] = BIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  result["mainTermSet"] = mainTermSet;
  result["interactTermsSet"] = interactTermsSet;
  return result;
}






// [[Rcpp::export]]
List boosting_logit_expand_both_constant_timevarying_v2_noinv_TimevaryingHierarchy(
                           arma::vec &t, arma::mat &z, arma::vec &delta, arma::vec &beta_t_init, arma::vec &beta_z_init, 
                           arma::mat &theta_init,
                           arma::mat &b_spline,
                           arma::vec &unique_t,
                           double &tol, int &Mstop, 
                           double step_size_day,
                           double step_size_beta,
                           const bool &StopByInfo = false,
                           const bool &PenalizeGroup = false,
                           const bool &truncate = true,
                           const double &delta_bound = 1e-5){
  int P = z.n_cols;               // number of covariates
  int n = z.n_rows;               // number of subjects
  int K = max(t);             // maximum time indicator
  int knot = b_spline.n_cols;
  double diff = datum::inf;     
  int time_length = unique_t.n_elem;
  arma::mat theta = theta_init;

  vec beta_t = beta_t_init;
  vec beta_z = beta_z_init;    

  double group_penalizer;
  if (PenalizeGroup){
    group_penalizer = sqrt(knot-1);
  }
  else{
    group_penalizer = 1;
  }

  int N_tilde = 0, n_count=0;
  //pseudo-outcome related variables:
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      N_tilde += 1;
    }
  }


  arma::mat A_inv = arma::zeros<arma::mat>(K,K);     //used to record the information matrix w.r.t. time(gamma)
  arma::mat theta_Tilde = zeros<mat>(knot, P);
  // arma::vec min_obj = arma::zeros<arma::vec>(P);
  arma::vec min_obj_constant = arma::zeros<arma::vec>(P);
  arma::vec min_obj_tv = arma::zeros<arma::vec>(P);

  arma::vec Y_Tilde = arma::zeros<arma::vec>(N_tilde);
  arma::vec gamma_Tilde = arma::zeros<arma::vec>(K);
  arma::mat Z_Tilde = arma::zeros<arma::mat>(N_tilde,P*knot);
  vec eta = zeros<vec>(N_tilde);
  vec U     = zeros<vec>(N_tilde);
  vec U_mgamma = zeros<vec>(N_tilde);
  vec p_hat = zeros<vec>(N_tilde);
  // mat H_matrix = zeros<mat>(N_tilde,N_tilde);
  mat W_matrix = zeros<mat>(N_tilde,N_tilde);
  mat B_matrix = zeros<mat>(N_tilde,N_tilde);

  mat Z_I_minus_B = zeros<mat>(N_tilde, N_tilde);
  mat Z_Tilde_tmp = zeros<mat>(N_tilde, K-1);
  mat Z_Tilde_square_inv_tmp = zeros<mat>(K-1, K-1);
  mat inv_Z_I_minus_B = zeros<mat>(K, N_tilde);
  mat Z_inv_Z_I_minus_B = zeros<mat>(N_tilde,N_tilde);

  vec Z_Tilde_tmp_constant = zeros<vec>(N_tilde);
  double Z_Tilde_square_inv_tmp_constant;
  mat Z_I_minus_B_constant = zeros<mat>(1,N_tilde);
  mat Z_inv_Z_I_minus_B_constant = zeros<mat>(N_tilde,N_tilde);


  vec n_count_index = zeros<vec>(K);
  vec U_k = zeros<vec>(K);  

  mat Identity_mat = zeros<mat>(N_tilde,N_tilde);
  for (int i = 0; i < N_tilde; ++i)
  {
    Identity_mat(i,i) = 1;
  }

  unordered_set <int> TVTermSet;

  int min, min_tv, min_constant;
  NumericVector logllk, select_index, AIC, df_all;
  NumericVector logBinomial_all;
  NumericVector select_index_constant, select_index_timevarying;

  vec delta_ik  = zeros<mat>(N_tilde);
  vec delta_ik_mgamma = zeros<mat>(N_tilde);
  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      if (t(i) == (k+1)){
        if(delta(i)==1){
          delta_ik(n_count) = 1;
        }
      }
      n_count++;
      n_count_index(k) +=1;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 1; k <= t(i); k++)
    {
      if (t(i) != k){
        Y_Tilde(n_count) = 0;
      }
      if (t(i) == k){
        Y_Tilde(n_count) = delta(i);
      }
      n_count++;
    }
  }

  n_count = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int k = 0; k < t(i); k++)
    {
      Z_Tilde.row(n_count) = kron(z.row(i), b_spline.row(k));
      n_count++;
    }
  }

  mat Z_Tilde_square_tv = zeros<mat>(knot-1, (knot-1)*P);
  for (int j = 0; j < P; ++j)
  {
    mat tmp = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*Z_Tilde.cols(j*knot+1,(knot*(j+1)-1));
    Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)) = tmp;
  }

  vec Z_Tilde_square_inv_constant = zeros<vec>(P);
  for (int j = 0; j < P; ++j)
  {
    vec tmp = Z_Tilde.col(j*knot).t()*Z_Tilde.col(j*knot);
    Z_Tilde_square_inv_constant(j) = 1/tmp(0);
  }

  cout <<"expanded data sample size :"<< N_tilde<<endl;


  double ll0,ll,ll2;

  //pseudo-outcome related variables:

  List result;
  arma::mat beta_v;
  bool select_constant = false;
  int m = 0;                      // number of iteration
  while (m<=Mstop){
    m = m+1;
    ll = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);   //calculate the score function w.r.t. beta
    if (m==1){
      ll0 = ll;
    }
    ll2 = ll;              //used to calculate the likelihood difference in order to determine when to stop

    vec zbeta_tmp;
    n_count = 0;
    beta_v  = theta*b_spline.t();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        zbeta_tmp = z.row(i)*beta_v.col(k);
        eta(n_count) = beta_t(k) + zbeta_tmp(0); 
        // delta_ik_mgamma(n_count) = delta_ik(n_count) - beta_t(k);
        n_count++;
      }
    }

    p_hat = exp(eta)/(1+exp(eta));
    U = delta_ik -  p_hat;

    //Outerloop: update gamma:
    n_count = 0;
    U_k.zeros();
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_k(k) += U(n_count);
        n_count += 1;
      }
    }
    U_k = U_k / n_count_index;

    n_count = 0;
    for (int i = 0; i < n; ++i)
    {
      for (int k = 0; k < t(i); k++)
      { 
        U_mgamma(n_count) = U(n_count) - U_k(k);
        n_count += 1;
      }
    }

    beta_t += step_size_day*U_k;

    //Inner loop: Update beta:

    //Time_varying part:
    theta_Tilde.zeros();
    for (int j = 0; j < P; ++j)
    {
      vec tmp_constant = Z_Tilde.col(j*knot).t()*U_mgamma;
      double tmp_constant_value = tmp_constant(0);
      theta_Tilde(0,j) = Z_Tilde_square_inv_constant(j)*tmp_constant_value;
      min_obj_constant(j) = norm(U_mgamma - Z_Tilde.col(j*knot)*theta_Tilde(0,j),2);

      //Time varying part:
      if (TVTermSet.find(j) == TVTermSet.end()){
        min_obj_tv(j) = INT_MAX;
      } else {
        vec tmp_tv = Z_Tilde.cols(j*knot+1,(knot*(j+1)-1)).t()*U_mgamma;
        theta_Tilde.submat(1,j,knot-1,j)  = solve(Z_Tilde_square_tv.submat(0, j*(knot-1), knot-2, ((knot-1)*(j+1)-1)), tmp_tv, solve_opts::fast);
        min_obj_tv(j) = norm(U_mgamma - Z_Tilde.cols(j*knot+1,(knot*(j+1)-1))*theta_Tilde.submat(1,j,knot-1,j)/group_penalizer,2);
      }
    }

    min_constant = abs(min_obj_constant).index_min();
    min_tv = abs(min_obj_tv).index_min();

    if (min_obj_constant(min_constant) < min_obj_tv(min_tv))
    {
      min = min_constant;
      theta(min,0) += step_size_beta * theta_Tilde(0,min);
      select_index_constant.push_back(min);
      select_index_timevarying.push_back(-1);
      select_constant = true;
    }
    else{
      min = min_tv;
      theta.submat(min,1,min,knot-1) += step_size_beta * theta_Tilde.submat(1,min,knot-1,min).t();
      select_index_timevarying.push_back(min);
      select_index_constant.push_back(-1);
      select_constant = false;
    }

    //update the T  VTermSet:
    TVTermSet.insert(min);

    ll2 = obj_function(t, z, delta, beta_t, theta, b_spline, K, P, n);
    diff = abs(ll2-ll)/abs(ll2-ll0);
    logllk.push_back(ll2);

    if(StopByInfo){
      //update only the time-varying parts
      if(!select_constant){
        Z_Tilde_tmp = Z_Tilde.cols(min*knot+1,(knot*(min+1)-1));
        Z_Tilde_square_inv_tmp = Z_Tilde_square_tv.submat(0, min*(knot-1), knot-2, ((knot-1)*(min+1)-1));
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t = solve(Z_Tilde_square_inv_tmp, Z_Tilde_tmp.t(), solve_opts::fast);
          mat H_matrix = Z_Tilde_tmp * Z_Tilde_square_inv_tmp_Z_Tilde_tmp_t;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B = (Z_Tilde_tmp.t())*(Identity_mat-B_matrix);
          inv_Z_I_minus_B = solve(Z_Tilde_square_inv_tmp, Z_I_minus_B, solve_opts::fast);
          Z_inv_Z_I_minus_B = Z_Tilde_tmp*inv_Z_I_minus_B;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      } 
      //update only the constant parts
      else{
        Z_Tilde_tmp_constant = Z_Tilde.col(min*knot);
        Z_Tilde_square_inv_tmp_constant = Z_Tilde_square_inv_constant(min);
        vec W_tmp = zeros<vec>(N_tilde);
        W_tmp     = step_size_beta*p_hat%(ones<vec>(N_tilde)-p_hat);
        W_matrix  = repmat(W_tmp,1,N_tilde);
        if(m==1){
          mat H_matrix = Z_Tilde_tmp_constant*Z_Tilde_tmp_constant.t();
          H_matrix = Z_Tilde_square_inv_tmp_constant*H_matrix;
          B_matrix = W_matrix%H_matrix;
        }
        else{
          Z_I_minus_B_constant = (Z_Tilde_tmp_constant.t())*(Identity_mat-B_matrix);
          Z_I_minus_B_constant = Z_Tilde_square_inv_tmp_constant * Z_I_minus_B_constant;
          Z_inv_Z_I_minus_B_constant = Z_Tilde_tmp_constant*Z_I_minus_B_constant;
          B_matrix = B_matrix + W_matrix%Z_inv_Z_I_minus_B_constant;
        }
        double df = trace(B_matrix);
        df += beta_t.n_elem;
        AIC.push_back(-2*ll2+2*df);
        df_all.push_back(df);

        // double logBinomial = obj_function_biomial(N_tilde, B_matrix, Y_Tilde, delta_bound);
        // logBinomial_all.push_back(logBinomial);
      }
    }

    // Rcout << "diff : " << diff << endl;
    if (diff<tol) {
      break;
    } 
  }

  result["ll"] = ll;
  result["beta_t"]=beta_t;
  result["theta"]=theta;
  result["m"]=m;
  result["logllk"] = logllk;
  result["n_count_index"] = n_count_index;
  // result["U_k"] = U_k;
  result["df_all"] = df_all;
  result["AIC"] = AIC;
  result["select_index_constant"] = select_index_constant;
  result["select_index_timevarying"] = select_index_timevarying;
  result["Z_Tilde"] = Z_Tilde;
  result["min_obj_constant"] = min_obj_constant;
  result["min_obj_tv"] = min_obj_tv;
  // result["logBinomial_all"] = logBinomial_all;
  return result;
}
