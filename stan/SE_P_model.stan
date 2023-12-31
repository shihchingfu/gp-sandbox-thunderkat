functions {
  vector gp_pred_rng(array[] real x_star,
                     vector y,
                     array[] real x,
                     real eta_SE,
                     real eta_P,
                     real ell_SE,
                     real ell_P,
                     real T,
                     vector sigma,
                     real jitter) {
    int N = rows(y);
    int N_star = size(x_star);
    vector[N_star] f_star;
    {
      matrix[N, N] K;
      matrix[N, N] L;
      vector[N] alpha;
      matrix[N, N_star] k_x_xstar;
      matrix[N, N_star] v;
      vector[N_star] fstar_mu;
      matrix[N_star, N_star] fstar_cov;

      K = eta_SE * gp_exp_quad_cov(x, 1.0, ell_SE) +
          eta_P  * gp_periodic_cov(x, 1.0, ell_P, T);

      for (n in 1:N)
        K[n, n] = K[n,n] + square(sigma[n]);

      L = cholesky_decompose(K);
      alpha = mdivide_left_tri_low(L, y);
      alpha = mdivide_right_tri_low(alpha', L)';

      k_x_xstar = eta_SE * gp_exp_quad_cov(x, x_star, 1.0, ell_SE) +
                  eta_P *  gp_periodic_cov(x, x_star, 1.0, ell_P, T);
      fstar_mu = k_x_xstar' * alpha;

      v = mdivide_left_tri_low(L, k_x_xstar);
      fstar_cov = ( eta_SE * gp_exp_quad_cov(x_star, 1.0, ell_SE) +
                    eta_P *  gp_periodic_cov(x_star, 1.0, ell_P, T) ) - v' * v;

      f_star = multi_normal_rng(fstar_mu, add_diag(fstar_cov, rep_vector(jitter, N_star)));
    }
    return f_star;
  }
}
data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y;
  vector[N] y_stderr;
  int<lower=1> N_star;
  array[N_star] real x_star;
  real min_xgap;
  real range_x;
  real T_lb;
  real T_ub;
}
transformed data {
  vector[N] mu = rep_vector(0, N);
}
parameters {
  real<lower=min_xgap> ell_SE;
  real<lower=min_xgap> ell_P;
  real<lower=0> eta_SE;
  real<lower=0> eta_P;
  real<lower=T_lb,upper=T_ub> T;
  //vector<lower=0>[N] sigma; // heteroskedastic
}
model {
  matrix[N, N] K = eta_SE *  gp_exp_quad_cov(x, 1.0, ell_SE) +
                   eta_P  * gp_periodic_cov(x, 1.0, ell_P, T);
  matrix[N, N] L = cholesky_decompose(add_diag(K, y_stderr^2));

  ell_SE ~ inv_gamma(3, 0.5*range_x);
  ell_P ~ inv_gamma(3, 0.5*range_x);
  eta_SE ~ std_normal();
  eta_P ~ std_normal();
  //sigma ~ normal(y_stderr, sd(y_stderr)); // use observed error estimates

  y ~ multi_normal_cholesky(mu, L);
}
generated quantities {
  vector[N_star] f_star = gp_pred_rng(x_star, y, x, eta_SE, eta_P, ell_SE, ell_P, T, y_stderr, 1e-9);
}
