functions {
  // See Rasmussen & Williams (2006), Equation 2.38
  vector gp_pred_rng(array[] real x_star,
                     vector y,
                     array[] real x,
                     real mu,
                     real eta,
                     real ell,
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
      vector[N_star] mu_star;

      K = gp_exp_quad_cov(x, eta, ell);
      for (n in 1:N)
        K[n, n] = K[n,n] + square(sigma[n]);

      L = cholesky_decompose(K);
      alpha = mdivide_left_tri_low(L, y - mu);
      alpha = mdivide_right_tri_low(alpha', L)';

      k_x_xstar = gp_exp_quad_cov(x, x_star, eta, ell);

      mu_star = rep_vector(mu, N_star);
      fstar_mu = mu_star + k_x_xstar' * alpha;

      v = mdivide_left_tri_low(L, k_x_xstar);
      fstar_cov = gp_exp_quad_cov(x_star, eta, ell) - v' * v;

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
  real const_mean_value;
}
transformed data {
  real mu = const_mean_value;
}
parameters {
  real<lower=0> ell;
  real<lower=0> eta;
  vector<lower=0>[N] sigma;
}
model {
  matrix[N, N] K = gp_exp_quad_cov(x, eta, ell);
  matrix[N, N] L = cholesky_decompose(add_diag(K, sigma^2));

  ell ~ inv_gamma(5, 5);
  eta ~ std_normal();
  sigma ~ normal(y_stderr, sd(y_stderr));

  y ~ multi_normal_cholesky(rep_vector(mu, N), L);
}
generated quantities {
  vector[N_star] f_star = gp_pred_rng(x_star, y, x, mu, eta, ell, sigma, 1e-9);
}
