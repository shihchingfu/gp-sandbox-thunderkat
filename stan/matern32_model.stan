functions {
  // c.f. Rasmussen & Williams (2006), Algorithm 2.1
  vector gp_pred_rng(array[] real x_star,
                     vector y,
                     array[] real x,
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

      K = gp_matern32_cov(x, eta, ell);
      for (n in 1:N)
        K[n, n] = K[n,n] + square(sigma[n]);

      L = cholesky_decompose(K);
      alpha = mdivide_left_tri_low(L, y);
      alpha = mdivide_right_tri_low(alpha', L)';

      k_x_xstar = gp_matern32_cov(x, x_star, eta, ell);
      fstar_mu = k_x_xstar' * alpha;

      v = mdivide_left_tri_low(L, k_x_xstar);
      fstar_cov = gp_matern32_cov(x_star, eta, ell) - v' * v;

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
}
transformed data {
  vector[N] mu = rep_vector(0, N);
}
parameters {
  real<lower=0> ell;
  real<lower=0> eta;
  vector<lower=0>[N] sigma; // heteroskedastic
}
model {
  matrix[N, N] K = gp_matern32_cov(x, eta, ell);
  matrix[N, N] L = cholesky_decompose(add_diag(K, sigma^2));

  ell ~ inv_gamma(5, 5);
  eta ~ std_normal();
  sigma ~ normal(y_stderr, sd(y_stderr)); // use observed error estimates

  y ~ multi_normal_cholesky(mu, L);
}
generated quantities {
  vector[N_star] f_star = gp_pred_rng(x_star, y, x, eta, ell, sigma, 1e-9);
}
