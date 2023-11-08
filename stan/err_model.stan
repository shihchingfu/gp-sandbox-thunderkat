functions {
  // c.f. Rasmussen & Williams (2006), Algorithm 2.1
  vector gp_pred_rng(array[] real x2,
                     vector y1,
                     array[] real x1,
                     real eta,
                     real ell,
                     vector sigma,
                     real jitter) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N1, N1] K;
      K = eta * gp_exp_quad_cov(x1, 1.0, ell);
      for (n in 1:N1)
        K[n, n] = K[n,n] + square(sigma[n]);
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = eta * gp_exp_quad_cov(x1, x2, 1.0, ell);
      f2_mu = (k_x1_x2' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = eta * gp_exp_quad_cov(x2, 1.0, ell) - v_pred' * v_pred;

      f2 = multi_normal_rng(f2_mu, add_diag(cov_f2, rep_vector(jitter, N2)));
    }
    return f2;
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
  matrix[N, N] K = eta * gp_exp_quad_cov(x, 1.0, ell);
  matrix[N, N] L = cholesky_decompose(add_diag(K, sigma^2));

  ell ~ inv_gamma(5, 5);
  eta ~ std_normal();
  sigma ~ normal(y_stderr, sd(y_stderr)); // use observed error estimates

  y ~ multi_normal_cholesky(mu, L);
}
generated quantities {
  vector[N_star] f_star = gp_pred_rng(x_star, y, x, eta, ell, sigma, 1e-9);
}
