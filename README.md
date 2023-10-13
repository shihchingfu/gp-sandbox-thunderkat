# Effect of compound kernels on GP fit

This work examines the incremental affect of adding more terms to the GP kernel expression, namely Squared Exponential (SE), Matern $\frac{3}{2}$ (M32), and Periodic kernels (P).

There is the possibility of degeneracy between the various hyperparameters of the chosen kernels, especially the overlapping influences on input and output scale parameters ($\ell$ and $\eta$).

Four light curves taken from the ThunderKAT survey have been selected as test cases:
- `80_ra271.352_dec-29.642_MAXIJ1803`: evenly spaced, no obvious outliers, weak correlated noise.
- `1817_ra284.905_dec-8.658_J1858TraPDB`: non-stationary series.
- `428_ra236.530_dec-46.922_4U1543TraPDB`: evenly spaced, no obvious outliers, moderate correlated noise.
- `502_ra236.310_dec-47.644_4U1543TraPDB`: bi-modal distribution in observational errors.

For a light curve each of the following models is fitted:

1. GP using SE kernel
2. GP using M32 kernel
3. GP using SE + M32 kernel
4. Sum of SE GP and M32 GP
5. GP using SE $\times$ M32 kernel


