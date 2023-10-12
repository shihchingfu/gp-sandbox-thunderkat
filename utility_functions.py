"""Module containing utility functions for fitting and plotting results of GP fitting"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
from pymc.gp.util import plot_gp_dist
import arviz as az
import graphviz as gv

N_DRAWS = 1000
N_TUNE = 1000
N_PPC = 200 # No. prior predictive samples
N_NEW = 200 # No. posterior predictive samples

def plot_lc(path_to_csv):
    """Plot light curve from raw CSV data including 1 sigma error bars and overall mean."""
    
    this_lc = pd.read_csv(path_to_csv)
    this_x = this_lc['mjd']
    this_y = this_lc['f_peak']
    this_yerr = this_lc['f_peak_err']
    mean_y = np.nanmean(this_y)

    plt.figure(figsize=(12, 5))
    plt.plot(this_x, this_y, "_b", ms=8, alpha=1, label="Observed data")
    plt.axhline(y=mean_y, c='blue', ls=':')
    plt.errorbar(x=this_x, y=this_y, yerr=this_yerr,
                 fmt="none", ecolor="k", elinewidth=1, capsize=3,
                 label=r"1 $\sigma$")
    plt.title(f"{path_to_csv.stem} (N={len(this_y)})")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()

def plot_postpred_samples(path_to_trace, path_to_csv):
    """Plot posterior predicted samples and original data light curve"""

    this_lc = pd.read_csv(path_to_csv)
    this_trace = az.from_netcdf(filename=path_to_trace)

    this_x = this_lc['mjd']
    this_y = this_lc['f_peak']
    this_yerr = this_lc['f_peak_err']

    this_xnew = np.linspace(
        start=np.floor(this_x.min()),
        stop=np.ceil(this_x.max()),
        num = len(this_trace.posterior_predictive.f_star_dim_2)
    ).reshape(-1,1)

    y_postpred = az.extract(this_trace, "posterior_predictive", var_names=["f_star"])
    y_postpred_median = y_postpred.median(dim="sample")

    fig = plt.figure(figsize=(12, 5))
    axes = fig.gca()
    plt.plot(this_x, this_y, "_b", ms=8, alpha=1, label="Observed data")
    plt.errorbar(x=this_x, y=this_y, yerr=this_yerr,
                 fmt="none", ecolor="k", elinewidth=1, capsize=3,
                 label=r"Observed 1$\sigma$")
    plot_gp_dist(
        ax=axes,
        samples=y_postpred.transpose("sample", ...),
        x=this_xnew,
        plot_samples=False
    )
    plt.plot(this_xnew.flatten(), y_postpred_median, "y", linewidth=1, label="Median")
    plt.title(f"Post. pred. samples ({path_to_trace.stem})")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()

def plot_priorpost_cnr(path_to_trace, variable_names=None):
    """Plot 'corner plot' of prior and posterior samples for each GP hyperparameter"""

    this_trace = az.from_netcdf(path_to_trace)

    ax_list = az.plot_pair(
        this_trace,
        group="prior",
        var_names=variable_names,
        marginals=True,
        figsize=(10,10),
        kind=["scatter"],
        marginal_kwargs={"color":"red"},
        scatter_kwargs={"alpha":0.5, "color":"red"}
    )
    az.plot_pair(
        this_trace,
        group="posterior",
        var_names=variable_names,
        marginals=True,
        kind=["scatter"],
        ax=ax_list,
        scatter_kwargs={"alpha":0.01}
    )

def print_post_summary(path_to_trace, variable_names=None):
    """Wrapper for printing MCMC trace summary statistics from .NC file"""

    this_trace = az.from_netcdf(path_to_trace)
    return az.summary(this_trace, var_names=variable_names,
                      stat_focus='median', stat_funcs={'mean': np.mean, 'sd': np.std}, extend=True,
                      kind='all', round_to='none', hdi_prob=0.68)

def plot_traces(path_to_trace, variable_names=None):
    """Wrapper for plotting MCMC traces from .NC file"""

    az.style.use("arviz-white")
    this_trace = az.from_netcdf(filename=path_to_trace)
    return az.plot_trace(this_trace, var_names=variable_names, combined=False)

def fit_se_gp(path_to_csv, rng_seed=None):
    """Fit GP using squared exponential kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    N = t.shape[0]
    y_min, y_max, y_range = np.nanmin(y), np.nanmax(y), np.ptp(y)
    y_mean, y_sd = np.nanmean(y), np.nanstd(y)
    y_stderr_mean, y_stderr_sd = np.nanmean(y_stderr), np.nanstd(y_stderr)
    t_min, t_max, t_range = np.nanmin(t), np.nanmax(t), np.ptp(t)
    t_mingap, t_maxgap = np.diff(t).min(), np.diff(t).max()

    t = t - t_min # translate minimum to origin

    with pm.Model() as model:
        log_2ell_SE_sq = pm.Uniform("log_2ell_SE_sq", lower=-10, upper=math.log(2*t_range**2))
        ell_SE = pm.Deterministic("ell_SE", 0.5*math.sqrt(2) * pm.math.exp(0.5*log_2ell_SE_sq))

        log_eta_SE= pm.Uniform("log_eta_SE", lower=-15, upper=5)
        eta_SE = pm.Deterministic("eta_SE", pm.math.exp(log_eta_SE))

        cov_func = eta_SE**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE)
        
        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function 

        sig = pm.HalfNormal("sig", sigma=y_stderr) 
        cov_noise = pm.gp.cov.WhiteNoise(sigma=y_stderr)

        y_ = gp.marginal_likelihood(
            "y", 
            X=t.reshape(-1,1), 
            y=y.reshape(-1,1).flatten(), 
            sigma=cov_noise
        ) 
        
        se_trace = pm.sample_prior_predictive(samples=N_PPC, random_seed=rng_seed)

        se_trace.extend(
            pm.sample(
                draws=N_DRAWS, 
                tune=N_TUNE, 
                chains=4,
                cores=4, 
                random_seed=rng_seed
            )
        )       
    
        t_new = np.linspace(
            start=np.floor(t.min()),
            stop=np.ceil(t.max()),
            num = N_NEW
        ).reshape(-1,1)

        f_star = gp.conditional(name="f_star", Xnew=t_new, jitter=1e-6, pred_noise=False)

        se_trace.extend(
            pm.sample_posterior_predictive(
                se_trace.posterior,
                var_names=["f_star"],
                random_seed=rng_seed
            )
        )
        se_dag = pm.model_to_graphviz(model)

    return se_trace, se_dag

def fit_m32_gp(path_to_csv, rng_seed=None):
    """Fit GP using Matern 3/2 kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    N = t.shape[0]
    t_min, t_range = np.nanmin(t), np.ptp(t)
    
    t = t - t_min # translate minimum to origin

    with pm.Model() as model:
        log_2ell_M32_sq = pm.Uniform("log_2ell_M32_sq", lower=-10, upper=math.log(2*t_range**2))
        ell_M32 = pm.Deterministic("ell_M32", 0.5*math.sqrt(2) * pm.math.exp(0.5*log_2ell_M32_sq))

        log_eta_M32= pm.Uniform("log_eta_M32", lower=-15, upper=5)
        eta_M32 = pm.Deterministic("eta_M32", pm.math.exp(log_eta_M32))

        cov_func = eta_M32**2 * pm.gp.cov.Matern32(input_dim=1, ls=ell_M32)
        
        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function 

        sig = pm.HalfNormal("sig", sigma=y_stderr) 
        cov_noise = pm.gp.cov.WhiteNoise(sigma=y_stderr)

        y_ = gp.marginal_likelihood(
            "y", 
            X=t.reshape(-1,1), 
            y=y.reshape(-1,1).flatten(), 
            sigma=cov_noise
        ) 
        
        m32_trace = pm.sample_prior_predictive(samples=N_PPC, random_seed=rng_seed)

        m32_trace.extend(
            pm.sample(
                draws=N_DRAWS, 
                tune=N_TUNE, 
                chains=4,
                cores=4, 
                random_seed=rng_seed
            )
        )       
    
        t_new = np.linspace(
            start=np.floor(t.min()),
            stop=np.ceil(t.max()),
            num = N_NEW
        ).reshape(-1,1)

        f_star = gp.conditional(name="f_star", Xnew=t_new, jitter=1e-6, pred_noise=False)

        m32_trace.extend(
            pm.sample_posterior_predictive(
                m32_trace.posterior,
                var_names=["f_star"],
                random_seed=rng_seed
            )
        )
        m32_dag = pm.model_to_graphviz(model)

    return m32_trace, m32_dag

def fit_sem32_gp(path_to_csv, multiplicative_kernel=False, rng_seed=None):
    """Fit GP using Squared Exponential + Matern 3/2 kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    N = t.shape[0]
    t_min, t_range = np.nanmin(t), np.ptp(t)
    
    t = t - t_min # translate minimum to origin

    with pm.Model() as model:
        log_2ell_SE_sq = pm.Uniform("log_2ell_SE_sq", lower=-10, upper=math.log(2*t_range**2))
        ell_SE = pm.Deterministic("ell_SE", 0.5*math.sqrt(2) * pm.math.exp(0.5*log_2ell_SE_sq))

        log_2ell_M_sq = pm.Uniform("log_2ell_M_sq", lower=-10, upper=math.log(2*t_range**2))
        ell_M = pm.Deterministic("ell_M", 0.5*math.sqrt(2) * pm.math.exp(0.5*log_2ell_M_sq))

        if not multiplicative_kernel:

            log_eta_SE= pm.Uniform("log_eta_SE", lower=-15, upper=5)
            eta_SE = pm.Deterministic("eta_SE", pm.math.exp(log_eta_SE))

            log_eta_M= pm.Uniform("log_eta_M", lower=-15, upper=5)
            eta_M = pm.Deterministic("eta_M", pm.math.exp(log_eta_M))

            cov_func = eta_SE**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE) + \
                eta_M**2 * pm.gp.cov.Matern32(input_dim=1, ls=ell_M)
        else:

            log_eta= pm.Uniform("log_eta", lower=-15, upper=5)
            eta = pm.Deterministic("eta", pm.math.exp(log_eta))

            cov_func = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE) * \
                pm.gp.cov.Matern32(input_dim=1, ls=ell_M)

        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function 

        sig = pm.HalfNormal("sig", sigma=y_stderr) 
        cov_noise = pm.gp.cov.WhiteNoise(sigma=y_stderr)

        y_ = gp.marginal_likelihood(
            "y", 
            X=t.reshape(-1,1), 
            y=y.reshape(-1,1).flatten(), 
            sigma=cov_noise
        ) 
        
        sem32_trace = pm.sample_prior_predictive(samples=N_PPC, random_seed=rng_seed)

        sem32_trace.extend(
            pm.sample(
                draws=N_DRAWS, 
                tune=N_TUNE, 
                chains=4,
                cores=4, 
                random_seed=rng_seed
            )
        )       
    
        t_new = np.linspace(
            start=np.floor(t.min()),
            stop=np.ceil(t.max()),
            num = N_NEW
        ).reshape(-1,1)

        f_star = gp.conditional(name="f_star", Xnew=t_new, jitter=1e-6, pred_noise=False)

        sem32_trace.extend(
            pm.sample_posterior_predictive(
                sem32_trace.posterior,
                var_names=["f_star"],
                random_seed=rng_seed
            )
        )
        sem32_dag = pm.model_to_graphviz(model)

    return sem32_trace, sem32_dag

def fit_gpSE_gpM32(path_to_csv, rng_seed=None):
    """Fit compound GP model: Squared Exponential GP + Matern 3/2 GP."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    N = t.shape[0]
    t_min, t_range = np.nanmin(t), np.ptp(t)
    
    t = t - t_min # translate minimum to origin

    with pm.Model() as model:
        log_2ell_SE_sq = pm.Uniform("log_2ell_SE_sq", lower=-10, upper=math.log(2*t_range**2))
        ell_SE = pm.Deterministic("ell_SE", 0.5*math.sqrt(2) * pm.math.exp(0.5*log_2ell_SE_sq))
        log_eta_SE= pm.Uniform("log_eta_SE", lower=-15, upper=5)
        eta_SE = pm.Deterministic("eta_SE", pm.math.exp(log_eta_SE))

        cov_SE = eta_SE**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE) 
        gp_SE = pm.gp.Marginal(cov_func=cov_SE) 

        log_2ell_M32_sq = pm.Uniform("log_2ell_M32_sq", lower=-10, upper=math.log(2*t_range**2))
        ell_M32 = pm.Deterministic("ell_M32", 0.5*math.sqrt(2) * pm.math.exp(0.5*log_2ell_M32_sq))
        log_eta_M32 = pm.Uniform("log_eta_M32", lower=-15, upper=5)
        eta_M32 = pm.Deterministic("eta_M32", pm.math.exp(log_eta_M32))

        cov_M32 = eta_M32**2 * pm.gp.cov.Matern32(input_dim=1, ls=ell_M32)
        gp_M32 = pm.gp.Marginal(cov_func=cov_M32)

        gp = gp_SE + gp_M32

        sig = pm.HalfNormal("sig", sigma=y_stderr) 
        cov_noise = pm.gp.cov.WhiteNoise(sigma=y_stderr)

        y_ = gp.marginal_likelihood(
            "y", 
            X=t.reshape(-1,1), 
            y=y.reshape(-1,1).flatten(), 
            sigma=cov_noise
        ) 

        gpSE_gpM32_trace = pm.sample_prior_predictive(samples=N_PPC, random_seed=rng_seed)

        gpSE_gpM32_trace.extend(
            pm.sample(
                draws=N_DRAWS, 
                tune=N_TUNE, 
                chains=4,
                cores=4, 
                random_seed=rng_seed
            )
        )       
    
        t_new = np.linspace(
            start=np.floor(t.min()),
            stop=np.ceil(t.max()),
            num = N_NEW
        ).reshape(-1,1)

        f_star = gp.conditional(name="f_star", Xnew=t_new, jitter=1e-6, pred_noise=False)

        gpSE_gpM32_trace.extend(
            pm.sample_posterior_predictive(
                gpSE_gpM32_trace.posterior,
                var_names=["f_star"],
                random_seed=rng_seed
            )
        )
        gpSE_gpM32_dag = pm.model_to_graphviz(model)

    return gpSE_gpM32_trace, gpSE_gpM32_dag

