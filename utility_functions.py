"""Module containing utility functions for fitting and plotting results of GP fitting"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
from pymc.gp.util import plot_gp_dist
import arviz as az
import xarray as xr
from scipy import signal

N_DRAWS = 1000
N_TUNE = 2000
N_PPC = N_DRAWS # No. prior predictive samples
N_NEW = 300 # No. observation points in each posterior predictive sample

FIG_SIZE = (10,6)

WELCH_NFFT=512
WELCH_DETREND=False
WELCH_SCALING="density"
WELCH_AVERAGE="median"

def plot_lc(path_to_csv):
    """Plot light curve from raw CSV data including 1 sigma error bars and overall mean."""

    this_lc = pd.read_csv(path_to_csv)
    this_x = this_lc['mjd']
    this_y = this_lc['f_peak']
    this_yerr = this_lc['f_peak_err']
    mean_y = np.nanmean(this_y)

    plt.figure(figsize=FIG_SIZE)
    plt.plot(this_x, this_y, "_b", ms=8, alpha=1, label="Observed data")
    plt.axhline(y=mean_y, c='blue', ls=':')
    plt.errorbar(x=this_x, y=this_y, yerr=this_yerr,
                 fmt="none", ecolor="k", elinewidth=1, capsize=3,
                 label=r"1 $\sigma$")
    plt.title(f"{path_to_csv.stem} (N={len(this_y)})")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()

def plot_priorpred_samples(path_to_trace, path_to_csv):
    """Plot prior predictive samples and original data points"""

    this_lc = pd.read_csv(path_to_csv)
    this_trace = az.from_netcdf(filename=path_to_trace)

    this_x = this_lc['mjd']
    this_y = this_lc['f_peak']

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)

    for prior_pred in this_trace.prior_predictive.y.to_numpy()[0]:
        ax.plot(this_x, prior_pred, lw=0.5, alpha=0.2, color="red")

    ax.scatter(x=this_x, y=this_y, s=1.5, c="blue", zorder=10)
    ax.axhline(0, color="black")
    ax.set_title("Samples from the GP prior")
    ax.set_ylabel("y")
    ax.set_xlabel("t")

def plot_priorpred_PSD(path_to_trace, variable_name="y"):
    """Plot the power spectral density of prior predictive samples"""

    this_trace = az.from_netcdf(filename=path_to_trace)

    priorpred_DataArray = az.extract(this_trace, "prior_predictive", var_names=variable_name)

    freqs_nd, welch_psds_nd  = signal.welch(
        x=priorpred_DataArray, axis=0,
        fs=1,
        nfft=WELCH_NFFT, detrend=WELCH_DETREND, scaling=WELCH_SCALING, average=WELCH_AVERAGE
    )

    welch_psds_dataset = xr.Dataset(
        data_vars=dict(
            power=(["freq", "sample"], welch_psds_nd)
        ),
        coords=dict(
            freq=freqs_nd,
            sample=range(1,priorpred_DataArray.shape[1] + 1)
        )
    )

    psd_median = welch_psds_dataset.median(dim="sample").to_array().to_numpy().flatten()
    psd_q975 = welch_psds_dataset.quantile(q=0.975, dim="sample").to_array().to_numpy().flatten()
    psd_q84 = welch_psds_dataset.quantile(q=0.84, dim="sample").to_array().to_numpy().flatten()
    psd_q16 = welch_psds_dataset.quantile(q=0.16, dim="sample").to_array().to_numpy().flatten()
    psd_q025 = welch_psds_dataset.quantile(q=0.025, dim="sample").to_array().to_numpy().flatten()

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)
    ax.fill_between(freqs_nd, psd_q16, psd_q84, alpha=0.7, color="blue", label=r"68% HDI")
    ax.fill_between(freqs_nd, psd_q025, psd_q975, alpha=0.5, color="blue", label=r"95% HDI")
    ax.loglog(freqs_nd, psd_median, lw=2,color="red", alpha=0.8, label=r"Median")
    ax.set_xlabel("Frequency of modulation (Hz)")
    ax.set_ylabel(r"PSD (Jy$^2$ Hz)")
    ax.set_title(f"Welch PSD of Prior Predictive Samples")
    ax.legend()

def plot_postpred_samples(path_to_trace, path_to_csv, variable_name="f_star"):
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

    y_postpred = az.extract(this_trace, "posterior_predictive", var_names=variable_name)
    y_postpred_median = y_postpred.median(dim="sample")

    fig = plt.figure(figsize=FIG_SIZE)
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
    plt.title(f"{variable_name} ({path_to_trace.stem})")
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
        figsize=(8,8),
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

def plot_welch_psd(trace, variable_name="f_star"):
    """Plot Welch approximated power spectral density (PSD) of GP posterior predictive samples."""

    postpred_DataArray = az.extract(trace, "posterior_predictive", var_names=variable_name)

    freqs_nd, welch_psds_nd  = signal.welch(
        x=postpred_DataArray, axis=0,
        fs=1,
        nfft=WELCH_NFFT, detrend=WELCH_DETREND, scaling=WELCH_SCALING, average=WELCH_AVERAGE
    )

    welch_psds_dataset = xr.Dataset(
        data_vars=dict(
            power=(["freq", "sample"], welch_psds_nd)
        ),
        coords=dict(
            freq=freqs_nd,
            sample=range(1,postpred_DataArray.shape[1] + 1)
        )
    )

    psd_median = welch_psds_dataset.median(dim="sample").to_array().to_numpy().flatten()
    psd_q975 = welch_psds_dataset.quantile(q=0.975, dim="sample").to_array().to_numpy().flatten()
    psd_q84 = welch_psds_dataset.quantile(q=0.84, dim="sample").to_array().to_numpy().flatten()
    psd_q16 = welch_psds_dataset.quantile(q=0.16, dim="sample").to_array().to_numpy().flatten()
    psd_q025 = welch_psds_dataset.quantile(q=0.025, dim="sample").to_array().to_numpy().flatten()

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)
    ax.fill_between(freqs_nd, psd_q16, psd_q84, alpha=0.7, color="blue", label=r"68% HDI")
    ax.fill_between(freqs_nd, psd_q025, psd_q975, alpha=0.5, color="blue", label=r"95% HDI")
    ax.loglog(freqs_nd, psd_median, lw=2,color="red", alpha=0.8, label=r"Median")
    ax.set_xlabel("Frequency of modulation (Hz)")
    ax.set_ylabel(r"PSD (Jy$^2$ Hz)")
    ax.set_title(f"Welch PSD of {variable_name}")
    ax.legend()

def fit_se_gp(path_to_csv, rng_seed=None):
    """Fit GP using squared exponential kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    y_stderr_sd = np.nanstd(y_stderr)
    t = lc["mjd"].to_numpy()

    t_mingap = np.diff(t).min()

    with pm.Model() as model:
        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)

        eta_SE = pm.Truncated("eta_SE", std_norm_dist, lower=0, upper=None)
        ell_SE = pm.InverseGamma("ell_SE", alpha=3, beta=8*math.ceil(t_mingap))

        cov_func = eta_SE * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE)
        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function

        err_norm_dist = pm.Normal.dist(mu=y_stderr, sigma=y_stderr_sd)
        sig_SE = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)
        cov_noise = pm.gp.cov.WhiteNoise(sigma=sig_SE)

        y_ = gp.marginal_likelihood(
            "y", 
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=cov_noise
        )

        se_dag = pm.model_to_graphviz(model)
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

    return se_trace, se_dag

def fit_m32_gp(path_to_csv, rng_seed=None):
    """Fit GP using Matern 3/2 kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    y_stderr_sd = np.nanstd(y_stderr)
    t = lc["mjd"].to_numpy()
    t_mingap = np.diff(t).min()

    with pm.Model() as model:
        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)

        eta_M32 = pm.Truncated("eta_M32", std_norm_dist, lower=0, upper=None)
        ell_M32 = pm.InverseGamma("ell_M32", alpha=3, beta=8*math.ceil(t_mingap))

        cov_func = eta_M32 * pm.gp.cov.Matern32(input_dim=1, ls=ell_M32)
        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function

        err_norm_dist = pm.Normal.dist(mu=y_stderr, sigma=y_stderr_sd)
        sig_M32 = pm.Truncated("sig_M32", err_norm_dist, lower=0, upper=None)
        cov_noise = pm.gp.cov.WhiteNoise(sigma=sig_M32)

        y_ = gp.marginal_likelihood(
            "y",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=cov_noise
        )

        m32_dag = pm.model_to_graphviz(model)
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

    return m32_trace, m32_dag

def fit_se_m32_gp(path_to_csv, multiplicative_kernel=False, rng_seed=None):
    """Fit GP using Squared Exponential + Matern 3/2 kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    y_stderr_sd = np.nanstd(y_stderr)
    t = lc["mjd"].to_numpy()
    t_mingap = np.diff(t).min()

    with pm.Model() as model:
        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
        eta = pm.Truncated("eta", std_norm_dist, lower=0, upper=None)

        ell_SE = pm.InverseGamma("ell_SE", alpha=3, beta=8*math.ceil(t_mingap))
        ell_M32 = pm.InverseGamma("ell_M32", alpha=3, beta=8*math.ceil(t_mingap))

        if not multiplicative_kernel:
            cov_func = eta * \
                (pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE) + pm.gp.cov.Matern32(input_dim=1, ls=ell_M32))
        else:
            cov_func = eta * \
                pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE) * pm.gp.cov.Matern32(input_dim=1, ls=ell_M32)

        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function

        err_norm_dist = pm.Normal.dist(mu=y_stderr, sigma=y_stderr_sd)
        sig = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)
        cov_noise = pm.gp.cov.WhiteNoise(sigma=sig)

        y_ = gp.marginal_likelihood(
            "y",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=cov_noise
        )

        sem32_dag = pm.model_to_graphviz(model)
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
    return sem32_trace, sem32_dag

def fit_gpSE_gpM32(path_to_csv, rng_seed=None):
    """Fit compound GP model: Squared Exponential GP + Matern 3/2 GP."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    y_stderr_sd = np.nanstd(y_stderr)
    t = lc["mjd"].to_numpy()
    t_mingap = np.diff(t).min()

    with pm.Model() as model:

        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
        eta_SE = pm.Truncated("eta_SE", std_norm_dist, lower=0, upper=None)
        eta_M32 = pm.Truncated("eta_M32", std_norm_dist, lower=0, upper=None)

        ell_SE = pm.InverseGamma("ell_SE", alpha=3, beta=8*math.ceil(t_mingap))
        ell_M32 = pm.InverseGamma("ell_M32", alpha=3, beta=8*math.ceil(t_mingap))

        cov_SE = eta_SE * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE)
        gp_SE = pm.gp.Marginal(cov_func=cov_SE)

        cov_M32 = eta_M32 * pm.gp.cov.Matern32(input_dim=1, ls=ell_M32)
        gp_M32 = pm.gp.Marginal(cov_func=cov_M32)

        gp = gp_SE + gp_M32

        err_norm_dist = pm.Normal.dist(mu=y_stderr, sigma=y_stderr_sd)
        sig = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)
        cov_noise = pm.gp.cov.WhiteNoise(sigma=sig)

        f = gp.marginal_likelihood(
            "f",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=cov_noise
        )

        gpSE_gpM32_dag = pm.model_to_graphviz(model)
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

        f_star_SE = gp_SE.conditional("f_star_SE", Xnew=t_new,
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": sig, "gp": gp})

        f_star_M32 = gp_M32.conditional("f_star_M32", Xnew=t_new,
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": sig, "gp": gp})

        f_star = gp.conditional(name="f_star", Xnew=t_new, jitter=1e-6, pred_noise=False)

        gpSE_gpM32_trace.extend(
            pm.sample_posterior_predictive(
                gpSE_gpM32_trace.posterior,
                var_names=["f_star", "f_star_SE", "f_star_M32"],
                random_seed=rng_seed
            )
        )
    return gpSE_gpM32_trace, gpSE_gpM32_dag

def fit_gpSE_gpPer(path_to_csv, rng_seed=None):
    """Fit compound GP model: Squared Exponential GP + Periodic GP."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    y_stderr_sd = np.nanstd(y_stderr)
    t = lc["mjd"].to_numpy()
    t_mingap = np.diff(t).min()
    t_range = np.ptp(t)

    with pm.Model() as model:

        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
        eta_SE = pm.Truncated("eta_SE", std_norm_dist, lower=0, upper=None)
        eta_Per = pm.Truncated("eta_Per", std_norm_dist, lower=0, upper=None)

        ell_SE = pm.InverseGamma("ell_SE", alpha=3, beta=8*math.ceil(t_mingap))
        ell_Per = pm.InverseGamma("ell_Per", alpha=3, beta=8*math.ceil(t_mingap))

        T = pm.Uniform("T", lower=t_range/4, upper=t_range)

        cov_SE = eta_SE * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE)
        gp_SE = pm.gp.Marginal(cov_func=cov_SE)

        cov_Per = eta_Per * pm.gp.cov.Periodic(input_dim=1, period=T, ls=ell_Per)
        gp_Per = pm.gp.Marginal(cov_func=cov_Per)

        gp = gp_SE + gp_Per

        err_norm_dist = pm.Normal.dist(mu=y_stderr, sigma=y_stderr_sd)
        sig = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)
        cov_noise = pm.gp.cov.WhiteNoise(sigma=sig)

        f = gp.marginal_likelihood(
            "f",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=cov_noise
        )

        gpSE_gpPer_dag = pm.model_to_graphviz(model)
        gpSE_gpPer_trace = pm.sample_prior_predictive(samples=N_PPC, random_seed=rng_seed)

        gpSE_gpPer_trace.extend(
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

        f_star_SE = gp_SE.conditional("f_star_SE", Xnew=t_new,
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": sig, "gp": gp})

        f_star_Per = gp_Per.conditional("f_star_Per", Xnew=t_new,
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": sig, "gp": gp})

        f_star = gp.conditional(name="f_star", Xnew=t_new, jitter=1e-6, pred_noise=False)

        gpSE_gpPer_trace.extend(
            pm.sample_posterior_predictive(
                gpSE_gpPer_trace.posterior,
                var_names=["f_star", "f_star_SE", "f_star_Per"],
                random_seed=rng_seed
            )
        )
    return gpSE_gpPer_trace, gpSE_gpPer_dag



def plot_welch_psds(trace, variable_names=("f_star", "f_star_SE", "f_star_M32")):
    """Plot Welch approximated PSD of GP posterior predictive samples for each constituent kernel."""

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)

    for var in variable_names:
        postpred_DataArray = az.extract(trace, "posterior_predictive", var_names=var)

        freqs_nd, welch_psds_nd  = signal.welch(
            x=postpred_DataArray, axis=0,
            fs=1,
            nfft=WELCH_NFFT, detrend=WELCH_DETREND, scaling=WELCH_SCALING, average=WELCH_AVERAGE
        )

        welch_psds_dataset = xr.Dataset(
            data_vars=dict(
                power=(["freq", "sample"], welch_psds_nd)
            ),
            coords=dict(
                freq=freqs_nd,
                sample=range(1,postpred_DataArray.shape[1] + 1)
            )
        )

        psd_median = welch_psds_dataset.median(dim="sample").to_array().to_numpy().flatten()
        psd_q975 = welch_psds_dataset.quantile(q=0.975, dim="sample").to_array().to_numpy().flatten()
        psd_q84 = welch_psds_dataset.quantile(q=0.84, dim="sample").to_array().to_numpy().flatten()
        psd_q16 = welch_psds_dataset.quantile(q=0.16, dim="sample").to_array().to_numpy().flatten()
        psd_q025 = welch_psds_dataset.quantile(q=0.025, dim="sample").to_array().to_numpy().flatten()

        #ax.fill_between(freqs_nd, psd_q16, psd_q84, alpha=0.7, label=r"68% HDI")
        #ax.fill_between(freqs_nd, psd_q025, psd_q975, alpha=0.5, label=r"95% HDI")
        ax.loglog(freqs_nd, psd_median, lw=2, alpha=0.8, label=f"{var}")

    ax.set_xlabel("Frequency of modulation (Hz)")
    ax.set_ylabel(r"PSD (Jy$^2$ Hz)")
    ax.set_title(f"Welch PSD")
    ax.legend()


