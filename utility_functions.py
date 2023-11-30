"""Module containing utility functions for fitting and plotting results of GP fitting"""
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pymc as pm
from pymc.gp.util import plot_gp_dist
import arviz as az
from scipy import signal
import seaborn as sns
from astropy.timeseries import LombScargle

print(f"Running on PyMC v{pm.__version__}")

N_DRAWS = 1000
N_TUNE = 2000
N_PPC = N_DRAWS # No. prior predictive samples
N_NEW = 200 # No. observation points in each posterior predictive sample

FIG_SIZE = (10,5)

LSP_FITMEAN = True
LSP_CENTER_DATA = False
LSP_NORMALIZATION = "standard"

WELCH_FS=1.0
WELCH_NPERSEG=256
WELCH_NOVERLAP=None
WELCH_NFFT=512
WELCH_DETREND=False
WELCH_SCALING="density"
WELCH_AVERAGE="median"

PLOT_COLOURS = ['g', 'r', 'b', 'c', 'm', 'y', 'k']

def plot_lc(path_to_csv, plot_mean=False, show_title=False, show_legend=False, save_plot=False):
    """Plot light curve from raw CSV data including 1 sigma error bars and overall mean."""

    this_lc = pd.read_csv(path_to_csv)
    this_x = this_lc['mjd']
    this_y = this_lc['f_peak']
    this_yerr = this_lc['f_peak_err']
    mean_y = np.nanmean(this_y)

    fig = plt.figure(figsize=FIG_SIZE)
    plt.plot(this_x, this_y, "_b", ms=8, alpha=1, label="Observed data")
    if plot_mean:
        plt.axhline(y=mean_y, c='blue', ls=':')
    plt.errorbar(x=this_x, y=this_y, yerr=this_yerr,
                 fmt="none", ecolor="red", elinewidth=1, capsize=3,
                 label=r"1 $\sigma$")
    sns.rugplot(this_x, height=0.025, color='red')
    if show_title:
        plt.title(f"{path_to_csv.stem} (N={len(this_y)})")
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux Density (Jy)")
    if show_legend:
        plt.legend()

    if save_plot:
        fig.savefig(f'figures/{path_to_csv.stem}_lc.jpg', dpi=300, bbox_inches='tight')

def plot_priorpred_samples(trace, variable_name="y"):
    """Plot prior predictive samples and original data points"""

    this_x = trace.constant_data.t
    this_y = trace.observed_data.y

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)

    for prior_pred in az.extract(trace, "prior_predictive", var_names=variable_name).to_numpy().T:
        ax.plot(this_x, prior_pred, lw=0.5, alpha=0.2, color="red")

    ax.scatter(x=this_x, y=this_y, s=1.5, c="blue", zorder=10)
    ax.axhline(0, color="black")
    ax.set_title("Samples from the GP prior")
    ax.set_ylabel("y")
    ax.set_xlabel("t")

def plot_postpred_samples(trace, variable_name="f_star", show_title=False, show_legend=False, save_plot=False):
    """Plot posterior predicted samples and original data light curve"""

    csv_filename = trace.constant_data.attrs['csv_filename']

    this_x = trace.constant_data.t
    this_y = trace.observed_data.y
    this_yerr = trace.constant_data.y_stderr

    y_postpred = az.extract(trace, "posterior_predictive", var_names=variable_name)
    y_postpred_median = y_postpred.median(dim="sample")

    this_xnew = trace.constant_data.t_star.to_numpy()

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
    sns.rugplot(this_xnew, height=0.025, color='red')
    if show_title:
        plt.title(f"{variable_name} ({csv_filename})")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    if show_legend:
        plt.legend()
    if save_plot:
        fig.savefig(f'figures/{csv_filename}_postpred.jpg', dpi=300, bbox_inches='tight')

def plot_priorpost_cnr(trace, variable_names=None):
    """Plot 'corner plot' of prior and posterior samples for each GP hyperparameter"""

    ax_list = az.plot_pair(
        trace,
        group="prior",
        var_names=variable_names,
        marginals=True,
        figsize=(8,8),
        kind=["scatter"],
        marginal_kwargs={"color":"red"},
        scatter_kwargs={"alpha":0.5, "color":"red"}
    )
    az.plot_pair(
        trace,
        group="posterior",
        var_names=variable_names,
        marginals=True,
        kind=["scatter"],
        ax=ax_list,
        scatter_kwargs={"alpha":0.01}
    )

def plot_post_cnr(trace, variable_names=None):
    """Plot 'corner plot' of posterior samples for each GP hyperparameter"""

    az.plot_pair(
        trace,
        group="posterior",
        var_names=variable_names,
        marginals=True,
        figsize=(6,6),
        textsize=14,
        kind=["scatter"],
        scatter_kwargs={"alpha":0.1}
    )

def print_post_summary(trace, variable_names=None):
    """Wrapper for printing MCMC trace summary statistics from .NC file"""

    return az.summary(trace, 
                      var_names=variable_names,
                      stat_focus='median', 
                      stat_funcs={'mean': np.mean, 'sd': np.std}, 
                      extend=True,
                      kind='all', 
                      round_to='none', 
                      hdi_prob=0.68)

def plot_traces(trace, variable_names=None):
    """Wrapper for plotting MCMC traces from .NC file"""

    az.style.use("arviz-white")
    return az.plot_trace(trace, var_names=variable_names, combined=False)

def plot_welch_psd(trace, group="posterior_predictive", variable_name="f_star", show_title=False):
    """Plot Welch approximated power spectral density (PSD) of GP posterior/prior predictive samples."""

    pred_DataArray = az.extract(trace, group=group, var_names=variable_name)

    freqs_nd, welch_psds_nd  = signal.welch(
        x=pred_DataArray, axis=0,
        fs=WELCH_FS,
        nperseg=WELCH_NPERSEG,
        nfft=WELCH_NFFT,
        detrend=WELCH_DETREND,
        scaling=WELCH_SCALING,
        average=WELCH_AVERAGE
    )

    welch_psds_dataset = xr.Dataset(
        data_vars=dict(
            power=(["freq", "sample"], welch_psds_nd)
        ),
        coords=dict(
            freq=freqs_nd,
            sample=range(1,pred_DataArray.shape[1] + 1)
        )
    )

    psd_median = welch_psds_dataset.median(dim="sample").to_array().to_numpy().flatten()
    psd_q975 = welch_psds_dataset.quantile(q=0.975, dim="sample").to_array().to_numpy().flatten()
    psd_q84 = welch_psds_dataset.quantile(q=0.84, dim="sample").to_array().to_numpy().flatten()
    psd_q16 = welch_psds_dataset.quantile(q=0.16, dim="sample").to_array().to_numpy().flatten()
    psd_q025 = welch_psds_dataset.quantile(q=0.025, dim="sample").to_array().to_numpy().flatten()

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)
    sns.rugplot(freqs_nd, height=0.025, ax=ax,  color='red')
    ax.fill_between(freqs_nd, psd_q16, psd_q84, alpha=0.7, color="blue", label=r"68% HDI")
    ax.fill_between(freqs_nd, psd_q025, psd_q975, alpha=0.5, color="blue", label=r"95% HDI")
    ax.loglog(freqs_nd, psd_median, lw=2,color="red", alpha=0.8, label=r"Median")
    ax.set_xlabel("Frequency of modulation (Hz)")
    ax.set_ylabel(r"PSD (Jy$^2$ Hz)")
    if show_title:
        ax.set_title(f"Welch PSD: {variable_name} ({trace.constant_data.attrs['csv_filename']})")
    ax.legend()

def plot_welch_psds(trace, group="posterior_predictive", variable_names=("f_star_SE", "f_star_Per", "f_star"), show_title=False):
    """Plot Welch approximated PSD of GP posterior predictive samples for each constituent kernel."""

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)

    for col, var in zip(PLOT_COLOURS, variable_names):
        postpred_DataArray = az.extract(trace, group=group, var_names=var)

        freqs_nd, welch_psds_nd  = signal.welch(
            x=postpred_DataArray, axis=0,
            fs=WELCH_FS,
            nperseg=WELCH_NPERSEG,
            nfft=WELCH_NFFT,
            detrend=WELCH_DETREND,
            scaling=WELCH_SCALING,
            average=WELCH_AVERAGE
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

        ax.loglog(freqs_nd, psd_median, lw=2, color=col, alpha=0.8, label=f"{var}")

    sns.rugplot(freqs_nd, height=0.025, ax=ax,  color='brown');

    ax.set_xlabel("Frequency of modulation (Hz)")
    ax.set_ylabel(r"PSD (Jy$^2$ Hz)")
    if show_title:
        ax.set_title(f"Welch PSD ({trace.constant_data.attrs['csv_filename']})")
    ax.legend()

def plot_lsp(trace, group="posterior_predictive", variable_name="f_star", show_title=False):

    pred_DataArray = az.extract(trace, group=group, var_names=variable_name)
    freqs_f = np.geomspace(start=1e-4, stop=1, num=200)

    n_pred_samples = pred_DataArray.shape[1]
    n_freqs = freqs_f.shape[0]

    pred_LSPs_np = np.ndarray((n_freqs, n_pred_samples))

    if group == "prior_predictive":
        this_t = trace.constant_data.t
    elif group == "posterior_predictive":
        this_t = trace.constant_data.t_star

    for lc_index in range(n_pred_samples):

        this_y = pred_DataArray[:, lc_index]

        this_power = LombScargle(
            t=this_t,
            y=this_y,
            dy=None,
            fit_mean=LSP_FITMEAN,
            center_data=LSP_CENTER_DATA,
            normalization=LSP_NORMALIZATION
        ).power(freqs_f)

        pred_LSPs_np[:, lc_index] = this_power

    pred_LSPs_xr = xr.Dataset(
        data_vars=dict(
            power=(["frequency", "draw"], pred_LSPs_np)
        ),
        coords=dict(
            frequency=freqs_f,
            draw=range(0, n_pred_samples)
        )
    )

    obs_power = LombScargle(
        t=trace.constant_data.t,
        y=trace.observed_data.y,
        dy=None,
        fit_mean=LSP_FITMEAN,
        center_data=LSP_CENTER_DATA,
        normalization=LSP_NORMALIZATION
    ).power(freqs_f)

    psd_median = pred_LSPs_xr.median(dim="draw").to_array().to_numpy().flatten()
    psd_q975 = pred_LSPs_xr.quantile(q=0.975, dim="draw").to_array().to_numpy().flatten()
    psd_q84 = pred_LSPs_xr.quantile(q=0.84, dim="draw").to_array().to_numpy().flatten()
    psd_q16 = pred_LSPs_xr.quantile(q=0.16, dim="draw").to_array().to_numpy().flatten()
    psd_q025 = pred_LSPs_xr.quantile(q=0.025, dim="draw").to_array().to_numpy().flatten()

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)
    ax.loglog(freqs_f, obs_power, lw=2,color="gray", alpha=0.8, label=r"Data")
    sns.rugplot(freqs_f, height=0.025, ax=ax,  color='red')
    ax.fill_between(freqs_f, psd_q16, psd_q84, alpha=0.7, color="blue", label=r"68% HDI")
    ax.fill_between(freqs_f, psd_q025, psd_q975, alpha=0.5, color="blue", label=r"95% HDI")
    ax.loglog(freqs_f, psd_median, lw=2,color="red", alpha=0.8, label=r"Median")
    
    ax.set_xlabel("Frequency")
    ax.set_ylabel(r"Power")
    if show_title:
        ax.set_title(f"LSP of {group} ({trace.constant_data.attrs['csv_filename']})")
    ax.legend()

def plot_lsps(trace, group="posterior_predictive", variable_names=["f_star_SE", "f_star_Per", "f_star"], show_title=False):
    """Plot Lomb-Scargle periodogram of posterior predictive samples for each constituent kernel."""

    freqs_f = np.geomspace(start=1e-4, stop=10, num=200)

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1,1,1)

    obs_power = LombScargle(
        t=trace.constant_data.t,
        y=trace.observed_data.y,
        dy=None,
        fit_mean=LSP_FITMEAN,
        center_data=LSP_CENTER_DATA,
        normalization=LSP_NORMALIZATION
    ).power(freqs_f)

    ax.loglog(freqs_f, obs_power, lw=2,color="gray", alpha=0.8, label=r"Data")
    sns.rugplot(freqs_f, height=0.025, ax=ax,  color='brown')

    for col, var in zip(PLOT_COLOURS, variable_names):

        pred_DataArray = az.extract(trace, group=group, var_names=var)
        

        n_pred_samples = pred_DataArray.shape[1]
        n_freqs = freqs_f.shape[0]

        pred_LSPs_np = np.ndarray((n_freqs, n_pred_samples))

        if group == "prior_predictive":
            this_t = trace.constant_data.t
        elif group == "posterior_predictive":
            this_t = trace.constant_data.t_star

        for lc_index in range(n_pred_samples):

            this_y = pred_DataArray[:, lc_index]

            this_power = LombScargle(
                t=this_t,
                y=this_y,
                dy=None,
                fit_mean=LSP_FITMEAN,
                center_data=LSP_CENTER_DATA,
                normalization=LSP_NORMALIZATION
            ).power(freqs_f)

            pred_LSPs_np[:, lc_index] = this_power

        pred_LSPs_xr = xr.Dataset(
            data_vars=dict(
                power=(["frequency", "draw"], pred_LSPs_np)
            ),
            coords=dict(
                frequency=freqs_f,
                draw=range(0, n_pred_samples)
            )
        )

        psd_median = pred_LSPs_xr.median(dim="draw").to_array().to_numpy().flatten()
        #psd_q975 = pred_LSPs_xr.quantile(q=0.975, dim="draw").to_array().to_numpy().flatten()
        #psd_q84 = pred_LSPs_xr.quantile(q=0.84, dim="draw").to_array().to_numpy().flatten()
        #psd_q16 = pred_LSPs_xr.quantile(q=0.16, dim="draw").to_array().to_numpy().flatten()
        #psd_q025 = pred_LSPs_xr.quantile(q=0.025, dim="draw").to_array().to_numpy().flatten()
        #ax.fill_between(freqs_f, psd_q16, psd_q84, alpha=0.5, color=col)
        #ax.fill_between(freqs_f, psd_q025, psd_q975, alpha=0.3, color=col)
        ax.loglog(freqs_f, psd_median, lw=2, alpha=0.8, color=col, label=f"{var}")

    ax.set_xlabel("Frequency")
    ax.set_ylabel(r"Power")
    if show_title:
        ax.set_title(f"LSP of {group} ({trace.constant_data.attrs['csv_filename']})")
    ax.legend()



def fit_se_gp(path_to_csv, rng_seed=None):
    """Fit GP using squared exponential kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    y_stderr_mean = np.nanmean(y_stderr)
    y_stderr_sd = np.nanstd(y_stderr)
    t_mingap = np.diff(np.sort(t)).min()

    t_star = np.linspace(
        start=np.floor(t.min()),
        stop=np.ceil(t.max()),
        num = N_NEW
    )

    coords = {"t": t, "t_star": t_star}

    with pm.Model(coords=coords) as model:
        t_ = pm.ConstantData("t", t, dims="obs_id")
        t_star_ = pm.ConstantData("t_star", t_star)
        y_stderr_ = pm.ConstantData("y_stderr", y_stderr, dims="obs_id")

        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)

        eta_SE = pm.Truncated("eta_SE", std_norm_dist, lower=0, upper=None)
        ell_SE = pm.InverseGamma("ell_SE", alpha=3, beta=8*math.ceil(t_mingap))

        cov_func = eta_SE * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE)
        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function

        err_norm_dist = pm.Normal.dist(mu=y_stderr_mean, sigma=y_stderr_sd)
        sig = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)

        y_ = gp.marginal_likelihood(
            "y", 
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=sig,
            jitter=1e-6,
        )

        se_dag = pm.model_to_graphviz(model)
        se_trace = pm.sample_prior_predictive(samples=N_PPC,
                                              random_seed=rng_seed,
                                              idata_kwargs={"dims": {"y": ["t"]}})

        se_trace.extend(
            pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=4,
                cores=4,
                random_seed=rng_seed
            )
        )

        f_star = gp.conditional(name="f_star",
                                Xnew=t_star.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=False)

        y_star = gp.conditional(name="y_star",
                                Xnew=t_star.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=True)

        se_trace.extend(
            pm.sample_posterior_predictive(
                se_trace,
                var_names=["f_star", "y_star"],
                random_seed=rng_seed,
                idata_kwargs={"dims": {"f_star": ["t_star"], "y_star": ["t_star"]}}
            )
        )
        se_trace.constant_data = se_trace.constant_data.assign_attrs(csv_filename=path_to_csv.stem)

    return se_trace, se_dag

def fit_m32_gp(path_to_csv, rng_seed=None):
    """Fit GP using Matern 3/2 kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    y_stderr_mean = np.nanmean(y_stderr)
    y_stderr_sd = np.nanstd(y_stderr)
    t_mingap = np.diff(np.sort(t)).min()

    t_new = np.linspace(
        start=np.floor(t.min()),
        stop=np.ceil(t.max()),
        num = N_NEW
    )

    coords = {"t": t, "t_star": t_new}

    with pm.Model(coords=coords) as model:
        t_ = pm.ConstantData("t", t, dims="obs_id")
        t_star_ = pm.ConstantData("t_star", t_new)
        y_stderr_ = pm.ConstantData("y_stderr", y_stderr, dims="obs_id")

        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)

        eta_M32 = pm.Truncated("eta_M32", std_norm_dist, lower=0, upper=None)
        ell_M32 = pm.InverseGamma("ell_M32", alpha=3, beta=8*math.ceil(t_mingap))

        cov_func = eta_M32 * pm.gp.cov.Matern32(input_dim=1, ls=ell_M32)
        gp = pm.gp.Marginal(cov_func=cov_func) # zero mean function

        err_norm_dist = pm.Normal.dist(mu=y_stderr_mean, sigma=y_stderr_sd)
        sig = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)

        y_ = gp.marginal_likelihood(
            "y",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=sig
        )

        m32_dag = pm.model_to_graphviz(model)
        m32_trace = pm.sample_prior_predictive(samples=N_PPC,
                                               random_seed=rng_seed,
                                               idata_kwargs={"dims": {"y": ["t"]}})

        m32_trace.extend(
            pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=4,
                cores=4,
                random_seed=rng_seed
            )
        )

        f_star = gp.conditional(name="f_star",
                                Xnew=t_new.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=False)
        
        y_star = gp.conditional(name="y_star",
                                Xnew=t_new.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=True)

        m32_trace.extend(
            pm.sample_posterior_predictive(
                m32_trace,
                var_names=["f_star", "y_star"],
                random_seed=rng_seed,
                idata_kwargs={"dims": {"f_star": ["t_star"], "y_star": ["t_star"]}}
            )
        )
        m32_trace.constant_data = m32_trace.constant_data.assign_attrs(csv_filename=path_to_csv.stem)

    return m32_trace, m32_dag

def fit_se_m32_gp(path_to_csv, multiplicative_kernel=False, rng_seed=None):
    """Fit GP using Squared Exponential + Matern 3/2 kernel."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    y_stderr_mean = np.nanmean(y_stderr)
    y_stderr_sd = np.nanstd(y_stderr)
    t_mingap = np.diff(np.sort(t)).min()

    t_new = np.linspace(
        start=np.floor(t.min()),
        stop=np.ceil(t.max()),
        num = N_NEW
    )

    coords = {"t": t, "t_star": t_new}

    with pm.Model(coords=coords) as model:
        t_ = pm.ConstantData("t", t, dims="obs_id")
        t_star_ = pm.ConstantData("t_star", t_new)
        y_stderr_ = pm.ConstantData("y_stderr", y_stderr, dims="obs_id")

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

        err_norm_dist = pm.Normal.dist(mu=y_stderr_mean, sigma=y_stderr_sd)
        sig = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)

        y_ = gp.marginal_likelihood(
            "y",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=sig
        )

        sem32_dag = pm.model_to_graphviz(model)
        sem32_trace = pm.sample_prior_predictive(samples=N_PPC,
                                                 random_seed=rng_seed,
                                                 idata_kwargs={"dims": {"y": ["t"]}})

        sem32_trace.extend(
            pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=4,
                cores=4,
                random_seed=rng_seed
            )
        )

        f_star = gp.conditional(name="f_star",
                                Xnew=t_new.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=False)

        y_star = gp.conditional(name="y_star",
                                Xnew=t_new.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=False)


        sem32_trace.extend(
            pm.sample_posterior_predictive(
                sem32_trace,
                var_names=["f_star", "y_star"],
                random_seed=rng_seed,
                idata_kwargs={"dims": {"f_star": ["t_star"], "y_star": ["t_star"]}}
            )
        )
        sem32_trace.constant_data = sem32_trace.constant_data.assign_attrs(csv_filename=path_to_csv.stem)

    return sem32_trace, sem32_dag

def fit_gpSE_gpM32(path_to_csv, rng_seed=None):
    """Fit compound GP model: Squared Exponential GP + Matern 3/2 GP."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    y_stderr_mean = np.nanmean(y_stderr)
    y_stderr_sd = np.nanstd(y_stderr)
    t_mingap = np.diff(np.sort(t)).min()

    t_new = np.linspace(
        start=np.floor(t.min()),
        stop=np.ceil(t.max()),
        num = N_NEW
    )

    coords = {"t": t, "t_star": t_new}
    with pm.Model(coords=coords) as model:
        t_ = pm.ConstantData("t", t, dims="obs_id")
        t_star_ = pm.ConstantData("t_star", t_new)
        y_stderr_ = pm.ConstantData("y_stderr", y_stderr, dims="obs_id")

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

        err_norm_dist = pm.Normal.dist(mu=y_stderr_mean, sigma=y_stderr_sd)
        sig = pm.Truncated("sig", err_norm_dist, lower=0, upper=None)

        y_ = gp.marginal_likelihood(
            "y",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=sig
        )

        gpSE_gpM32_dag = pm.model_to_graphviz(model)
        gpSE_gpM32_trace = pm.sample_prior_predictive(samples=N_PPC,
                                                      random_seed=rng_seed,
                                                      idata_kwargs={"dims": {"y": ["t"]}})

        gpSE_gpM32_trace.extend(
            pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=4,
                cores=4,
                random_seed=rng_seed
            )
        )

        f_star_SE = gp_SE.conditional("f_star_SE", Xnew=t_new.reshape(-1,1),
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": sig, "gp": gp})

        f_star_M32 = gp_M32.conditional("f_star_M32", Xnew=t_new.reshape(-1,1),
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": sig, "gp": gp})

        f_star = gp.conditional(name="f_star",
                                Xnew=t_new.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=False)
        
        y_star = gp.conditional(name="y_star",
                                Xnew=t_new.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=True)

        gpSE_gpM32_trace.extend(
            pm.sample_posterior_predictive(
                gpSE_gpM32_trace,
                var_names=["f_star", "f_star_SE", "f_star_M32", "y_star"],
                random_seed=rng_seed,
                idata_kwargs={"dims": {"f_star": ["t_star"], "f_star_SE": ["t_star"], "f_star_M32": ["t_star"], "y_star": ["t_star"]}}
            )
        )
        gpSE_gpM32_trace.constant_data = gpSE_gpM32_trace.constant_data.assign_attrs(csv_filename=path_to_csv.stem)

    return gpSE_gpM32_trace, gpSE_gpM32_dag

def fit_gpSE_gpPer(path_to_csv, standardise_y=False, rng_seed=None):
    """Fit compound GP model: Squared Exponential GP + Periodic GP."""

    lc = pd.read_csv(path_to_csv)
    y = lc["f_peak"].to_numpy()
    y_stderr = lc["f_peak_err"].to_numpy()
    t = lc["mjd"].to_numpy()

    y_sd = np.nanstd(y)
    y_mean = np.nanmean(y)

    if standardise_y:
        # Standardise and translate
        y = (y - y_mean)/y_sd
        y_stderr = y_stderr / y_sd

    #y_stderr_mean = np.nanmean(y_stderr)
    #y_stderr_sd = np.nanstd(y_stderr)
    t_mingap = np.diff(np.sort(t)).min()
    t_range = np.ptp(t)

    t_star = np.linspace(
        start=np.floor(t.min()),
        stop=np.ceil(t.max()),
        num = N_NEW
    )

    coords = {"t": t, "t_star": t_star}
    with pm.Model(coords=coords) as model:
        t_ = pm.ConstantData("t", t, dims="obs_id")
        t_star_ = pm.ConstantData("t_star", t_star)
        y_stderr_ = pm.ConstantData("y_stderr", y_stderr, dims="obs_id")

        std_norm_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
        eta_SE = pm.Truncated("eta_SE", std_norm_dist, lower=0, upper=None)
        eta_Per = pm.Truncated("eta_Per", std_norm_dist, lower=0, upper=None)

        ell_SE = t_mingap + pm.InverseGamma("ell_SE", alpha=3, beta=t_range/2)
        ell_Per = t_mingap + pm.InverseGamma("ell_Per", alpha=3, beta=t_range/2)

        T = pm.Uniform("T", lower=4*t_mingap, upper=t_range/4)
        cov_SE = eta_SE * pm.gp.cov.ExpQuad(input_dim=1, ls=ell_SE)
        gp_SE = pm.gp.Marginal(cov_func=cov_SE)

        cov_Per = eta_Per * pm.gp.cov.Periodic(input_dim=1, period=T, ls=ell_Per)
        gp_Per = pm.gp.Marginal(cov_func=cov_Per)

        gp = gp_SE + gp_Per

        cov_Sigma = pm.gp.cov.WhiteNoise(sigma=y_stderr_)

        y_ = gp.marginal_likelihood(
            "y",
            X=t.reshape(-1,1),
            y=y.reshape(-1,1).flatten(),
            sigma=cov_Sigma
        )

        gpSE_gpPer_dag = pm.model_to_graphviz(model)
        gpSE_gpPer_trace = pm.sample_prior_predictive(samples=N_PPC,
                                                      random_seed=rng_seed,
                                                      idata_kwargs={"dims": {"y": ["t"]}})

        gpSE_gpPer_trace.extend(
            pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=4,
                cores=4,
                random_seed=rng_seed
            )
        )

        f_star_SE = gp_SE.conditional("f_star_SE", Xnew=t_star.reshape(-1,1),
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": cov_Sigma, "gp": gp})

        f_star_Per = gp_Per.conditional("f_star_Per", Xnew=t_star.reshape(-1,1),
                                      given={"X": t.reshape(-1,1), "y": y.reshape(-1,1).flatten(), "sigma": cov_Sigma, "gp": gp})

        f_star = gp.conditional(name="f_star",
                                Xnew=t_star.reshape(-1,1),
                                jitter=1e-6,
                                pred_noise=False)

        #y_star = gp.conditional(name="y_star", Xnew=t_star.reshape(-1,1), jitter=1e-6, pred_noise=True)

        gpSE_gpPer_trace.extend(
            pm.sample_posterior_predictive(
                gpSE_gpPer_trace,
                var_names=["f_star", "f_star_SE", "f_star_Per"],#, "y_star"],
                random_seed=rng_seed,
                idata_kwargs={"dims": {"f_star": ["t_star"], "f_star_SE": ["t_star"], "f_star_Per": ["t_star"]}}#, "y_star": ["t_star"]}}
            )
        )
        gpSE_gpPer_trace.constant_data = gpSE_gpPer_trace.constant_data.assign_attrs(csv_filename=path_to_csv.stem)

    return gpSE_gpPer_trace, gpSE_gpPer_dag
