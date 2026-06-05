import numpy as np
from HiggsML.systematics import systematics
from iminuit import Minuit
from scipy.stats import poisson


def binning_signal(holdout_set, model, number_bins, threshold):
    bins = np.linspace(threshold, 1, number_bins + 1)
    set_signal = holdout_set["data"][holdout_set["labels"] == 1]
    set_background = holdout_set["data"][holdout_set["labels"] == 0]
    score_signal = model.predict(set_signal)
    score_background = model.predict(set_background)
    hs, edges = np.histogram(
        score_signal,
        bins=bins,
        weights=holdout_set["weights"][holdout_set["labels"] == 1],
    )
    hb, edges = np.histogram(
        score_background,
        bins=bins,
        weights=holdout_set["weights"][holdout_set["labels"] == 0],
    )
    return hs, hb, edges


def compute_mu_shape(
    score,
    weight,
    saved_info,
):

    score = score.flatten()

    bins = saved_info["bins"]
    S_hist = saved_info["S_hist"]
    B_hist = saved_info["B_hist"]
    score_min = None  #
    mask = score > score_min  # le seuil de tri des signaux

    score = score[mask]
    weight = weight[mask]

    n_obs = np.histogram(score, bins=bins, weights=weight)[0]

    eps = 1e-12

    def NLL(mu):

        n_pred = mu * S_hist + B_hist

        n_pred = np.maximum(n_pred, eps)

        return -2 * np.sum(poisson.logpmf(np.round(n_obs), n_pred))

    m = Minuit(NLL, mu=1.0)

    m.limits["mu"] = (0, None)

    m.migrad()
    m.hesse()

    mu_hat = m.values["mu"]

    del_mu_stat = m.errors["mu"]

    del_mu_sys = 0.0

    del_mu_tot = del_mu_stat

    return {
        "mu_hat": mu_hat,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }
