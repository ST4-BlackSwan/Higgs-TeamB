from networkx import sigma
import numpy as np
from iminuit import Minuit
from scipy.stats import poisson


#################################################
# BUILD SHAPE TEMPLATES
#################################################

def calculate_saved_info_shape(
    model,
    holdout_set,
    number_bins=20,
    threshold=0.5,
):

    # --------------------------------------------------
    # CHANGE #1
    # Predict score and flatten immediately
    # so that both XGB and NN outputs work
    # --------------------------------------------------

    score = model.predict(
        holdout_set["data"]
    ).flatten()

    labels = holdout_set["labels"]
    weights = holdout_set["weights"]

    # --------------------------------------------------
    # CHANGE #2
    # Explicitly keep only signal-like region
    # instead of relying on histogram behaviour
    # --------------------------------------------------

    mask_signal = (labels == 1) & (score > threshold)
    mask_background = (labels == 0) & (score > threshold)

    score_signal = score[mask_signal]
    score_background = score[mask_background]

    weight_signal = weights[mask_signal]
    weight_background = weights[mask_background]

    bins = np.linspace(
        threshold,
        1.0,
        number_bins + 1
    )

    # --------------------------------------------------
    # Weighted templates
    #
    # S_hist[i] = expected signal yield in bin i
    # B_hist[i] = expected background yield in bin i
    # --------------------------------------------------

    S_hist, edges = np.histogram(
        score_signal,
        bins=bins,
        weights=weight_signal
    )

    B_hist, _ = np.histogram(
        score_background,
        bins=bins,
        weights=weight_background
    )

    # --------------------------------------------------
    # CHANGE #3
    # Protect against numerical issues
    # --------------------------------------------------

    S_hist = np.maximum(S_hist, 0.0)
    B_hist = np.maximum(B_hist, 0.0)

    return {

        "bins": edges,

        "S_hist": S_hist,

        "B_hist": B_hist

    }


#################################################
# FIT MU USING BINNED SHAPE LIKELIHOOD
#################################################

def compute_mu_shape(saved_info, theta0, sigma):

    bins = saved_info["bins"]

    S_hist = saved_info["S_hist"]

    B_hist = saved_info["B_hist"]

    eps = 1e-12

    # --------------------------------------------------
    # CHANGE #4
    # Asimov dataset
    #
    # observed = S+B
    # predicted = mu*S+B
    #
    # This follows exactly the project statement.
    # --------------------------------------------------

    def obs(theta):        #ask systematics for the exact function
        
        d_S = np.array([get_S(i, systematic, theta) for i in range(len(S_hist))])
        d_B = np.array([get_B(i, systematic, theta) for i in range(len(B_hist))])
        return np.round(S_hist + d_S + B_hist + d_B)

    def NLL(mu,theta):


        d_S = np.array([get_S(i, systematic, theta) for i in range(len(S_hist))])
        d_B = np.array([get_B(i, systematic, theta) for i in range(len(B_hist))])
#        n_pred = mu * S_hist + B_hist
        n_pred = mu * (S_hist + d_S )+ (B_hist + d_B)
        n_pred = np.maximum(
            n_pred,
            eps
        )
        n_obs = obs(theta)
        
        return -2.0 * np.sum(poisson.logpmf(n_obs, n_pred)) +np.sum((theta-theta0)**2/sigma**2)

    # --------------------------------------------------
    # CHANGE #5
    # Use Minuit for the fit
    # --------------------------------------------------

    m = Minuit(
        NLL,
        mu=1.0,theta=theta0
    )

    # --------------------------------------------------
    # CHANGE #6
    # Physical constraint:
    # mu >= 0
    # --------------------------------------------------

    m.limits["mu"] = (0, None)
    m.limits["theta"] = (theta0 - 3 * sigma, theta0 + 3 * sigma)
    # --------------------------------------------------
    # CHANGE #7
    # MIGRAD:
    # finds mu_hat
    # --------------------------------------------------

    m.migrad()

    # --------------------------------------------------
    # CHANGE #8
    # HESSE:
    # computes sigma_mu
    # --------------------------------------------------

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


#################################################
# PUBLIC INTERFACE
#################################################

def evaluate_shape_analysis(
    model,
    holdout_set,
    number_bins=20,
    threshold=0.5,
):

    saved_info = calculate_saved_info_shape(
        model=model,
        holdout_set=holdout_set,
        number_bins=number_bins,
        threshold=threshold,
    )

    result = compute_mu_shape(
        saved_info, 
        
        theta0=1.0,
        
        sigma=0.03
    )

    return result