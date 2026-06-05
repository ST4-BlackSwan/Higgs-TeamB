#from networkx import sigma
import numpy as np
from iminuit import Minuit
from scipy.stats import poisson
from stat_test import FakeModel, S,B, score, holdout 
import matplotlib.pyplot as plt


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
        return np.round(
            S_hist + B_hist
        )

    def NLL(mu,theta):

        n_pred = mu * S_hist + B_hist

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
        saved_info
    )

    return result



def plot_mu_shape(saved_info, theta0, sigma):
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
        return np.round(
            S_hist + B_hist
        )

    def NLL(mu,theta):

        n_pred = mu * S_hist + B_hist

        n_pred = np.maximum(
            n_pred,
            eps
        )
        n_obs = obs(theta)
        
        return -2.0 * np.sum(poisson.logpmf(n_obs, n_pred)) +np.sum((theta-theta0)**2/sigma**2)
    
    theta_hat = []
    min_values = []
    del_theta_stat = []   # ✅ initialisée
    del_theta_tot = []    # ✅ initialisée

    for mu_val in np.linspace(0.8, 1.2, 100):
        m = Minuit(NLL, mu=mu_val, theta=theta0)
        m.fixed["mu"] = True
        m.limits["mu"] = (0, None)
        m.limits["theta"] = (theta0 - 3 * sigma, theta0 + 3 * sigma)
        m.migrad()
        m.hesse()

        theta_hat.append(m.values["theta"])
        min_values.append(NLL(mu_val, m.values["theta"]))   # ✅ 2 arguments
        del_theta_stat.append(m.errors["theta"])
        del_theta_tot.append(m.errors["theta"])             # ✅ remplie

    del_theta_sys = 0.0

    min_nll = min(min_values)
    min_values_shifted = [v - min_nll for v in min_values]  # ✅ recentre à 0

    plt.plot(np.linspace(0, 2, 100), min_values_shifted)
    plt.axhline(y=1, color='r', linestyle='--', label=r'$\Delta NLL = 1$')
    plt.title(r"Profil de vraisemblance en fonction de $\mu$")
    plt.ylabel(r"$\min_{\theta} \left(-2 \ln \mathcal{L}(\mu, \theta)\right)$")
    plt.xlabel(r"$\mu$ (Signal Strength)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

    return {
        "theta_hat": theta_hat,
        "del_theta_stat": del_theta_stat,
        "del_theta_sys": del_theta_sys,
        "del_theta_tot": del_theta_tot,
    }
    
plot_mu_shape(calculate_saved_info_shape(
    FakeModel(score),
    holdout,
    number_bins=20,
    threshold=0.7,
), 1, 0.1)