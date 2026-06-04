import numpy as np
from iminuit import Minuit
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import données_syst as syst
import stat_test as stat
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

def compute_mu_shape(saved_info,systematic, theta0, sigma):

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

    def shifted_templates(theta, systematic):
        """Return theta-dependent signal and background templates for the current nuisance value."""
        delta_s = np.array([syst.get_S(i, systematic, theta) for i in range(len(S_hist))], dtype=float)
        delta_b = np.array([syst.get_B(i, systematic, theta) for i in range(len(B_hist))], dtype=float)
        s_theta = np.maximum(S_hist + delta_s, 0.0)
        b_theta = np.maximum(B_hist + delta_b, 0.0)
        return s_theta, b_theta

    def NLL(mu, theta):
        s_theta, b_theta = shifted_templates(theta, systematic)

        n_obs = np.rint(np.maximum(s_theta + b_theta, 0.0)).astype(int)
        n_pred = np.maximum(mu * s_theta + b_theta, eps)

        return -2.0 * np.sum(poisson.logpmf(n_obs, n_pred)) + (theta - theta0) ** 2 / sigma ** 2

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
    m.fixed["mu"] = True
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
    
    theta_hat = m.values["theta"]
    del_theta_stat = m.errors["theta"]
    del_mu_sys = 0.0

    del_mu_tot = del_mu_stat

    theta_grid = np.linspace(theta0 - 3 * sigma, theta0 + 3 * sigma, 200)

    def profile_theta(theta_value):
        result = minimize_scalar(
            lambda mu_value: NLL(mu_value, theta_value),
            bounds=(0.0, 5.0),
            method="bounded",
        )
        return result.x, result.fun

    profile_values = np.array([profile_theta(theta_i)[1] for theta_i in theta_grid])
    profile_min = np.min(profile_values)
    delta_profile = profile_values - profile_min

    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid, delta_profile, label="Profiled NLL")
    plt.axvline(theta_hat, color="red", linestyle="--", label=r"$	heta_{fit}$")
    plt.axhline(0.0, color="gray", linestyle=":")
    plt.xlabel("theta")
    plt.ylabel("$\\Delta(-2\\log L)$")
    plt.title("Profiled objective vs theta")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return {

        "mu_hat": mu_hat,

        "del_mu_stat": del_mu_stat,

        "del_mu_sys": del_mu_sys,

        "del_mu_tot": del_mu_tot,
        "theta_hat": theta_hat,
        "del_theta_stat": del_theta_stat

    }


#################################################
# PUBLIC INTERFACE
#################################################

def evaluate_shape_analysis(
    model,
    holdout_set,
    number_bins=3,
    threshold=0.7,
):

    saved_info = calculate_saved_info_shape(
        model=model,
        holdout_set=holdout_set,
        number_bins=number_bins,
        threshold=threshold,
    )

    result = compute_mu_shape(
        saved_info,
        systematic="tes",  # Replace with the desired systematic
        theta0=1.0,
        sigma=0.1
    )
    print(f"Estimated mu: {result['mu_hat']}")
    print(f"Statistical uncertainty on mu: {result['del_mu_stat']}")
    print(f"Estimated  tes : {result['theta_hat']}")
    print(f"Statistical uncertainty on  tes : {result['del_theta_stat']}")
    return result

evaluate_shape_analysis(stat.FakeModel(stat.score), stat.holdout)