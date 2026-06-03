import numpy as np
from HiggsML.systematics import systematics
from iminuit import Minuit
from scipy.stats import poisson

"""
Task 1a : Counting Estimator
1.write the saved_info dictionary such that it contains the following keys
    1. beta
    2. gamma
2. Estimate the mu using the formula
    mu = (sum(score * weight) - beta) / gamma
3. return the mu and its uncertainty

Task 1b : Stat-Only Likelihood Estimator
1. Modify the estimation of mu such that it uses the likelihood function
    1. Write a function for the likelihood function which profiles over mu
    2. Use Minuit to minimize the NLL

Task 2 : Systematic Uncertainty
1. substitute the beta and gamma with the tes_fit and jes_fit functions
2. Write a function to likelihood function which profiles over mu, tes and jes
3. Use Minuit to minimize the NLL
4. return the mu and its uncertainty

"""


def calculate_saved_info(model, holdout_set):

    score = model.predict(holdout_set["data"])

    print("score shape before threshold", score.shape)

    score = score.flatten() > 0.5
    score = score.astype(int)

    label = holdout_set["labels"]

    print("score shape after threshold", score.shape)

    gamma = np.sum(holdout_set["weights"] * score * label)

    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {"beta": beta, "gamma": gamma}

    print("saved_info", saved_info)

    return saved_info


def compute_mu(score, weight, saved_info):

    score = score.flatten() > 0.5
    score = score.astype(int)

    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.0 * mu)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def compute_mub(score, weight, saved_info):

    beta = saved_info["beta"]
    gamma = saved_info["gamma"]

    score = score.flatten() > 0.5
    score = score.astype(int)
    n_obs = np.sum(score * weight)

    def NLL(mu):
        n_pred = mu * gamma + beta
        return -2 * poisson.logpmf(n_obs, n_pred)

    m = Minuit(NLL, mu=1.0)
    m.limits["mu"] = (0, None)
    m.migrad()

    mu_hat = m.values["mu"]
    del_mu_stat = m.errors["mu"]
    del_mu_sys = abs(0.0 * mu_hat)
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu_hat,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }