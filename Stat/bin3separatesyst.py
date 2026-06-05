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

#def compute_mu_shape(saved_info,systematic, theta0, sigma):
def compute_mu_shape(saved_info, theta0_tes=1.0, theta0_jes=1.0, theta0_smet=0.0):
    
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

    def shifted_templates(tes, jes, smet):
        delta_s = (
            np.array([syst.get_S(i, "tes",   tes)   for i in range(len(S_hist))]) +
            np.array([syst.get_S(i, "jes",   jes)   for i in range(len(S_hist))]) +
            np.array([syst.get_S(i, "smet",  smet)  for i in range(len(S_hist))])
        # bnorm pas encore disponible
        )
        delta_b = (
            np.array([syst.get_B(i, "tes",   tes)   for i in range(len(B_hist))]) +
            np.array([syst.get_B(i, "jes",   jes)   for i in range(len(B_hist))]) +
            np.array([syst.get_B(i, "smet",  smet)  for i in range(len(B_hist))])
        # bnorm pas encore disponible
        )
        s_theta = np.maximum(S_hist + delta_s, 0.0)
        b_theta = np.maximum(B_hist + delta_b, 0.0)
        return s_theta, b_theta
    def log_poisson(n, lam):
        return n * np.log(lam) - lam 
    def NLL(mu, tes, jes, smet):
        s_theta, b_theta = shifted_templates(tes, jes, smet)
        
        n_obs = s_theta + b_theta
        n_pred = np.maximum(mu * s_theta + b_theta, eps)
        return (
            -2.0 * np.sum(log_poisson(n_obs, n_pred))
            + (tes  - theta0_tes)**2  / 0.03**2
            + (jes  - theta0_jes)**2  / 0.03**2
            + (smet - theta0_smet)**2 / 3.0**2
        )
    
    # --------------------------------------------------
    # CHANGE #5
    # Use Minuit for the fit
    # --------------------------------------------------

    m = Minuit(NLL, mu=1.0, tes=theta0_tes, jes=theta0_jes, smet=theta0_smet)
    
    m.limits["mu"]   = (0, None)
    m.limits["tes"]  = (theta0_tes  - 3 * 0.03, theta0_tes  + 3 * 0.03)   # ±3*3%
    m.limits["jes"]  = (theta0_jes  - 3 * 0.03, theta0_jes  + 3 * 0.03)   # ±3*3%
    m.limits["smet"] = (theta0_smet - 1 * 3.0,  theta0_smet + 1 * 3.0)   

    m.fixed["mu"]=True


    m.migrad()


    m.hesse()

    mu_hat     = m.values["mu"]
    del_mu_stat = m.errors["mu"]

    tes_hat    = m.values["tes"]
    jes_hat    = m.values["jes"]
    smet_hat   = m.values["smet"]

    del_tes    = m.errors["tes"]
    del_jes    = m.errors["jes"]
    del_smet   = m.errors["smet"]
    del_mu_sys = 0.0

    del_mu_tot = del_mu_stat

    return {
        "mu_hat"      : mu_hat,
        "del_mu_stat" : del_mu_stat,
        "del_mu_sys"  : del_mu_sys,
        "del_mu_tot"  : del_mu_tot,
        "tes_hat"     : tes_hat,
        "jes_hat"     : jes_hat,
        "smet_hat"    : smet_hat,
        "del_tes"     : del_tes,
        "del_jes"     : del_jes,
        "del_smet"    : del_smet,
        "NLL"         : NLL,
    }


#################################################
# PUBLIC INTERFACE
#################################################

def evaluate_shape_analysis(
    model,
    holdout_set,
    number_bins=10,
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
        theta0_tes=1.0,
        theta0_jes=1.0,
        theta0_smet=0.0
    )
    
    mu_grid = np.linspace(0.5, 1.5, 500)
    nll_tes = []
    nll_jes = []
    nll_smet = []   
    NLL = result["NLL"]
    for mu_val in mu_grid:
        # --- TES variation (JES and SMET fixed) ---
        m = Minuit(NLL, mu=mu_val, tes=1.0, jes=1.0, smet=0.0)
        
        # Explicitly set your boundaries so Migrad doesn't drift into unphysical space
        m.limits["tes"]  = (0.91, 1.09)
        m.limits["jes"]  = (0.91, 1.09)
        m.limits["smet"] = (-3.0, 3.0)
        
        # Enforce your exact rule: Only TES is allowed to minimize
        m.fixed["mu"]   = True   # Fixed at grid point
        m.fixed["tes"]  = False  # Floating/Minimizing
        m.fixed["jes"]  = True   # Fixed at 1.0
        m.fixed["smet"] = True   # Fixed at 0.0
        
        m.migrad()
        nll_tes.append(m.fval)

        # --- JES variation (TES and SMET fixed) ---
        m = Minuit(NLL, mu=mu_val, tes=1.0, jes=1.0, smet=0.0)
        m.limits["tes"], m.limits["jes"], m.limits["smet"] = (0.91, 1.09), (0.91, 1.09), (-3.0, 3.0)
        m.fixed["mu"]   = True
        m.fixed["tes"]  = True   # Fixed at 1.0
        m.fixed["jes"]  = False  # Floating/Minimizing
        m.fixed["smet"] = True   # Fixed at 0.0
        m.migrad()
        nll_jes.append(m.fval)

        # --- SMET variation (TES and JES fixed) ---
        m = Minuit(NLL, mu=mu_val, tes=1.0, jes=1.0, smet=0.0)
        m.limits["tes"], m.limits["jes"], m.limits["smet"] = (0.5, 1.5), (0.5, 1.5), (-3.0, 3.0)
        m.fixed["mu"]   = True
        m.fixed["tes"]  = True   # Fixed at 1.0
        m.fixed["jes"]  = True   # Fixed at 1.0
        m.fixed["smet"] = False  # Floating/Minimizing
        m.migrad()
        nll_smet.append(m.fval)
    
    #print(nll_tes)
    nll_tes = np.array(nll_tes) - min(nll_tes)
    nll_jes = np.array(nll_jes) - min(nll_jes)
    nll_smet = np.array(nll_smet) - min(nll_smet)

    plt.grid()
    plt.plot(mu_grid, nll_tes, label="NLL profilée (1 syst: tes)")
    plt.plot(mu_grid, nll_jes, label="NLL profilée (1 syst: jes)")
    plt.plot(mu_grid, nll_smet, label="NLL profilée (1 syst: smet)")
    plt.hlines(1, 0.5, 1.5, linestyle='--', color='gray')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$-2\log\mathcal{L}(\mu)$')
    plt.legend()
    plt.show()
    print(f"Estimated mu: {result['mu_hat']}")
    print(f"Statistical uncertainty on mu: {result['del_mu_stat']}")
    print(f"Estimated tes  : {result['tes_hat']} ± {result['del_tes']}")
    print(f"Estimated jes  : {result['jes_hat']} ± {result['del_jes']}")
    print(f"Estimated smet : {result['smet_hat']} ± {result['del_smet']}")
    return result

evaluate_shape_analysis(stat.FakeModel(stat.score), stat.holdout)