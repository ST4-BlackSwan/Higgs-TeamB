import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def plot_likelihood_profile(saved_info):
    beta = saved_info["beta"]
    gamma = saved_info["gamma"]
    
    # Asimov dataset: on observe exactement ce qu'on attend pour mu = 1
    n_obs = 1.0 * gamma + beta 

    # Définition du -2 ln(L)
    def NLL(mu):
        n_pred = mu * gamma + beta
        # On utilise logpmf de scipy.stats pour avoir le ln(Poisson)
        return -2 * poisson.logpmf(n_obs, n_pred)

    # Création d'une grille de valeurs pour mu (par exemple de 0.0 à 2.0)
    mu_vals = np.linspace(0.0, 2.0, 100)
    nll_vals = np.array([NLL(m) for m in mu_vals])
    
    # On soustrait le minimum pour avoir Delta(-2lnL) qui commence à 0
    min_nll = np.min(nll_vals)
    delta_nll = nll_vals - min_nll
    
    # Extraction naïve des bornes à 1 sigma (où Delta NLL approx 1)
    # C'est juste pour le plot, iminuit le fait plus proprement
    mu_hat = mu_vals[np.argmin(delta_nll)]

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(mu_vals, delta_nll, label=r'$-2 \Delta \ln \mathcal{L}(\mu)$', color='blue', lw=2)
    
    # Ligne horizontale à +1 pour l'incertitude à 1 sigma
    plt.axhline(1.0, color='red', linestyle='--', label=r'$1\sigma$ interval ($\Delta = 1$)')
    plt.axvline(mu_hat, color='gray', linestyle=':', label=rf'Best fit $\hat{{\mu}} \approx {mu_hat:.2f}$')
    
    plt.ylim(0, 5) # On zoome sur le bas de la parabole
    plt.xlabel(r'$\mu$ (Signal Strength)', fontsize=14)
    plt.ylabel(r'$-2 \Delta \ln \mathcal{L}$', fontsize=14)
    plt.title('Likelihood Profile (Counting Experiment)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()

# Utilisation (en supposant que tu as déjà calculé saved_info)
# plot_likelihood_profile(saved_info)

# 1. On crée des fausses données (par exemple 100 bruits de fond, 50 signaux)
dummy_saved_info = {
    "beta": 200.0,  
    "gamma": 5.0   
}

# 2. On lance la fonction
plot_likelihood_profile(dummy_saved_info)