import numpy as np
import matplotlib.pyplot as plt
from bincorrected import calculate_saved_info_shape, compute_mu_shape
from stat_test import FakeModel, S, B, score, holdout

# 1. On crée la figure 
plt.figure(figsize=(10, 6))

# Petites variables pour ajuster automatiquement l'axe X à la fin
x_min, x_max = float('inf'), float('-inf')

for n_bins in [1, 5, 10, 20]:
    saved_shape = calculate_saved_info_shape(
        FakeModel(score), holdout, number_bins=n_bins
    )
    result = compute_mu_shape(saved_shape)
    
    mu_hat = result['mu_hat']
    del_mu = result['del_mu_stat']
    NLL = result['NLL_func']
    
    print(f"n_bins={n_bins:2d} : mu={mu_hat:.3f} ± {del_mu:.3f}")
    
  
    mu_scan = np.linspace(mu_hat - 3 * del_mu, mu_hat + 3 * del_mu, 100)
    nll_scan = np.array([NLL(mu_val) for mu_val in mu_scan])
    delta_nll = nll_scan - np.min(nll_scan)
    
    # On trace la parabole pour ce n_bins précis, ajoutée à la figure commune
    plt.plot(
        mu_scan, 
        delta_nll, 
        linewidth=2, 
        label=rf'$n_{{bins}} = {n_bins}$ ($\hat{{\mu}}={mu_hat:.3f} \pm {del_mu:.3f}$)'
    )
    
  
    x_min = min(x_min, mu_scan[0])
    x_max = max(x_max, mu_scan[-1])


# Une seule ligne rouge horizontale pour la montée de 1
plt.axhline(1.0, color='red', linestyle='--', label=r'Incertitude $1\sigma$ ($\Delta = 1$)')


plt.ylim(0, 4) 
plt.xlim(x_min, x_max)


plt.xlabel(r'Signal strength $\mu$')
plt.ylabel(r'$-2 \Delta \ln \mathcal{L}$')
plt.title('Superposition des profils de vraisemblance selon le nombre de bins')
plt.legend()
plt.grid(True, alpha=0.4)

total_signal_attendu = np.sum(saved_shape["S_hist"])
total_bruit_attendu = np.sum(saved_shape["B_hist"])

print(total_signal_attendu,total_bruit_attendu)

# On affiche le résultat final contenant toutes les courbes
plt.show()

