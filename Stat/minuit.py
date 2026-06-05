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
    print(f"n_bins={n_bins:2d} : mu={result['mu_hat']:.3f} ± {result['del_mu_stat']:.3f}")
mu_values = np.linspace(0.5 , 1.5, 200)

for n_bins in [1, 5, 10, 20]:
    saved_shape = calculate_saved_info_shape(FakeModel(score), holdout, number_bins=n_bins)
    result = compute_mu_shape(saved_shape)
    nll_values = [result["NLL_func"](mu) for mu in mu_values]
    plt.plot(mu_values, nll_values - min(nll_values), label=f"n_bins={n_bins}")

plt.xlabel(r'$\mu$')
plt.ylabel(r'$-2\log\mathcal{L}(\mu)$')
plt.hlines(1, 0, 2, linestyle='--', color='gray')
plt.legend()
plt.grid()
plt.show()
