from bincorrected import calculate_saved_info_shape, compute_mu_shape
from stat_test import FakeModel, S, B, score, holdout

for n_bins in [1, 5, 10, 20]:
    saved_shape = calculate_saved_info_shape(
        FakeModel(score), holdout, number_bins=n_bins
    )
    result = compute_mu_shape(saved_shape)
    print(f"n_bins={n_bins:2d} : mu={result['mu_hat']:.3f} ± {result['del_mu_stat']:.3f}")