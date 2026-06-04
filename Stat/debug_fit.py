import sys
import numpy as np
sys.path.insert(0, 'Stat')
import stat_test as st
import données_syst as syst
from iminuit import Minuit
from scipy.stats import poisson

saved = __import__('bincorrected', fromlist=['calculate_saved_info_shape']).calculate_saved_info_shape(
    st.FakeModel(st.score), st.holdout, number_bins=3, threshold=0.7
)
S_hist = saved['S_hist']
B_hist = saved['B_hist']

def shifted_templates(theta, systematic):
    delta_s = np.array([syst.get_S(i, systematic, theta) for i in range(len(S_hist))], dtype=float)
    delta_b = np.array([syst.get_B(i, systematic, theta) for i in range(len(B_hist))], dtype=float)
    s_theta = np.maximum(S_hist + delta_s, 0.0)
    b_theta = np.maximum(B_hist + delta_b, 0.0)
    return s_theta, b_theta

def NLL(mu, theta):
    s_theta, b_theta = shifted_templates(theta, 'tes')
    n_obs = np.maximum(s_theta + b_theta, 0.0)
    n_pred = np.maximum(mu * s_theta + b_theta, 1e-12)
    val = -2.0 * np.sum(poisson.logpmf(n_obs, n_pred)) + (theta - 1.0) ** 2 / 0.1 ** 2
    print('theta=', theta, 'obs min/max=', n_obs.min(), n_obs.max(), 'pred min/max=', n_pred.min(), n_pred.max(), 'val=', val)
    return val

m = Minuit(NLL, mu=1.0, theta=1.0)
m.limits['mu'] = (0, None)
m.limits['theta'] = (1.0 - 3 * 0.1, 1.0 + 3 * 0.1)
print('MIGRAD result:', m.migrad())
print('HESSE result:', m.hesse())
print('ERRORS:', m.errors)
print('VALUES:', m.values)
