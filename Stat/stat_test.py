import numpy as np
from statistical_analysis import calculate_saved_info, compute_mu, compute_mub
from bincorrected import  calculate_saved_info_shape, compute_mu_shape
S = 500
B = 10_000

class FakeModel:
    def __init__(self, score):
        self.score = score
    def predict(self, X):
        return self.score
    
np.random.seed(42)
score  = np.concatenate([
    np.clip(np.random.normal(0.7, 0.1, S), 0, 1),
    np.clip(np.random.normal(0.3, 0.15, B), 0, 1)
])
label  = np.concatenate([np.ones(S), np.zeros(B)])
weight = np.ones(S + B)
holdout = {"data": score, "labels": label, "weights": weight}

def test1():
    """ Asimov parfait : attendu mu = 1.0 exactement """
    score  = np.concatenate([np.ones(S), np.zeros(B)])
    label  = np.concatenate([np.ones(S), np.zeros(B)])
    weight = np.ones(S + B)
    holdout = {"data": score, "labels": label, "weights": weight}
    saved = calculate_saved_info(FakeModel(score), holdout)
    saved_shape = calculate_saved_info_shape(FakeModel(score), holdout) 
    print("TEST 1 - Attendu mu=1.0")
    print(f"Counting : mu={compute_mu(score, weight, saved)['mu_hat']:.3f} ± {compute_mu(score, weight, saved)['del_mu_stat']:.3f}")
    print(f"Likelihood : mu={compute_mub(score, weight, saved)['mu_hat']:.3f} ± {compute_mub(score, weight, saved)['del_mu_stat']:.3f}")
    print(f"Shape : mu={compute_mu_shape(saved_shape)['mu_hat']:.3f} ± {compute_mu_shape(saved_shape)['del_mu_stat']:.3f}") 

def test2():
    """ Scores mélangés : attendu mu ≈ 1.0 avec δμ > 0 """
    np.random.seed(42)
    score  = np.concatenate([
        np.clip(np.random.normal(0.7, 0.1, S), 0, 1),
        np.clip(np.random.normal(0.3, 0.15, B), 0, 1)
    ])
    label  = np.concatenate([np.ones(S), np.zeros(B)])
    weight = np.ones(S + B)
    holdout = {"data": score, "labels": label, "weights": weight}
    saved = calculate_saved_info(FakeModel(score), holdout)
    print("TEST 2 - Attendu mu≈1.0")
    print(f"Counting : mu={compute_mu(score, weight, saved)['mu_hat']:.3f} ± {compute_mu(score, weight, saved)['del_mu_stat']:.3f}")
    print(f"Likelihood : mu={compute_mub(score, weight, saved)['mu_hat']:.3f} ± {compute_mub(score, weight, saved)['del_mu_stat']:.3f}")

def test3():
    """ mu=0.5 imposé : on réduit le signal de moitié """
    score_holdout  = np.concatenate([np.ones(S), np.zeros(B)])
    label_holdout  = np.concatenate([np.ones(S), np.zeros(B)])
    weight_holdout = np.ones(S + B)
    holdout = {"data": score_holdout, "labels": label_holdout, "weights": weight_holdout}
    saved = calculate_saved_info(FakeModel(score_holdout), holdout)
    score_test  = np.concatenate([np.ones(int(0.5*S)), np.zeros(B)])
    weight_test = np.ones(int(0.5*S) + B)
    print("TEST 3 - Attendu mu≈0.5")
    print(f"Counting : mu={compute_mu(score_test, weight_test, saved)['mu_hat']:.3f} ± {compute_mu(score_test, weight_test, saved)['del_mu_stat']:.3f}")
    print(f"Likelihood : mu={compute_mub(score_test, weight_test, saved)['mu_hat']:.3f} ± {compute_mub(score_test, weight_test, saved)['del_mu_stat']:.3f}")

def test4():
    """ mu=1.5 imposé : on augmente le signal """
    score_holdout  = np.concatenate([np.ones(S), np.zeros(B)])
    label_holdout  = np.concatenate([np.ones(S), np.zeros(B)])
    weight_holdout = np.ones(S + B)
    holdout = {"data": score_holdout, "labels": label_holdout, "weights": weight_holdout}
    saved = calculate_saved_info(FakeModel(score_holdout), holdout)
    score_test  = np.concatenate([np.ones(int(1.5*S)), np.zeros(B)])
    weight_test = np.ones(int(1.5*S) + B)
    print("TEST 4 - Attendu mu≈1.5")
    print(f"Counting : mu={compute_mu(score_test, weight_test, saved)['mu_hat']:.3f} ± {compute_mu(score_test, weight_test, saved)['del_mu_stat']:.3f}")
    print(f"Likelihood : mu={compute_mub(score_test, weight_test, saved)['mu_hat']:.3f} ± {compute_mub(score_test, weight_test, saved)['del_mu_stat']:.3f}")