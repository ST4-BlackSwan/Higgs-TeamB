import optuna
import matplotlib.pyplot as plt
from HiggsML.datasets import download_dataset
from boosted_decision_tree_hyperparameters import BoostedDecisionTreeHyperParameters
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load HiggsML dataset
print("Chargement des données...")
data = download_dataset("blackSwan_data")
data.load_train_set()
data_set = data.get_train_set()

train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
labels = data_set["labels"].values
weights = data_set["weights"].values

# Objective function for Optuna optimization loop
def objective(trial):
    # Hyperparameter search space boundaries
    max_depth = trial.suggest_int('max_depth', 3, 9, step=2)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0, step=0.2)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.2)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

    bdt_hpo = BoostedDecisionTreeHyperParameters(
        n_estimators=1000,        # Fixed (managed by early stopping)
        tree_method='hist',       # Fixed (for computing speed)
        device='cuda',
        random_state=31415,       # Fixed (for reproducibility)
        early_stopping_rounds=15, # Fixed (to prevent overfitting)
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight
    )
    
    bdt_hpo.fit(train_data, labels, weights=weights)
    
    predictions = bdt_hpo.predict(train_data)
    score_auc = roc_auc_score(labels, predictions, sample_weight=weights)
    
    return score_auc

# Run Bayesian optimization
NB_ITERATIONS = 50
print(f"\nDébut de l'optimisation Bayésienne avec Optuna ({NB_ITERATIONS} itérations)...")

study = optuna.create_study(direction="maximize")

# Arrays to store results for the diagnostic curve
historique_scores = []
historique_meilleurs_scores = []

for i in range(1, NB_ITERATIONS + 1):
    study.optimize(objective, n_trials=1)
    
    dernier_trial = study.trials[-1]
    score_courant = dernier_trial.value
    meilleur_score_actuel = study.best_value
    
    historique_scores.append(score_courant)
    historique_meilleurs_scores.append(meilleur_score_actuel)
    
    print(f"Itération {i}/{NB_ITERATIONS} | AUC: {score_courant:.4f} | Meilleur AUC historique: {meilleur_score_actuel:.4f}")

# Terminal outputs
print("\n" + "="*50)
print("FIN DE L'OPTIMISATION BAYÉSIENNE")
print(f"Meilleur score AUC mémorisé : {study.best_value:.5f}")
print("Meilleurs hyperparamètres correspondants :")
for param, valeur in study.best_params.items():
    print(f"  - {param}: {valeur}")
print("="*50)

# Plot HPO diagnostic curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, NB_ITERATIONS + 1), historique_scores, marker='o', color='teal', alpha=0.6, label="Score de l'itération (Choix Bayésien)")
plt.step(range(1, NB_ITERATIONS + 1), historique_meilleurs_scores, where='mid', color='crimson', linewidth=2, label="Optimum historique (Meilleur mémorisé)")

plt.title("Courbe de Diagnostic HPO (Optimisation Bayésienne - Optuna)", fontsize=12, fontweight='bold')
plt.xlabel("Numéro de l'itération (En entrée)", fontsize=10)
plt.ylabel("Score Performance (AUC pondéré)", fontsize=10)
plt.xticks(range(1, NB_ITERATIONS + 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="lower right")

plt.savefig("courbe_diagnostic_hpo_bayesian.png")
print("\nGraphique sauvegardé sous le nom 'courbe_diagnostic_hpo_bayesian.png'")
plt.show()