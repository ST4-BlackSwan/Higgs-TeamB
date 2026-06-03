import numpy as np
import matplotlib.pyplot as plt
from HiggsML.datasets import download_dataset
from boosted_decision_tree_hyperparameters import BoostedDecisionTreeHyperParameters  # Adaptez le nom de l'import selon votre fichier
from sklearn.metrics import roc_auc_score

print("Chargement des données...")
data = download_dataset("blackSwan_data")
data.load_train_set()
data_set = data.get_train_set()

train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
labels = data_set["labels"].values
weights = data_set["weights"].values

# 2. Définition du dictionnaire d'hyperparamètres
# On définit des listes de valeurs réalistes pour la physique des particules
dictionnaire_parametres = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 5, 10]
}

# Recherche selon la méthode "random search"
NB_ITERATIONS = 20  # Nombre de combinaisons de modèles à tester
historique_scores = []
historique_meilleurs_scores = []

meilleur_score_absolu = -1
meilleurs_parametres_trouves = None

print(f"\nDébut de l'optimisation HPO ({NB_ITERATIONS} itérations)...")

# Boucle d'optimisation
for i in range(1, NB_ITERATIONS + 1):
    # Sélection aléatoire d'une combinaison d'hyperparamètres
    params_choisis = {k: np.random.choice(v) for k, v in dictionnaire_parametres.items()}
    

    bdt_hpo = BoostedDecisionTreeHyperParameters(
        n_estimators=1000,  # Fixé, géré par l'early stopping interne
        max_depth=int(params_choisis['max_depth']),
        learning_rate=params_choisis['learning_rate'],
        subsample=params_choisis['subsample'],
        colsample_bytree=params_choisis['colsample_bytree'],
        min_child_weight=int(params_choisis['min_child_weight']),
        early_stopping_rounds=15
    )
    
    # Entraînement du modèle
    bdt_hpo.fit(train_data, labels, weights=weights)
    
    predictions = bdt_hpo.predict(train_data)
    score_auc = roc_auc_score(labels, predictions, sample_weight=weights)
    
    # Mémorisation du score
    historique_scores.append(score_auc)
    
    if score_auc > meilleur_score_absolu:
        meilleur_score_absolu = score_auc
        meilleurs_parametres_trouves = params_choisis
        
    historique_meilleurs_scores.append(meilleur_score_absolu)
    
    print(f"Itération {i}/{NB_ITERATIONS} | AUC: {score_auc:.4f} | Meilleur AUC historique: {meilleur_score_absolu:.4f}")

print("\n" + "="*50)
print("FIN DE L'OPTIMISATION")
print(f"Meilleur score AUC mémorisé : {meilleur_score_absolu:.5f}")
print("Meilleurs hyperparamètres correspondants :")
for param, valeur in meilleurs_parametres_trouves.items():
    print(f"  - {param}: {valeur}")
print("="*50)

plt.figure(figsize=(10, 6))
plt.plot(range(1, NB_ITERATIONS + 1), historique_scores, marker='o', color='royalblue', alpha=0.6, label="Score de l'itération (Essai)")
plt.step(range(1, NB_ITERATIONS + 1), historique_meilleurs_scores, where='mid', color='crimson', linewidth=2, label="Optimum historique (Meilleur mémorisé)")

plt.title("Courbe de Diagnostic HPO (Optimisation des Hyperparamètres)", fontsize=12, fontweight='bold')
plt.xlabel("Numéro de l'itération (En entrée)", fontsize=10)
plt.ylabel("Score Performance (AUC pondéré)", fontsize=10)
plt.xticks(range(1, NB_ITERATIONS + 1, max(1, NB_ITERATIONS // 10)))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="lower right")

plt.show()