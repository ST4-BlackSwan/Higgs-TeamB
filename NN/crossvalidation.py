import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# =====================================================================
# 1. DÉFINITION DU MODÈLE (ARCHITECTURE)
# =====================================================================


def build_model(input_dim):
    """
    Fonction usine (factory) pour générer un nouveau modèle Keras "neuf".
    Indispensable en Cross-Validation pour réinitialiser les poids à chaque pli.
    """
    model = Sequential(
        [
            Dense(10, input_dim=input_dim, activation="relu"),
            Dense(10, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# =====================================================================
# 2. FONCTION D'OPTIMISATION DE LA SIGNIFICATION (METRIQUE HEP)
# =====================================================================


def compute_best_significance(y_true, y_pred, sample_weights, step=0.01):
    """
    Parcourt les seuils de décision pour trouver la coupure optimale
    qui maximise la signification statistique s / sqrt(s + b).
    """
    thresholds = np.arange(0.0, 1.0, step)
    max_sig = 0.0
    best_threshold = 0.5

    signal_mask = y_true == 1
    background_mask = y_true == 0

    for t in thresholds:
        accepted = y_pred >= t
        s = np.sum(sample_weights[signal_mask & accepted])
        b = np.sum(sample_weights[background_mask & accepted])

        sig = s / np.sqrt(s + b) if (s + b) > 0 else 0.0
        if sig > max_sig:
            max_sig = sig
            best_threshold = t

    return best_threshold, max_sig


# =====================================================================
# 3. PIPELINE DE VALIDATION CROISÉE STRATIFIÉE
# =====================================================================


def run_stratified_cross_validation(X, y, weights, n_splits=5, epochs=5):
    """
    Orchestre la validation croisée stratifiée sur le dataset complet.
    Gère le scaling et l'ajustement des poids de manière étanche à chaque pli.
    """
    # Conversion en numpy arrays pour faciliter l'indexation par masque
    X_arr = np.array(X)
    y_arr = np.array(y)
    w_arr = np.array(weights)

    # Initialisation du découpeur stratifié
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=31415)

    # Tableaux pour stocker les scores de chaque pli
    fold_significances = []
    fold_thresholds = []

    print("=" * 60)
    print(f"DÉMARRAGE DE LA VALIDATION CROISÉE ({n_splits} PLIS)")
    print("=" * 60)

    # Boucle principale de la CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        print(f"\n--- Entraînement du Pli {fold}/{n_splits} ---")

        # Split des données pour ce pli
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        w_train, w_val = w_arr[train_idx].copy(), w_arr[val_idx].copy()

        # --- ÉTAPE CRUCIALE 1 : Normalisation étanche (Pas de data leakage) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # --- ÉTAPE CRUCIALE 2 : Égalisation des poids sur le Train Set de ce pli ---
        class_weights_train = [w_train[y_train == 0].sum(), w_train[y_train == 1].sum()]
        max_weight = max(class_weights_train)

        for i in range(2):
            w_train[y_train == i] *= max_weight / class_weights_train[i]

        # --- ÉTAPE CRUCIALE 3 : Réinitialisation du modèle ---
        model = build_model(input_dim=X_train.shape[1])

        # Entraînement sur le pli actuel
        model.fit(
            X_train_scaled,
            y_train,
            sample_weight=w_train,
            epochs=epochs,
            verbose=0,  # 0 pour éviter d'inonder la console
            batch_size=256,
        )

        # Prédiction sur la portion de validation du pli
        y_pred_val = model.predict(X_val_scaled, verbose=0).flatten()

        # Calcul de la métrique physique sur le pli
        best_t, max_sig = compute_best_significance(y_val, y_pred_val, w_val)

        fold_thresholds.append(best_t)
        fold_significances.append(max_sig)

        print(
            f"Pli {fold} Terminé | Seuil Optimal: {best_t:.2f} | Signification: {max_sig:.3f} sigma"
        )

    # --- BILAN DE LA VALIDATION CROISÉE ---
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX DE LA VALIDATION CROISÉE")
    print("=" * 60)
    print(
        f"Signification Moyenne : {np.mean(fold_significances):.3f} ± {np.std(fold_significances):.3f} sigma"
    )
    print(f"Seuil de coupure Moyen: {np.mean(fold_thresholds):.2f}")
    print("=" * 60)

    return fold_significances


# =====================================================================
# 4. CHARGEMENT ET NETTOYAGE (TON CODE CORRIGÉ)
# =====================================================================

try:
    from HiggsML.datasets import download_dataset
except ImportError:
    pass


def load_and_clean_blackswan():
    """Télécharge et nettoie le dataset BlackSwan"""
    dataset = download_dataset("blackSwan_data")
    dataset.load_train_set()
    df_raw = dataset.get_train_set()

    if not isinstance(df_raw, pd.DataFrame):
        df_raw = df_raw.df if hasattr(df_raw, "df") else pd.DataFrame(df_raw)

    target = df_raw["labels"].copy()
    if target.dtype == "object" or target.dtype == "string":
        target = target.map({"s": 1, "b": 0, "S": 1, "B": 0, 1: 1, 0: 0})

    weights = df_raw["weights"].copy()

    feature_mapping = {
        "PRI_met": "PRI_met",
        "PRI_met_phi": "PRI_met_phi",
        "PRI_lep_pt": "PRI_lep_pt",
        "PRI_lep_phi": "PRI_lep_phi",
        "PRI_had_pt": "PRI_had_pt",
        "PRI_had_phi": "PRI_had_phi",
    }

    available_features = [
        col for col in feature_mapping.keys() if col in df_raw.columns
    ]
    features = (
        df_raw[available_features].copy()
        if available_features
        else df_raw.drop(
            columns=["labels", "detailed_labels", "weights"], errors="ignore"
        )
    )
    features = features.fillna(features.mean())

    return features, target, weights


# =====================================================================
# 5. BLOC D'ÉCUTION
# =====================================================================
if __name__ == "__main__":
    # 1. Chargement des données brutes
    features, target, weights = load_and_clean_blackswan()

    # 2. Correction et ajustement global des poids du Test Set avant toute coupure.
    # Ici, si tu gardes 100% du jeu pour la CV, pas besoin de multiplier w_test.
    # Mais si tu voulais appliquer un facteur global, tu le ferais ici, hors de la boucle.

    # 3. Lancement de la validation croisée à 5 plis
    scores_cv = run_stratified_cross_validation(
        features, target, weights, n_splits=5, epochs=10
    )
