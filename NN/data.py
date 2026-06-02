import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TENTATIVE D'IMPORTATION DE HIGGSML (avec fallback si exécuté en local)
try:
    from HiggsML.datasets import download_dataset
except ImportError:
    print("Attention: La librairie HiggsML n'est pas installée localement.")

def load_and_clean_blackswan():
    """
    Télécharge, charge et nettoie les données spécifiques au challenge.
    Version corrigée avec les colonnes exactes du dataset BlackSwan (labels, weights, PRI_*).
    """
    # 1. Téléchargement et chargement via l'API HiggsML
    dataset = download_dataset("blackSwan_data")
    dataset.load_train_set()
    df_raw = dataset.get_train_set()
    
    # Conversion en DataFrame standard si nécessaire
    if not isinstance(df_raw, pd.DataFrame):
        if hasattr(df_raw, 'df'):
            df_raw = df_raw.df
        else:
            df_raw = pd.DataFrame(df_raw)
            
    # 2. Extraction de la cible ('labels') et conversion automatique en 0/1 si c'est du texte
    target = df_raw['labels'].copy()
    if target.dtype == 'object' or target.dtype == 'string':
        target = target.map({'s': 1, 'b': 0, 'S': 1, 'B': 0, 1: 1, 0: 0})

    # 3. Extraction des poids ('weights')
    weights = df_raw['weights'].copy()
    
    # 4. Sélection des features pertinentes pour l'exercice
    # On mappe les variables demandées vers leurs équivalents exacts dans ton dataset :
    # "met_et" -> "PRI_met", "lep_pt_0" -> "PRI_lep_pt", etc.
    feature_mapping = {
        "PRI_met": "PRI_met",
        "PRI_met_phi": "PRI_met_phi",
        "PRI_lep_pt": "PRI_lep_pt",
        "PRI_lep_phi": "PRI_lep_phi",
        "PRI_had_pt": "PRI_had_pt",
        "PRI_had_phi": "PRI_had_phi"
    }
    
    available_features = [col for col in feature_mapping.keys() if col in df_raw.columns]
    
    if len(available_features) == 0:
        print("Warning: Les features spécifiques ciblées sont introuvables. Utilisation de toutes les variables DER/PRI.")
        # Sécurité : On prend tout sauf les colonnes de cibles et de poids
        features = df_raw.drop(columns=['labels', 'detailed_labels', 'weights'], errors='ignore')
    else:
        features = df_raw[available_features].copy()
    
    # Remplacement des valeurs manquantes (NaN ou indéfinies) par la moyenne
    features = features.fillna(features.mean()) 
    
    return features, target, weights

def prepare_datasets(features, target, weights, train_size=0.75, random_seed=31415):
    """
    Sépare en Train/Test, applique le StandardScaler et normalise les poids.
    """
    # Fixer la seed pour la reproductibilité
    np.random.seed(random_seed)
    
    # 1. Séparation Train / Test
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        features, target, weights, train_size=train_size, random_state=random_seed
    )
    
    # conversion en copies explicites pour éviter les SettingWithCopyWarning
    w_train = w_train.copy()
    w_test = w_test.copy()

    # 2. Standardisation (Moyenne 0, Variance 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Renormalisation des poids (Code d'égalisation du notebook original)
    class_weights_train = [w_train[y_train == 0].sum(), w_train[y_train == 1].sum()]
    max_weight = max(class_weights_train)
    
    for i in range(2):
        # Égalise la masse du Background (0) et du Signal (1) sur le Train set
        w_train.loc[y_train == i] *= max_weight / class_weights_train[i]
        # Compense l'effet de l'échantillonnage sur le Test set
        w_test.loc[y_test == i] *= 1 / (1 - train_size)
        
    return X_train_scaled, X_test_scaled, y_train, y_test, w_train, w_test, scaler

def plot_distributions(features, target, weights):
    """
    Génère les graphiques de distribution pour le Signal (S) et le Background (B)
    """
    plt.figure()
    # Histogramme pour le Background (B) en bleu
    ax = features[target == 0].hist(weights=weights[target == 0], figsize=(15, 12), color='b', alpha=0.5, density=True, label="B")
    ax = ax.flatten()[:features.shape[1]] 
    
    # Histogramme pour le Signal (S) en rouge sur le même graphique
    features[target == 1].hist(weights=weights[target == 1], figsize=(15, 12), color='r', alpha=0.5, density=True, ax=ax, label="S")
    
    for a in ax:
        a.legend(["B", "S"], loc="best")
    
    plt.show()

# Bloc d'exécution principal
if __name__ == "__main__":
    print("--- 1. Chargement et Nettoyage des données ---")
    features, target, weights = load_and_clean_blackswan()
    
    print("--- 2. Affichage des distributions (optionnel) ---")
    # Décommentez la ligne ci-dessous si vous voulez voir les plots au lancement du script
    plot_distributions(features, target, weights)
    
    print("--- 3. Séparation, Scaler et Renormalisation des poids ---")
    X_train, X_test, y_train, y_test, w_train, w_test, scaler = prepare_datasets(features, target, weights)
    
    print("\n[Vérification des Shapes]")
    print(f"Train features: {X_train.shape} | Test features: {X_test.shape}")
    print(f"Somme des poids Train - Bkg: {w_train[y_train==0].sum():.2f} | Signal: {w_train[y_train==1].sum():.2f}")
    
    