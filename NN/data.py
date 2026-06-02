#%pip install HiggsML 0.1.5
from HiggsML.datasets import download_dataset
data = download_dataset( "blackSwan_data")
# load train set
data.load_train_set()
data_set = data.get_train_set()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def clean_blackswan_data(data_set):
    """
    Prend le dataset brut fourni par la bibliothèque HiggsML de Black Swan,
    convertit les labels, et nettoie les valeurs manquantes -999.0.
    """
    # 1. Extraction des données brutes de l'objet Black Swan
    # (Selon la structure standard des kits HiggsML)
    X_raw = data_set['data'] if isinstance(data_set, dict) else data_set.data
    y_raw = data_set['labels'] if isinstance(data_set, dict) else data_set.labels
    weights = data_set['weights'] if isinstance(data_set, dict) else data_set.weights
    event_ids = data_set['event_ids'] if isinstance(data_set, dict) else data_set.event_ids

    # 2. Transformation automatique des labels s/b -> 1/0
    # On utilise numpy pour aller vite sans boucle for
    y = np.where(y_raw == 's', 1, 0)
    
    # 3. Remplacement automatique des -999.0 par la moyenne de la colonne
    imputer = SimpleImputer(missing_values=-999.0, strategy='mean')
    X = imputer.fit_transform(X_raw)

    return X, y, weights, event_ids

def split_train_validation(X, y, weights):
    """Découpage automatique Train/Validation"""
    return train_test_split(
        X, y, weights, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )