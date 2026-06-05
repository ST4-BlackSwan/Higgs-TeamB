import os
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from HiggsML.datasets import download_dataset

data = download_dataset("blackSwan_data")

data.load_train_set()
data_set = data.get_train_set()

# 1. On localise le dossier où se trouvent nos fichiers sauvegardés

# (os.path.abspath(__file__) garantit qu'on cherche dans le dossier du script)

model_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in locals()
    else os.getcwd()
)

# 2. ON CHARGE LE SCALER ET LE MODÈLE

# On charge l'outil de normalisation

scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

# On charge le cerveau du réseau de neurones complet

model = load_model(os.path.join(model_dir, "model.keras"))

print("🚀 Modèle et Scaler pré-entraînés chargés avec succès ! Prêt pour l'inférence.")


# --- COMMENT L'UTILISER SUR DE NOUVELLES DONNÉES (Ex: Les données du prof) ---

# Imaginons que 'X_new' contient les nouvelles données brutes du CERN envoyées par le prof

# X_new = pd.read_csv("donnees_evaluation.csv")

# Étape A : On applique la règle de mesure sauvegardée (SURTOUT PAS de fit_transform ici, juste transform !)

X_new_scaled = scaler.transform(data_set)

# Étape B : On demande au cerveau de prédire instantanément

predictions = model.predict(X_new_scaled, batch_size=2048).flatten().ravel()

# 'predictions' contient maintenant tes scores de probabilité (entre 0 et 1) pour chaque événement !
