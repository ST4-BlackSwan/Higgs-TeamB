{
    "author": "FAIR Universe",
    "total_rows": 2000000,
    "total_columns": 31,
    "columns": [
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_had_pt",
        "PRI_had_eta",
        "PRI_had_phi",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_n_jets",
        "PRI_jet_all_pt",
        "PRI_met",
        "PRI_met_phi",
        "weights",
        "detailed_labels",
        "labels",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_deltar_had_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_had",
        "DER_met_phi_centrality",
        "DER_lep_eta_centrality"
    ],
    "detailed_labels": [
        "ztautau",
        "diboson",
        "ttbar",
        "htautau"
    ],
    "sum_weights": 105719.0,
    "luminosity": 10
}



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# REMPLACEZ par le chemin réel (gardez le 'r' devant !)
# Exemple : r'C:\Users\carol\Documents\GitHub\Higgs-TeamB\FA\training.csv'
file_path = r'VOTRE_CHEMIN_COMPLET_ICI\training.csv'

# Chargement
df = pd.read_csv(file_path)

# --- Histogrammes ---
variables = ['PRI_lep_phi', 'PRI_met', 'DER_mass_vis', 'DER_deltaeta_jet_jet']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, var in enumerate(variables):
    s = df[(df['Label'] == 's') & (df[var] != -999)][var]
    b = df[(df['Label'] == 'b') & (df[var] != -999)][var]
    
    axes[i].hist(s, bins=50, alpha=0.5, label='Signal', color='red', density=True)
    axes[i].hist(b, bins=50, alpha=0.5, label='Background', color='blue', density=True)
    axes[i].set_title(var)
    axes[i].legend()

plt.tight_layout()
plt.show()

# --- Matrice de corrélation ---
plt.figure(figsize=(8, 6))
sns.heatmap(df[variables].corr(), annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()