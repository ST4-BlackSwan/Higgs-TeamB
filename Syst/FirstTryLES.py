from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset

"""Index(['PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_had_pt', 'PRI_had_eta',
       'PRI_had_phi', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta',
       'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_n_jets',
       'PRI_jet_all_pt', 'PRI_met', 'PRI_met_phi', 'weights',
       'detailed_labels', 'labels', 'DER_mass_transverse_met_lep',
       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_had_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_had', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality'],
      dtype='str')"""

# Charger les données normales
data_normal = download_dataset("blackSwan_data")  # votre dataset de base

data_normal.load_train_set()
data_train = data_normal.get_train_set()

# Appliquer une variation TES de 3%
data_with_TES = systematics(data_train, tes=1.03)  # +3% sur l'énergie des taus

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(data_train["PRI_had_pt"], bins=50, alpha=0.5, label="Normal")
plt.hist(data_with_TES["PRI_had_pt"], bins=50, alpha=0.5, label="TES +3%")
plt.legend()
plt.title("PRI_had_pt — sensible au TES")

plt.subplot(1, 2, 2)
plt.hist(data_train["PRI_met"], bins=50, alpha=0.5, label="Normal")
plt.hist(data_with_TES["PRI_met"], bins=50, alpha=0.5, label="TES +3%")
plt.legend()
plt.title("PRI_met — sensible au TES")

plt.tight_layout()
plt.show()
