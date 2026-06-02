import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset
from boosted_decision_tree import BoostedDecisionTree

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
data_normal = download_dataset(
    "blackSwan_data"
)   # votre dataset de base

data_normal.load_train_set()
data_train = data_normal.get_train_set()
bdt = BoostedDecisionTree(data_train)
# Appliquer une variation TES de 3%
data_with_TES = systematics(
    data_train,
    tes=1.03  # +3% sur l'énergie des taus
)

model = GradientBoostingClassifier(n_estimators=100)
cols_a_exclure = ['labels', 'detailed_labels', 'weights']
X = data_train.drop(columns=cols_a_exclure)
y = data_train['labels']
w = data_train['weights']

bdt.fit(X, y)
score = bdt.predict(X) # probabilité d'être signal

# 3. Plotter
plt.figure(figsize=(8, 5))

plt.hist(score, bins=50, range=(0,1), 
         alpha=0.6, color='red', label='Signal')

plt.xlabel("Score BDT")
plt.ylabel("Nombre d'événements")
plt.title("Distribution du score — Signal vs Bruit de fond")
plt.legend()
plt.show()