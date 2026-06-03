# Pour tracer l'histogramme des scores pour différentes valeurs de JEs
from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset
from HiggsML.ingestion import Ingestion

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


L = np.linspace(0.97, 1.03, 5)
for jesval in L:
    data_with_JES = systematics(data_train, jes=jesval)  # variation de jes
    import matplotlib.pyplot as plt

    print("\n--- Score Distribution for JES-varied Data (data_with_JES, JES=0.97) ---")

    # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
    non_feature_cols = ["weights", "labels", "detailed_labels"]
    feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

    # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
    # Assuming predict returns probabilities/scores directly.
    raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

    signal_scores_jes = raw_scores_jes[data_with_JES["labels"] == 1]
    background_scores_jes = raw_scores_jes[data_with_JES["labels"] == 0]

    # Plot the distributions for data_with_JES
    plt.figure(figsize=(10, 6))
    plt.hist(
        background_scores_jes,
        bins=50,
        alpha=0.7,
        label="Background (data_with_JES)",
        color="lightskyblue",
        density=True,
    )
    plt.hist(
        signal_scores_jes,
        bins=50,
        alpha=0.7,
        label="Signal (data_with_JES)",
        color="lightcoral",
        density=True,
    )
    plt.title("Score Distribution for JES-varied Data (Signal vs Background)")
    plt.xlabel("Classification Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()





data_with_JES = systematics(
      data_train,
      jes=1 # variation de jes 
  )
import matplotlib.pyplot as plt

# Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
non_feature_cols = ['weights', 'labels', 'detailed_labels']
feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

  # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
  # Assuming predict returns probabilities/scores directly.
raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

signal_scores_jes = raw_scores_jes[data_with_JES['labels'] == 1]
background_scores_jes = raw_scores_jes[data_with_JES['labels'] == 0]

print(signal_scores_jes)

s = 0
b = 0 
n = 0
for i in signal_scores_jes :
  if i > 0.5 :
    s=s+1
for i in background_scores_jes :
    if i > 0.5 :
      b=b+1
n=s+b
print(n,s,b)

liste =[]

L = np.linspace(0.97,1.03,10)
for jesval in L:
  data_with_JES = systematics(
      data_train,
      jes=jesval  # variation de jes 
  )
  import matplotlib.pyplot as plt

  # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
  non_feature_cols = ['weights', 'labels', 'detailed_labels']
  feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

  # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
  # Assuming predict returns probabilities/scores directly.
  raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

  signal_scores_jes = raw_scores_jes[data_with_JES['labels'] == 1]
  background_scores_jes = raw_scores_jes[data_with_JES['labels'] == 0]

  print(signal_scores_jes)

  s = 0
  b = 0 
  for i in signal_scores_jes :
    if i > 0.5 :
      s=s+1
  for i in background_scores_jes :
      if i > 0.5 :
        b=b+1
  mu = (n-b)/s
  print(mu,s,b)
  liste.append(mu)
  print(liste)


import matplotlib.pyplot as plt
plt.plot(L,liste)
plt.show()



#Pour plot deltaS et deltaB en fonction du jes :
data_with_JES = systematics(
      data_train,
      jes=1 # variation de jes 
  )
import matplotlib.pyplot as plt

# Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
non_feature_cols = ['weights', 'labels', 'detailed_labels']
feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

  # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
  # Assuming predict returns probabilities/scores directly.
raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

signal_scores_jes = raw_scores_jes[data_with_JES['labels'] == 1]
background_scores_jes = raw_scores_jes[data_with_JES['labels'] == 0]

print(signal_scores_jes)

s0 = 0
b0 = 0 
for i in signal_scores_jes :
  if i > 0.9 :
    s0=s0+1
for i in background_scores_jes :
    if i > 0.9 :
      b0=b0+1


liste_delta_s =[]
liste_delta_b =[]
L = np.linspace(0.97,1.03,10)
for jesval in L:
  data_with_JES = systematics(
      data_train,
      jes=jesval  # variation de jes 
  )

  # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
  non_feature_cols = ['weights', 'labels', 'detailed_labels']
  feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

  # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
  # Assuming predict returns probabilities/scores directly.
  raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

  signal_scores_jes = raw_scores_jes[data_with_JES['labels'] == 1]
  background_scores_jes = raw_scores_jes[data_with_JES['labels'] == 0]

  print(signal_scores_jes)

  s = 0
  b = 0 
  for i in signal_scores_jes :
    if i > 0.9 :
      s=s+1
  for i in background_scores_jes :
      if i > 0.9 :
        b=b+1
  liste_delta_s.append(s-s0)
  liste_delta_b.append(b-b0)

plt.plot(L,liste_delta_s)
plt.plot(L,liste_delta_b)
plt.show()



#Variation pour Tes

data_with_TES = systematics(
      data_train,
      tes=1 # variation de tes 
  )

# Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
non_feature_cols = ['weights', 'labels', 'detailed_labels']
feature_cols = [col for col in data_with_TES.columns if col not in non_feature_cols]

  # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
  # Assuming predict returns probabilities/scores directly.
raw_scores_tes = ingestion.model.model.predict(data_with_TES[feature_cols])

signal_scores_tes = raw_scores_tes[data_with_TES['labels'] == 1]
background_scores_tes = raw_scores_tes[data_with_TES['labels'] == 0]


s0 = 0
b0 = 0 
for i in signal_scores_tes :
  if i > 0.9 :
    s0=s0+1
for i in background_scores_tes :
    if i > 0.9 :
      b0=b0+1



#Idem que le précédent mais avec TES cette fois
liste_delta_s =[]
liste_delta_b =[]
L = np.linspace(0.97,1.03,10)
for tesval in L:
  data_with_TES = systematics(
      data_train,
      tes=tesval  # variation de jes 
  )

  # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
  non_feature_cols = ['weights', 'labels', 'detailed_labels']
  feature_cols = [col for col in data_with_TES.columns if col not in non_feature_cols]

  # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
  # Assuming predict returns probabilities/scores directly.
  raw_scores_tes = ingestion.model.model.predict(data_with_TES[feature_cols])

  signal_scores_tes = raw_scores_tes[data_with_TES['labels'] == 1]
  background_scores_tes = raw_scores_tes[data_with_TES['labels'] == 0]


  s = 0
  b = 0 
  for i in signal_scores_tes :
    if i > 0.9 :
      s=s+1
  for i in background_scores_tes :
      if i > 0.9 :
        b=b+1
  liste_delta_s.append(s-s0)
  liste_delta_b.append(b-b0)

plt.plot(L,liste_delta_s,marker='o')
plt.show()
plt.plot(L,liste_delta_b,marker='o')
plt.show()