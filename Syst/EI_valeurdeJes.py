### Code de FirtsTryLES adapté ci-dessous ###

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


### Code de Ludovic ci-dessous ###

from HiggsML.ingestion import Ingestion

# ingestion = Ingestion(data)
# rajouter ici les autres lignes de traitement de ingestion (sinon ça marche pas)
import numpy as np

# Pour tracer l'histogramme des scores pour différentes valeurs de JEs
from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset
from HiggsML.ingestion import Ingestion
import numpy as np
import matplotlib.pyplot as plt

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

# Pour tracer l'histogramme des scores pour différentes valeurs de JEs

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


data_with_JES = systematics(data_train, jes=1)  # variation de jes
import matplotlib.pyplot as plt

# Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
non_feature_cols = ["weights", "labels", "detailed_labels"]
feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

# Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
# Assuming predict returns probabilities/scores directly.
raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

signal_scores_jes = raw_scores_jes[data_with_JES["labels"] == 1]
background_scores_jes = raw_scores_jes[data_with_JES["labels"] == 0]

print(signal_scores_jes)

s = 0
b = 0
n = 0
for i in signal_scores_jes:
    if i > 0.9:
        s = s + 1
for i in background_scores_jes:
    if i > 0.9:
        b = b + 1
n = s + b
print(n, s, b)

liste = []

L = np.linspace(0.97, 1.03, 10)
for jesval in L:
    data_with_JES = systematics(data_train, jes=jesval)  # variation de jes

    # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
    non_feature_cols = ["weights", "labels", "detailed_labels"]
    feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

    # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
    # Assuming predict returns probabilities/scores directly.
    raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

    signal_scores_jes = raw_scores_jes[data_with_JES["labels"] == 1]
    background_scores_jes = raw_scores_jes[data_with_JES["labels"] == 0]

    print(signal_scores_jes)

    s = 0
    b = 0
    for i in signal_scores_jes:
        if i > 0.9:
            s = s + 1
    for i in background_scores_jes:
        if i > 0.9:
            b = b + 1
    mu = (n - b) / s
    print(mu, s, b)
    liste.append(mu)
print(liste)


import matplotlib.pyplot as plt

plt.plot(L, liste)
plt.show()

# Plot du delta jes
data_with_JES = systematics(data_train, jes=1)  # variation de jes
import matplotlib.pyplot as plt

# Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
non_feature_cols = ["weights", "labels", "detailed_labels"]
feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

# Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
# Assuming predict returns probabilities/scores directly.
raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

signal_scores_jes = raw_scores_jes[data_with_JES["labels"] == 1]
background_scores_jes = raw_scores_jes[data_with_JES["labels"] == 0]

print(signal_scores_jes)

s0 = 0
b0 = 0
for i in signal_scores_jes:
    if i > 0.9:
        s0 = s0 + 1
for i in background_scores_jes:
    if i > 0.9:
        b0 = b0 + 1


liste_delta_s = []
liste_delta_b = []
L = np.linspace(0.9, 1.1, 10)
for jesval in L:
    data_with_JES = systematics(data_train, jes=jesval)  # variation de jes

    # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
    non_feature_cols = ["weights", "labels", "detailed_labels"]
    feature_cols = [col for col in data_with_JES.columns if col not in non_feature_cols]

    # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
    # Assuming predict returns probabilities/scores directly.
    raw_scores_jes = ingestion.model.model.predict(data_with_JES[feature_cols])

    signal_scores_jes = raw_scores_jes[data_with_JES["labels"] == 1]
    background_scores_jes = raw_scores_jes[data_with_JES["labels"] == 0]

    s = 0
    b = 0
    for i in signal_scores_jes:
        if i > 0.9:
            s = s + 1
    for i in background_scores_jes:
        if i > 0.9:
            b = b + 1
    liste_delta_s.append(s - s0)
    liste_delta_b.append(b - b0)

plt.plot(L, liste_delta_s, marker="o")
plt.show()
plt.plot(L, liste_delta_b, marker="o")
plt.show()


# Variation pour Tes

data_with_TES = systematics(data_train, tes=1)  # variation de tes

# Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
non_feature_cols = ["weights", "labels", "detailed_labels"]
feature_cols = [col for col in data_with_TES.columns if col not in non_feature_cols]

# Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
# Assuming predict returns probabilities/scores directly.
raw_scores_tes = ingestion.model.model.predict(data_with_TES[feature_cols])

signal_scores_tes = raw_scores_tes[data_with_TES["labels"] == 1]
background_scores_tes = raw_scores_tes[data_with_TES["labels"] == 0]


s0 = 0
b0 = 0
for i in signal_scores_tes:
    if i > 0.9:
        s0 = s0 + 1
for i in background_scores_tes:
    if i > 0.9:
        b0 = b0 + 1


liste_delta_s = []
liste_delta_b = []
L = np.linspace(0.9, 1.1, 10)
for tesval in L:
    data_with_TES = systematics(data_train, tes=tesval)  # variation de jes

    # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
    non_feature_cols = ["weights", "labels", "detailed_labels"]
    feature_cols = [col for col in data_with_TES.columns if col not in non_feature_cols]

    # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
    # Assuming predict returns probabilities/scores directly.
    raw_scores_tes = ingestion.model.model.predict(data_with_TES[feature_cols])

    signal_scores_tes = raw_scores_tes[data_with_TES["labels"] == 1]
    background_scores_tes = raw_scores_tes[data_with_TES["labels"] == 0]

    s = 0
    b = 0
    for i in signal_scores_tes:
        if i > 0.9:
            s = s + 1
    for i in background_scores_tes:
        if i > 0.9:
            b = b + 1
    liste_delta_s.append(s - s0)
    liste_delta_b.append(b - b0)

plt.plot(L, liste_delta_s, marker="o")
plt.show()
plt.plot(L, liste_delta_b, marker="o")
plt.show()


# Programme pour faire varier le parmètre d'intérêt argument par de la fonction, mettre par exemple 'jes', mettre ensuite la valeur de ce paramètre sans erreur systématique
# Puis mettre la liste de valeur du paramètre que l'on veut tester, puis la plage de score accepté
# Exemple d'input : varpar('jes',1,[0.97,1.03],[0.9,1])


def varpar(par, parnormal, listevalpar, intervalscore):
    c = intervalscore[0]
    d = intervalscore[1]
    kwargs = {par: parnormal}
    data_with_PAR = systematics(data_train, **kwargs)  # variation du paramètre

    # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
    non_feature_cols = ["weights", "labels", "detailed_labels"]
    feature_cols = [col for col in data_with_PAR.columns if col not in non_feature_cols]

    # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
    # Assuming predict returns probabilities/scores directly.
    raw_scores_par = ingestion.model.model.predict(data_with_PAR[feature_cols])

    signal_scores_par = raw_scores_par[data_with_PAR["labels"] == 1]
    background_scores_par = raw_scores_par[data_with_PAR["labels"] == 0]

    s0 = 0
    b0 = 0
    for i in range(len(raw_scores_par)):
        if (
            data_with_PAR["labels"][i] == 1
            and raw_scores_par[i] > c
            and raw_scores_par[i] < d
        ):
            s0 = s0 + data_with_PAR["weights"][i]
        if (
            data_with_PAR["labels"][i] == 0
            and raw_scores_par[i] > c
            and raw_scores_par[i] < d
        ):
            b0 = b0 + data_with_PAR["weights"][i]

    liste_delta_s = []
    liste_delta_b = []
    L = listevalpar
    for parval in L:
        kwargs = {par: parval}
        data_with_PAR = systematics(data_train, **kwargs)  # variation du paramètre

        # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
        non_feature_cols = ["weights", "labels", "detailed_labels"]
        feature_cols = [
            col for col in data_with_PAR.columns if col not in non_feature_cols
        ]

        # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
        # Assuming predict returns probabilities/scores directly.
        raw_scores_par = ingestion.model.model.predict(data_with_PAR[feature_cols])

        signal_scores_tes = raw_scores_par[data_with_PAR["labels"] == 1]
        background_scores_tes = raw_scores_par[data_with_PAR["labels"] == 0]

        s = 0
        b = 0
        for i in range(len(raw_scores_par)):
            if (
                data_with_PAR["labels"][i] == 1
                and raw_scores_par[i] > c
                and raw_scores_par[i] < d
            ):
                s = s + data_with_PAR["weights"][i]
            if (
                data_with_PAR["labels"][i] == 0
                and raw_scores_par[i] > c
                and raw_scores_par[i] < d
            ):
                b = b + data_with_PAR["weights"][i]
        liste_delta_s.append(s - s0)
        liste_delta_b.append(b - b0)

    plt.plot(L, liste_delta_s, marker="o")
    plt.show()
    plt.plot(L, liste_delta_b, marker="o")
    plt.show()
    return liste_delta_s, liste_delta_b
