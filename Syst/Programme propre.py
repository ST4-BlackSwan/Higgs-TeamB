#Programme propre

from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset
import matplotlib.pyplot as plt

data_normal = download_dataset("blackSwan_data")

data_normal.load_train_set()
data_train = data_normal.get_train_set()


#Premier test de la fonction systematics avec tes

data_with_TES = systematics(data_train,tes=0.97)
plt.figure(figsize=(12,4))
plt.hist(data_train['weights'], bins=50, alpha=0.5, label='Normal')
plt.hist(data_with_TES['weights'], bins=50, alpha=0.5, label='TES -3%')
plt.legend()
plt.title("PRI_had_pt — sensible au TES")



#Fonction pour déterminer le score
from statistical_analysis import compute_mu

non_feature_cols = ['weights', 'labels', 'detailed_labels']
feature_cols = [col for col in data_train.columns if col not in non_feature_cols]

# Prepare input for prediction for data_train
predict_input_train = {
    'data': data_train[feature_cols],
    'weights': data_train['weights']
}
# Get predictions (scores) from the model. This dictionary already contains mu_hat etc.
mu_results_train = ingestion.model.predict(predict_input_train)
print(mu_results_train)


# Prepare input for prediction for data_with_TES
predict_input_tes = {
    'data': data_with_TES[feature_cols],
    'weights': data_with_TES['weights']
}
# Get predictions (scores) from the model. This dictionary already contains mu_hat etc.
mu_results_tes = ingestion.model.predict(predict_input_tes)
print(mu_results_tes)


#Affichage des courbes de S et B en fonction des valeurs de tes

def vartes(listevaltes,intervalscore):
  c=intervalscore[0]
  d=intervalscore[1]
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
  for i in range (len(raw_scores_tes)) :
      if data_with_TES['labels'][i] == 1 and raw_scores_tes[i] > c and raw_scores_tes[i] <d :
        s0=s0+data_with_TES['weights'][i]
      if data_with_TES['labels'][i] == 0 and raw_scores_tes[i] > c and raw_scores_tes[i] <d :
        b0=b0+data_with_TES['weights'][i]

  liste_delta_s =[]
  liste_delta_b =[]
  L = listevaltes
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
    for i in range (len(raw_scores_tes)) :
      if data_with_TES['labels'][i] == 1 and raw_scores_tes[i] > c and raw_scores_tes[i] <d :
        s=s+data_with_TES['weights'][i]
      if data_with_TES['labels'][i] == 0 and raw_scores_tes[i] > c and raw_scores_tes[i] <d :
        b=b+data_with_TES['weights'][i]
    liste_delta_s.append(s-s0)
    liste_delta_b.append(b-b0)

  plt.plot(L,liste_delta_s,marker='o')
  plt.show()
  plt.plot(L,liste_delta_b,marker='o')
  plt.show()
  return liste_delta_s,liste_delta_b


# Généralisation pour deifférents paramètres

def varpar(par,parnormal,listevalpar,intervalscore):
  c=intervalscore[0]
  d=intervalscore[1]
  kwargs = {par: parnormal}
  data_with_PAR = systematics(
    data_train,
    **kwargs # variation du paramètre
  )

  # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
  non_feature_cols = ['weights', 'labels', 'detailed_labels']
  feature_cols = [col for col in data_with_PAR.columns if col not in non_feature_cols]

    # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
    # Assuming predict returns probabilities/scores directly.
  raw_scores_par = ingestion.model.model.predict(data_with_PAR[feature_cols])

  signal_scores_par = raw_scores_par[data_with_PAR['labels'] == 1]
  background_scores_par = raw_scores_par[data_with_PAR['labels'] == 0]


  s0 = 0
  b0 = 0
  for i in range (len(raw_scores_par)) :
      if data_with_PAR['labels'][i] == 1 and raw_scores_par[i] > c and raw_scores_par[i] <d :
        s0=s0+data_with_PAR['weights'][i]
      if data_with_PAR['labels'][i] == 0 and raw_scores_par[i] > c and raw_scores_par[i] <d :
        b0=b0+data_with_PAR['weights'][i]

  liste_delta_s =[]
  liste_delta_b =[]
  L = listevalpar
  for parval in L:
    kwargs = {par: parval}
    data_with_PAR = systematics(
    data_train,
    **kwargs # variation du paramètre
    )

    # Get the feature columns (assuming they are all columns except 'weights', 'labels', 'detailed_labels')
    non_feature_cols = ['weights', 'labels', 'detailed_labels']
    feature_cols = [col for col in data_with_PAR.columns if col not in non_feature_cols]

    # Get raw prediction scores for data_with_JES from the BDT model (ingestion.model.model.predict)
    # Assuming predict returns probabilities/scores directly.
    raw_scores_par = ingestion.model.model.predict(data_with_PAR[feature_cols])

    signal_scores_tes = raw_scores_par[data_with_PAR['labels'] == 1]
    background_scores_tes = raw_scores_par[data_with_PAR['labels'] == 0]


    s = 0
    b = 0
    for i in range (len(raw_scores_par)) :
      if data_with_PAR['labels'][i] == 1 and raw_scores_par[i] > c and raw_scores_par[i] <d :
        s=s+data_with_PAR['weights'][i]
      if data_with_PAR['labels'][i] == 0 and raw_scores_par[i] > c and raw_scores_par[i] <d :
        b=b+data_with_PAR['weights'][i]
    liste_delta_s.append(s-s0)
    liste_delta_b.append(b-b0)

  plt.plot(L,liste_delta_s,marker='o')
  plt.show()
  plt.plot(L,liste_delta_b,marker='o')
  plt.show()
  return liste_delta_s,liste_delta_b