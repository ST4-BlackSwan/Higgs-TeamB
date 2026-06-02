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



L = np.linspace(0.97,1.03,5)
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