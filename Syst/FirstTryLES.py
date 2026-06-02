from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset

# Charger les données normales
data_normal = download_dataset(
    "blackSwan_data"
)   # votre dataset de base

data_normal.load_train_set()
data_train = data_normal.get_train_set()

# Appliquer une variation TES de 3%
data_with_TES = systematics(
    data_train,
    tes=1.03  # +3% sur l'énergie des taus
)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.hist(data_train['PRI_had_pt'], bins=50, alpha=0.5, label='Normal')
plt.hist(data_with_TES['PRI_had_pt'], bins=50, alpha=0.5, label='TES +3%')
plt.legend()
plt.title("PRI_had_pt — sensible au TES")

plt.subplot(1,2,2)
plt.hist(data_train['PRI_met'], bins=50, alpha=0.5, label='Normal')
plt.hist(data_with_TES['PRI_met'], bins=50, alpha=0.5, label='TES +3%')
plt.legend()
plt.title("PRI_met — sensible au TES")

plt.tight_layout()
plt.show()