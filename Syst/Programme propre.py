#Programme propre

from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset
import matplotlib.pyplot as plt

data_normal = download_dataset("blackSwan_data")

data_normal.load_train_set()
data_train = data_normal.get_train_set()


#Premier test de la fonction systematics avec tes

data_with_JES = systematics(data_train,tes=0.97)
plt.figure(figsize=(12,4))
plt.hist(data_train['weights'], bins=50, alpha=0.5, label='Normal')
plt.hist(data_with_JES['weights'], bins=50, alpha=0.5, label='TES -3%')
plt.legend()
plt.title("PRI_had_pt — sensible au TES")