from HiggsML.systematics import systematics
from HiggsML.datasets import download_dataset

# Charger les données normales
data_normal = download_dataset(
    "blackSwan_data"
)   # votre dataset de base

data_normal.load_train_set()
data_train = data_normal.get_train_set()

import matplotlib.pyplot as plt

# Teste différentes valeurs de TES et Soft MET
tes_values = [1.0, 1.03, 1.05, 1.10]      # 0%, +3%, +5%, +10%
soft_met_values = [0, 3, 5, 10]             # en GeV

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Ligne 1 : Impact du TES sur PRI_had_pt ---
ax = axes[0, 0]
for tes in tes_values:
    d = systematics(data_train, tes=tes)
    ax.hist(d['PRI_had_pt'], bins=50, alpha=0.5, label=f'TES={tes}')
ax.legend()
ax.set_title("TES → PRI_had_pt")
ax.set_xlabel("PRI_had_pt (MeV)")

# --- Ligne 1 : Impact du TES sur PRI_met ---
ax = axes[0, 1]
for tes in tes_values:
    d = systematics(data_train, tes=tes)
    ax.hist(d['PRI_met'], bins=50, alpha=0.5, label=f'TES={tes}')
ax.legend()
ax.set_title("TES → PRI_met")
ax.set_xlabel("PRI_met (MeV)")

# --- Ligne 2 : Impact du Soft MET sur PRI_met ---
ax = axes[1, 0]
for soft_met in soft_met_values:
    d = systematics(data_train, soft_met=soft_met)
    ax.hist(d['PRI_met'], bins=50, alpha=0.5, label=f'soft_met={soft_met} GeV')
ax.legend()
ax.set_title("Soft MET → PRI_met")
ax.set_xlabel("PRI_met (MeV)")

# --- Ligne 2 : Impact combiné TES + Soft MET sur PRI_met ---
ax = axes[1, 1]
combinations = [(1.0, 0), (1.03, 3), (1.05, 5), (1.10, 10)]
for tes, soft_met in combinations:
    d = systematics(data_train, tes=tes, soft_met=soft_met)
    ax.hist(d['PRI_met'], bins=50, alpha=0.5, label=f'TES={tes}, soft_met={soft_met}')
ax.legend()
ax.set_title("TES + Soft MET combinés → PRI_met")
ax.set_xlabel("PRI_met (MeV)")

plt.suptitle("Impact des systématiques TES et Soft MET", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()