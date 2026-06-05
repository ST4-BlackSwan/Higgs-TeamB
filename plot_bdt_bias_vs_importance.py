"""
Script: plot_bdt_bias_vs_importance.py
- Entraine le BDT (même config que main.py)
- Extrait les feature importances
- Parse le raw_data (biais système) fourni et calcule RMS par feature
- Trace : 1) horizontal bar chart des importances 2) scatter RMS(biais) vs importance
- Sauvegarde les résultats (PNG + CSV)
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from HiggsML.datasets import download_dataset
from boosted_decision_tree_scale_pos_weight import BoostedDecisionTreeScalePosWeight
from sklearn.model_selection import train_test_split


def parse_raw_biases(raw_data: str):
    sections = [s for s in raw_data.strip().split('\n\n') if s.strip()]
    all_features_biases = {}
    for section_content in sections:
        lines = section_content.strip().split('\n')
        section_name_label = lines[0].strip()
        csv_data = '\n'.join(lines[1:])
        if not csv_data.strip():
            continue
        df = pd.read_csv(io.StringIO(csv_data))
        bias_cols = [col for col in df.columns if 'moy' in col or 'moy/s' in col or 'Ï' in col or 'Î' in col]
        if not bias_cols:
            continue
        bias_col_name = bias_cols[0]
        for _, row in df.iterrows():
            feature_name = row['feature']
            bias_value = pd.to_numeric(row[bias_col_name], errors='coerce')
            if feature_name not in all_features_biases:
                all_features_biases[feature_name] = []
            if not pd.isna(bias_value):
                all_features_biases[feature_name].append(bias_value)
    rms_biases = {}
    for feature, biases in all_features_biases.items():
        if biases:
            rms_biases[feature] = np.sqrt(np.mean(np.array(biases) ** 2))
        else:
            rms_biases[feature] = np.nan
    return pd.Series(rms_biases)


def main():
    print("[*] Chargement des données...")
    data = download_dataset("blackSwan_data")
    data.load_train_set()
    data_set = data.get_train_set()

    # Préparer features, labels, weights
    features_df = data_set.drop(columns=["labels", "weights", "detailed_labels"])
    feature_names = features_df.columns.tolist()
    labels = data_set["labels"]
    weights = data_set["weights"]

    # Split comme dans main.py
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        features_df, labels, weights, test_size=0.3, random_state=42, stratify=labels
    )

    # Entraîner le BDT (mêmes hyperparamètres que main.py)
    print("[*] Entraînement du BDT...")
    bdt = BoostedDecisionTreeScalePosWeight(
        n_estimators=1000,
        max_depth=9,
        learning_rate=0.08501869815505453,
        subsample=1.0,
        colsample_bytree=0.6,
        min_child_weight=1,
        tree_method="hist",
        random_state=31415,
        early_stopping_rounds=15
    )
    bdt.fit(X_train, y_train, w_train)
    print("[*] Entraînement terminé.")

    # Extraire importances
    if not hasattr(bdt.model, 'feature_importances_'):
        raise AttributeError('Le modèle interne XGBoost ne fournit pas feature_importances_.')
    importances = bdt.model.feature_importances_
    importance_series = pd.Series(importances, index=feature_names)
    importance_sorted = importance_series.sort_values(ascending=False)

    # Raw bias data (copié depuis votre exemple)
    raw_data = """
TES
feature,famille,TV,Î”moy/Ïƒ
PRI_had_pt,PRI,0.0295,0.0447
DER_mass_vis,DER,0.0258,0.0285
DER_pt_ratio_lep_had,DER,0.0206,-0.0298
DER_sum_pt,DER,0.0181,0.0062
DER_mass_transverse_met_lep,DER,0.017,-0.0234
PRI_met,PRI,0.0144,0.0114
DER_met_phi_centrality,DER,0.0102,-0.0121
PRI_jet_leading_pt,PRI,0.0032,-0.006
DER_deltar_had_lep,DER,0.0027,0.0056
PRI_jet_subleading_pt,PRI,0.0026,-0.004
PRI_met_phi,PRI,0.0019,1e-04
DER_pt_h,DER,0.0013,-0.0034
PRI_jet_subleading_phi,PRI,0.0013,-0.0004
DER_prodeta_jet_jet,DER,0.0012,-1e-04
PRI_jet_all_pt,PRI,0.0012,-0.0025
PRI_jet_subleading_eta,PRI,0.0011,-0.0003
DER_deltaeta_jet_jet,DER,0.001,0.001
DER_mass_jet_jet,DER,0.001,-0.0025
DER_pt_tot,DER,0.0009,-0.0023
PRI_n_jets,PRI,0.0009,-0.0014
DER_lep_eta_centrality,DER,0.0009,1e-04
PRI_jet_leading_eta,PRI,0.0008,1e-04
PRI_jet_leading_phi,PRI,0.0008,-0.0004
PRI_had_eta,PRI,0.0006,0.0003
PRI_lep_eta,PRI,0.0005,0.0004
PRI_lep_phi,PRI,0.0004,1e-04
PRI_lep_pt,PRI,0.0004,-0.0006
PRI_had_phi,PRI,0.0004,-1e-04


Soft_MET
feature,famille,TV,Î”moy/Ïƒ
DER_pt_tot,DER,0.0236,0.0359
DER_mass_transverse_met_lep,DER,0.0224,0.0278
PRI_met,PRI,0.0208,0.023
DER_pt_h,DER,0.0194,0.011
DER_met_phi_centrality,DER,0.0115,0.0027
PRI_met_phi,PRI,0.0032,0.0
PRI_lep_eta,PRI,0.0,0.0
DER_pt_ratio_lep_had,DER,0.0,0.0
DER_sum_pt,DER,0.0,0.0
DER_deltar_had_lep,DER,0.0,0.0
DER_prodeta_jet_jet,DER,0.0,0.0
DER_mass_jet_jet,DER,0.0,0.0
DER_deltaeta_jet_jet,DER,0.0,0.0
DER_mass_vis,DER,0.0,0.0
PRI_lep_pt,PRI,0.0,0.0
PRI_jet_all_pt,PRI,0.0,0.0
PRI_n_jets,PRI,0.0,0.0
PRI_jet_subleading_phi,PRI,0.0,0.0
PRI_jet_subleading_eta,PRI,0.0,0.0
PRI_jet_subleading_pt,PRI,0.0,0.0
PRI_jet_leading_phi,PRI,0.0,0.0
PRI_jet_leading_eta,PRI,0.0,0.0
PRI_jet_leading_pt,PRI,0.0,0.0
PRI_had_phi,PRI,0.0,0.0
PRI_had_eta,PRI,0.0,0.0
PRI_had_pt,PRI,0.0,0.0
PRI_lep_phi,PRI,0.0,0.0
DER_lep_eta_centrality,DER,0.0,0.0


JES
feature,famille,TV,Î”moy/Ïƒ
PRI_n_jets,PRI,0.0114,0.0173
PRI_jet_all_pt,PRI,0.0112,0.0201
DER_mass_jet_jet,DER,0.0104,0.0194
DER_sum_pt,DER,0.0096,0.0165
DER_met_phi_centrality,DER,0.0087,0.0121
PRI_jet_subleading_pt,PRI,0.0079,0.0109
PRI_met,PRI,0.0078,0.0221
DER_deltaeta_jet_jet,DER,0.0058,-0.0379
PRI_jet_leading_pt,PRI,0.0057,0.0121
DER_lep_eta_centrality,DER,0.0055,0.0003
DER_pt_h,DER,0.0052,0.0159
PRI_jet_subleading_eta,PRI,0.005,-0.0007
PRI_jet_leading_eta,PRI,0.0042,1e-04
DER_prodeta_jet_jet,DER,0.0041,1e-04
DER_pt_tot,DER,0.0038,-0.007
DER_mass_transverse_met_lep,DER,0.0018,0.0024
PRI_jet_subleading_phi,PRI,0.0015,0.0006
PRI_met_phi,PRI,0.0008,-0.0
PRI_jet_leading_phi,PRI,0.0008,-0.0006
PRI_lep_phi,PRI,0.0,0.0
DER_pt_ratio_lep_had,DER,0.0,0.0
PRI_had_pt,PRI,0.0,0.0
PRI_had_phi,PRI,0.0,0.0
DER_deltar_had_lep,DER,0.0,0.0
PRI_had_eta,PRI,0.0,0.0
DER_mass_vis,DER,0.0,0.0
PRI_lep_eta,PRI,0.0,0.0
PRI_lep_pt,PRI,0.0,0.0
"""

    
    # Parse RMS biases
    print("[*] Parsing biases and calcul RMS...")
    rms_series = parse_raw_biases(raw_data).sort_values(ascending=False)
    print("[*] RMS biases calculés.")

    # Align features present in both importance and rms
    common = list(set(importance_sorted.index) & set(rms_series.index))
    if not common:
        raise RuntimeError('Aucune feature commune entre importances et biais.')

    importance_aligned = importance_sorted[common]
    rms_aligned = rms_series[common]

    # Build DataFrame and sort by importance
    comp_df = pd.DataFrame({
        'Feature_Importance': importance_aligned,
        'RMS_Bias': rms_aligned
    }).dropna()
    comp_df = comp_df.sort_values(by='Feature_Importance', ascending=False)

    # Plot horizontal bar chart (importances)
    plt.figure(figsize=(12, 10))
    comp_df['Feature_Importance'].sort_values(ascending=True).plot(kind='barh', color='steelblue')
    plt.title('Feature Importances (BDT)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('bdt_feature_importances_horizontal.png', dpi=300)
    print("[✓] Saved bdt_feature_importances_horizontal.png")
    plt.show()

    # Plot scatter: importance (x) vs RMS bias (y)
    plt.figure(figsize=(14, 10))
    plt.scatter(comp_df['Feature_Importance'], comp_df['RMS_Bias'], s=100, alpha=0.8, edgecolors='k')
    for i, feat in enumerate(comp_df.index):
        plt.annotate(feat, (comp_df['Feature_Importance'].iloc[i], comp_df['RMS_Bias'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, color='dimgray')
    plt.xlabel('Feature Importance (XGBoost)')
    plt.ylabel('RMS of Biases (Î”moy/Ïƒ)')
    plt.title('RMS of Biases vs Feature Importance')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(x=comp_df['Feature_Importance'].mean(), color='r', linestyle='--', label='Avg Importance')
    plt.axhline(y=comp_df['RMS_Bias'].mean(), color='g', linestyle='--', label='Avg RMS Bias')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bdt_bias_vs_importance.png', dpi=300)
    print("[✓] Saved bdt_bias_vs_importance.png")
    plt.show()

    # Save CSV
    comp_df.to_csv('bdt_importance_bias_comparison.csv')
    print("[✓] Saved bdt_importance_bias_comparison.csv")


if __name__ == '__main__':
    main()
