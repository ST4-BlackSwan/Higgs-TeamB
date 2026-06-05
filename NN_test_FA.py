import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

def parse_raw_biases(raw_data):
    """
    Parses the raw bias data string to extract feature names and their RMS bias values.
    Handles multiple sections (TES, Soft_MET, JES).
    """
    bias_data = {}
    sections = raw_data.strip().split('\n\n')
    for section in sections:
        lines = section.strip().split('\n')
        if len(lines) > 1: # Ensure there are lines after the header
            # Skip the section header (e.g., 'TES') and process the CSV part
            csv_data = "\n".join(lines[1:])
            # Replace non-standard characters in column names before reading
            csv_data = csv_data.replace('Î”moy/Ïƒ', 'Delta_moy_sigma')
            df = pd.read_csv(io.StringIO(csv_data), sep=',')
            # Assuming we need the absolute value of 'Delta_moy_sigma' as RMS bias
            for _, row in df.iterrows():
                if 'Delta_moy_sigma' in row and row['feature'] not in bias_data:
                    bias_data[row['feature']] = abs(row['Delta_moy_sigma'])
    return pd.Series(bias_data)

def get_nn_feature_importances(nn_model, feature_names):
    """
    Calculates a proxy for feature importance from the first layer weights of a Keras NN model.
    The weights of the first Dense layer are used as a proxy for how much each input feature
    contributes to the initial transformations in the network.
    """
    # Get the weights of the first Dense layer (input layer)
    # The weights are in the format (input_dim, output_dim)
    first_layer_weights = nn_model.model.layers[0].get_weights()[0]

    # Sum the absolute values of weights connected to each input feature
    # This gives a measure of how much each input feature influences the first layer
    feature_importances_proxy = np.sum(np.abs(first_layer_weights), axis=1)

    return pd.Series(feature_importances_proxy, index=feature_names)

# Raw bias data (copied from your example)
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

def main():
    print("[*] Extracting feature names from loaded data...")
    # Assuming 'features' DataFrame is available from previous steps (aK36RUGf3Re7 cell)
    feature_names = features.columns.tolist()

    print("[*] Calculating NN feature importances...")
    # Assuming 'nn' object is available from previous steps (kSiG1Yg26iwc cell)
    nn_importances = get_nn_feature_importances(nn, feature_names)
    
    # Normalize to a range between 0 and 1
    nn_importances_scaled = (nn_importances / nn_importances.sum())
    importance_sorted = nn_importances_scaled.sort_values(ascending=False)

    print("[*] Parsing biases...")
    rms_series = parse_raw_biases(raw_data)

    # Align features present in both importance and rms
    common_features = list(set(importance_sorted.index) & set(rms_series.index))
    if not common_features:
        print('Warning: No common features found between NN importances and bias data. Cannot generate comparison plots.')
        return

    importance_aligned = importance_sorted[common_features]
    rms_aligned = rms_series[common_features]

    # Build DataFrame and sort by importance
    comp_df = pd.DataFrame({
        'Feature_Importance_Scaled': importance_aligned,
        'RMS_Bias': rms_aligned
    }).dropna()
    comp_df = comp_df.sort_values(by='Feature_Importance_Scaled', ascending=False)

    print("[*] Generating plots...")

    # Plot horizontal bar chart (importances)
    plt.figure(figsize=(12, 10))
    comp_df['Feature_Importance_Scaled'].sort_values(ascending=True).plot(kind='barh', color='darkgreen')
    plt.title('Feature Importances (Neural Network)')
    plt.xlabel('Importance Score (Proxy) [0-1]')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('nn_feature_importances_horizontal_scaled.png', dpi=300)
    print("[✓] Saved nn_feature_importances_horizontal_scaled.png")
    plt.show()

    # Plot scatter: importance (x) vs RMS bias (y)
    plt.figure(figsize=(14, 10))
    plt.scatter(comp_df['Feature_Importance_Scaled'], comp_df['RMS_Bias'], s=100, alpha=0.8, edgecolors='k', color='darkcyan')
    for i, feat in enumerate(comp_df.index):
        plt.annotate(feat, (comp_df['Feature_Importance_Scaled'].iloc[i], comp_df['RMS_Bias'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, color='dimgray')
    plt.xlabel('Feature Importance (Neural Network Proxy) [0-1]')
    plt.ylabel('RMS of Biases (Î”moy/Ïƒ)')
    plt.title('RMS of Biases vs Feature Importance (Neural Network)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axvline(x=comp_df['Feature_Importance_Scaled'].mean(), color='r', linestyle='--', label='Avg Importance [0-1]')
    plt.axhline(y=comp_df['RMS_Bias'].mean(), color='g', linestyle='--', label='Avg RMS Bias')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nn_bias_vs_importance_scaled.png', dpi=300)
    print("[✓] Saved nn_bias_vs_importance_scaled.png")
    plt.show()

    # Save CSV
    comp_df.to_csv('nn_importance_bias_comparison_scaled.csv')
    print("[✓] Saved nn_importance_bias_comparison_scaled.csv")

if __name__ == "__main__":
    main()
