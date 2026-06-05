"""
Feature Importance Analysis for BDT Model
This script calculates and visualizes feature importances from the trained BDT model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from HiggsML.datasets import download_dataset
from boosted_decision_tree_scale_pos_weight import BoostedDecisionTreeScalePosWeight


def main():
    """Main function to compute and plot feature importances."""
    
    # Download and load data
    print("[*] Loading data...")
    data = download_dataset("blackSwan_data")
    data.load_train_set()
    data_set = data.get_train_set()
    
    # Prepare features, labels, and weights
    train_data = data_set.drop(columns=["labels", "weights", "detailed_labels"])
    feature_names = train_data.columns.tolist()
    
    labels = data_set["labels"]
    weights = data_set["weights"]
    
    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        train_data,
        labels,
        weights,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )
    
    # Initialize and train the BDT model
    print("[*] Training BDT model...")
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
    print("[*] Training complete.")
    
    # Extract feature importances
    print("[*] Extracting feature importances...")
    feature_importances = bdt.model.feature_importances_
    
    # Create a DataFrame for better handling and sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Display results
    print("\n" + "="*60)
    print("Feature Importances (sorted by importance):")
    print("="*60)
    print(importance_df.to_string(index=False))
    print("="*60 + "\n")
    
    # Create multiple plots
    print("[*] Creating plots...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Horizontal bar chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature', fontsize=10)
    ax1.set_title('Feature Importance from BDT Model (All Features)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.invert_yaxis()
    
    # Plot 2: Cumulative importance (bias visualization)
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative_importance = np.cumsum(importance_df['Importance'].values)
    cumulative_normalized = cumulative_importance / cumulative_importance[-1] * 100
    
    ax2.plot(range(len(importance_df)), cumulative_normalized, 'o-', linewidth=2.5, markersize=4, color='darkred', label='Cumulative %')
    ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% threshold')
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% threshold')
    ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Feature Importance (Bias Distribution)', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Histogram of importance distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(importance_df['Importance'], bins=15, color='purple', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Feature Importances', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
    print("[✓] Plot saved as 'feature_importances.png'")
    plt.show()
    
    # Statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Total features: {len(feature_names)}")
    print(f"Mean importance: {feature_importances.mean():.6f}")
    print(f"Std importance: {feature_importances.std():.6f}")
    print(f"Max importance: {feature_importances.max():.6f} ({importance_df.iloc[0]['Feature']})")
    print(f"Min importance: {feature_importances.min():.6f} ({importance_df.iloc[-1]['Feature']})")
    print("="*60 + "\n")
    
    # Save to CSV
    importance_df.to_csv('feature_importances.csv', index=False)
    print("[✓] Results saved to 'feature_importances.csv'")


if __name__ == "__main__":
    main()
