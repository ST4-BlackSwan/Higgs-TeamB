import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def calculate_weighted_roc(y_true, y_pred, sample_weights, plot_output_path="nn_day1_roc_curve.png"):
    """
    Computes the False Positive Rate, True Positive Rate, and AUC score
    using physical event weights, and plots the standard ROC curve.
    """
    # Compute the weighted ROC curve components
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, sample_weight=sample_weights)
    auc_score = auc(fpr, tpr)
    
    print("-" * 40)
    print(f"ROC EVALUATION COMPLETED")
    print(f"Weighted ROC AUC Score: {auc_score:.4f}")
    print("-" * 40)
    
    # Generate the physics-standard ROC Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"NN Classifier (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Background Efficiency)")
    plt.ylabel("True Positive Rate (Signal Efficiency)")
    plt.title("Weighted Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    
    return fpr, tpr, auc_score