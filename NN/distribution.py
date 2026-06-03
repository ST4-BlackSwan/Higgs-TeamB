import matplotlib.pyplot as plt
import numpy as np


def plot_score_separability(y_true, y_pred, sample_weights, plot_output_path="nn_day1_score_distribution.png"):
    """
    Plots a normalized, weighted histogram overlay of the model scores 
    separately for true Higgs Signal and Standard Model Background.
    """
    plt.figure(figsize=(8, 6))
    
    # Define 50 bins scanning the full scale of the Sigmoid output layer
    bins = np.linspace(0.0, 1.0, 50)
    
    # Draw Weighted Signal Histogram
    plt.hist(
        y_pred[y_true == 1],
        bins=bins,
        weights=sample_weights[y_true == 1],
        alpha=0.5,
        color="royalblue",
        label="True Higgs Signal ($H \\rightarrow \\tau\\tau$)",
        density=True  # Normalizes area to 1 to compare shape shapes directly
    )
    
    # Draw Weighted Background Histogram
    plt.hist(
        y_pred[y_true == 0],
        bins=bins,
        weights=sample_weights[y_true == 0],
        alpha=0.5,
        color="crimson",
        label="Standard Model Background",
        density=True
    )
    
    plt.xlim([0.0, 1.0])
    plt.xlabel("Neural Network Selection Score (Probability)")
    plt.ylabel("Normalized Event Density (Weighted)")
    plt.title("Day 1 Neural Network Score Separability Distribution")
    plt.legend(loc="upper center")
    plt.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    print("Physics diagnostic: Score distribution overlay plot generated successfully.")