import matplotlib.pyplot as plt
import numpy as np


def optimize_poisson_significance(
    y_true,
    y_pred,
    sample_weights,
    step=0.01,
    plot_output_path="nn_day1_significance_scan.png",
):
    """
    Sweeps score cuts from 0.0 to 1.0 to find the optimal threshold
    maximizing s / sqrt(s + b) based on physical cross-section weights.
    """
    thresholds = np.arange(0.0, 1.0, step)
    significances = []
    s_yields = []
    b_yields = []

    signal_mask = y_true == 1
    background_mask = y_true == 0

    # Linearly scan the scores to evaluate threshold capabilities
    for t in thresholds:
        accepted_events = y_pred >= t

        # Aggregate physical counts for events passing the cut
        s = np.sum(sample_weights[signal_mask & accepted_events])
        b = np.sum(sample_weights[background_mask & accepted_events])

        # Compute standard Poisson significance approximation
        if (s + b) > 0:
            sig = s / np.sqrt(s + b)
        else:
            sig = 0.0

        significances.append(sig)
        s_yields.append(s)
        b_yields.append(b)

    significances = np.array(significances)

    # Locate the optimum working point
    max_idx = np.argmax(significances)
    optimal_threshold = thresholds[max_idx]
    max_significance = significances[max_idx]
    opt_s = s_yields[max_idx]
    opt_b = b_yields[max_idx]

    print("=" * 45)
    print("SIGNIFICANCE OPTIMIZATION COMPLETE")
    print(f"Max Significance:   {max_significance:.3f} sigma")
    print(f"Optimal Score Cut:  {optimal_threshold:.2f}")
    print(f"Yields at Cut:      Signal={opt_s:.2f}, Background={opt_b:.2f}")
    print("=" * 45)

    # Plot Significance Curve
    plt.figure(figsize=(7, 6))
    plt.plot(
        thresholds, significances, color="purple", lw=2, label="Significance Profile"
    )
    plt.axvline(
        x=optimal_threshold,
        color="black",
        linestyle=":",
        label=f"Optimal Working Cut ({optimal_threshold:.2f})",
    )

    plt.xlim([0.0, 1.0])
    plt.xlabel("Neural Network Selection Cut Threshold")
    plt.ylabel("Poisson Significance [s / sqrt(s+b)]")
    plt.title(f"Significance Scan Profile (Max: {max_significance:.3f} $\sigma$)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=300)
    plt.close()

    return optimal_threshold, max_significance
