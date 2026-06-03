import matplotlib.pyplot as plt
import numpy as np


class HEPEvaluationSuite:
    def __init__(self, output_dir="."):
        self.output_dir = output_dir

    def plot_learning_curves(self, history, metric="loss"):
        """
        Génère la courbe d'apprentissage (Loss et Métrique) pour monitorer
        l'overfitting et la vitesse de convergence (celerity).

        history: Objet history de Keras/TensorFlow ou dictionnaire contenant les listes
        """
        if hasattr(history, "history"):
            hist_dict = history.history
        else:
            hist_dict = history

        epochs = range(1, len(hist_dict["loss"]) + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Axe pour la Loss
        color = "tab:blue"
        ax1.set_xlabel("Époques")
        ax1.set_ylabel("Perte (Loss)", color=color)
        ax1.plot(
            epochs, hist_dict["loss"], label="Train Loss", color=color, linestyle="--"
        )
        if "val_loss" in hist_dict:
            ax1.plot(
                epochs,
                hist_dict["val_loss"],
                label="Validation Loss",
                color=color,
                lw=2,
            )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Axe secondaire pour la métrique (ex: AUC ou Accuracy)
        if metric in hist_dict:
            ax2 = ax1.twinx()
            color = "tab:orange"
            ax2.set_ylabel(metric.capitalize(), color=color)
            ax2.plot(
                epochs,
                hist_dict[metric],
                label=f"Train {metric}",
                color=color,
                linestyle="--",
            )
            if f"val_{metric}" in hist_dict:
                ax2.plot(
                    epochs,
                    hist_dict[f"val_{metric}"],
                    label=f"Val {metric}",
                    color=color,
                    lw=2,
                )
            ax2.tick_params(axis="y", labelcolor=color)

        plt.title("Courbes d'apprentissage & Convergence")
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/hep_learning_curve.png", dpi=300)
        plt.close()

    def plot_overtraining_check(
        self, y_train, y_pred_train, w_train, y_test, y_pred_test, w_test, bins=40
    ):
        """
        Le graphique standard en HEP : compare la distribution des scores du modèle
        entre le Train et le Test set pour le Signal et le Background.
        """
        plt.figure(figsize=(8, 6))

        # Masques
        bkg_train, sig_train = (y_train == 0), (y_train == 1)
        bkg_test, sig_test = (y_test == 0), (y_test == 1)

        # Histogrammes pour le Train set (Remplis / Plain)
        plt.hist(
            y_pred_train[bkg_train],
            bins=bins,
            weights=w_train[bkg_train],
            range=(0, 1),
            alpha=0.2,
            color="blue",
            label="Bkg (Train)",
            density=True,
        )
        plt.hist(
            y_pred_train[sig_train],
            bins=bins,
            weights=w_train[sig_train],
            range=(0, 1),
            alpha=0.2,
            color="red",
            label="Sig (Train)",
            density=True,
        )

        # Points pour le Test set (Points avec barres d'erreur ou marqueurs)
        # On calcule les valeurs des bins pour le test
        counts_bkg, bin_edges = np.histogram(
            y_pred_test[bkg_test],
            bins=bins,
            weights=w_test[bkg_test],
            range=(0, 1),
            density=True,
        )
        counts_sig, _ = np.histogram(
            y_pred_test[sig_test],
            bins=bins,
            weights=w_test[sig_test],
            range=(0, 1),
            density=True,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.plot(bin_centers, counts_bkg, "o", color="blue", label="Bkg (Test)")
        plt.plot(bin_centers, counts_sig, "v", color="red", label="Sig (Test)")

        plt.xlabel("Score du Réseau de Neurones")
        plt.ylabel("Densité de Probabilité (Normalisée)")
        plt.title("Test de surapprentissage (Overtraining Check)")
        plt.yscale(
            "log"
        )  # Souvent mis en log en physique pour voir les queues de distribution
        plt.legend(loc="upper center")
        plt.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hep_overtraining_check.png", dpi=300)
        plt.close()

    def optimize_poisson_significance(self, y_true, y_pred, sample_weights, step=0.01):
        """
        Ta fonction d'origine, intégrée proprement à la suite de graphiques.
        """
        thresholds = np.arange(0.0, 1.0, step)
        significances = []

        signal_mask = y_true == 1
        background_mask = y_true == 0

        for t in thresholds:
            accepted_events = y_pred >= t
            s = np.sum(sample_weights[signal_mask & accepted_events])
            b = np.sum(sample_weights[background_mask & accepted_events])

            sig = s / np.sqrt(s + b) if (s + b) > 0 else 0.0
            significances.append(sig)

        significances = np.array(significances)
        max_idx = np.argmax(significances)

        # Plotting
        plt.figure(figsize=(7, 6))
        plt.plot(
            thresholds,
            significances,
            color="purple",
            lw=2,
            label="Profil de Signification",
        )
        plt.axvline(
            x=thresholds[max_idx],
            color="black",
            linestyle=":",
            label=f"Coupure Optimale ({thresholds[max_idx]:.2f})",
        )

        plt.xlim([0.0, 1.0])
        plt.xlabel("Seuil de coupure du Réseau de Neurones")
        plt.ylabel("Signification de Poisson [s / sqrt(s+b)]")
        plt.title(f"Scan de Signification (Max: {significances[max_idx]:.3f} $\sigma$)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hep_significance_scan.png", dpi=300)
        plt.close()

        return thresholds[max_idx], significances[max_idx]
