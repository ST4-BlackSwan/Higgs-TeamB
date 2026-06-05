# liste_delta_s0,liste_delta_b0 = varpar('jes',1,np.linspace(0.9,1.1,10),[0.9,1])

import numpy as np
import matplotlib.pyplot as plt

# à ajouter en bas du jupyter notebook

def histogramme(par, parnormal, listevalpar, intervalscoretotal, number_of_bins):
    delta_s, delta_b=[], []
    intervallen = intervalscoretotal[1] - intervalscoretotal[0]
    for bin in range(number_of_bins):
        liste_delta_s, liste_delta_b = varpar(par, parnormal, listevalpar,np.linspace(intervalscoretotal+intervallen*bin/number_of_bins, intervalscoretotal+intervallen*(bin+1)/number_of_bins,10))
        delta_s.append(liste_delta_s)
        delta_b.append(liste_delta_b)
    delta_s = np.transpose(delta_s)
    delta_b = np.transpose(delta_b)
    for p in range(len(delta_s)):
        plt.plot(np.linspace(intervalscoretotal[0], intervalscoretotal[1], number_of_bins), delta_s[p])
        plt.plot(np.linspace(intervalscoretotal[0], intervalscoretotal[1], number_of_bins), delta_b[p])

histogramme('tes', 1, np.linspace(0.97, 1.03, 10, [0.7, 1.0], 10))
histogramme('jes', 1, np.linspace(0.97, 1.03, 10, [0.7, 1.0], 10))
histogramme('bkg_norm', 1, np.linspace(0.95, 1.05, 10, [0.7, 1.0], 10))
histogramme('soft_met', 1, np.linspace(0, 3, 10, [0.7, 1.0], 10))