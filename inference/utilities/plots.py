import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family']='monospace'

def plot_lightcurves(
    labels:np.ndarray, 
    real:np.ndarray, recon:np.ndarray, weight:np.ndarray,
    time_grid:np.ndarray, offset:float):

    for label, true, reco, w in zip(labels, real, recon, weight):
        plt.xlim(1, 8)
        plt.ylim(1-offset, 7-offset)
        plt.grid(True)

        plt.xlabel(r'$\log_{10}$[Time / s]', fontsize=15)
        plt.ylabel(r'$\log_{10}$[Source rate / s$^{-1}$]', fontsize=15)
        plt.title(label)

        mask = w.astype(bool)
        start, stop = np.where(mask)[[0, -1]]

        plt.errorbar(
            time_grid[mask]-offset, true[mask]-offset, w[mask]**(-1/2),
            fmt='s', markersize=8, capsize=4, color='black', label='true'
        )
        plt.plot(
            time_grid[start:stop+1]-offset, reco[start:stop+1]-offset,
            linewidth=2, color='dimgray', label='reco'
        )
        plt.legend(loc='upper right', fontsize=15)

        plt.savefig(f'./Figures/{label}.pdf', format='pdf', bbox_inches='tight')
        plt.close()