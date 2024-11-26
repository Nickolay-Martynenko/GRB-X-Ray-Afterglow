import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family']='monospace'

def plot_lightcurves(
    labels:np.ndarray, 
    real:np.ndarray, recon:np.ndarray, weight:np.ndarray,
    time_grid:np.ndarray, offset:float):

    for label, true, reco, w in zip(labels, real, recon, weight):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlim(1, 8)
        ax.set_ylim(-1-offset, 7-offset)
        ax.grid(True)

        ax.set_xlabel(r'$\log_{10}$[Time / s]', fontsize=15)
        ax.set_ylabel(r'$\log_{10}$[Source rate / s$^{-1}$]', fontsize=15)
        ax.set_title(label)

        mask = w.astype(bool)
        start, stop = np.argwhere(mask).ravel()[[0, -1]]

        ax.errorbar(
            time_grid[mask], true[mask]-offset, w[mask]**(-1/2),
            fmt='s', markersize=4, capsize=4, color='black', label='true'
        )
        ax.plot(
            time_grid[start:stop+1], reco[start:stop+1]-offset,
            linewidth=2, color='dimgray', label='reco'
        )
        ax.legend(loc='upper right', fontsize=15)

        fig.savefig(f'./Figures/{label}.pdf', format='pdf', bbox_inches='tight')