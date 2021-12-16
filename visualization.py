import matplotlib.pyplot as plt
import numpy as np
from species_transport import species_names

def plot_species_overview(Y, T, axs=None):

    nsp = Y.shape[0]
    if axs is None:
        fig, axs =plt.subplots(2, 4, figsize=(8,6),sharex=True, sharey=True)
    else:
        fig = axs[0,0].get_figure()

    axs = axs.flatten()
    np.arange(nsp+1)

    for k in np.arange(nsp+2):
        if k==nsp:
            dat = T.T
            title = 'temperature [K]'
            vmin,vmax=0,T.max()
            cmap = 'hot'
        elif k==nsp+1:
            dat = np.sum(Y, axis=0).T
            title = r'$\Sigma_i{Y_i}$'
            vmin,vmax=0,2
            cmap = 'seismic'
        else:
            dat = Y[k].T
            title = species_names[k]
            vmin, vmax=0,1
            cmap = 'seismic'

        im = axs[k].imshow(dat, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        axs[k].set_title(title)
        fig.colorbar(im, ax=axs[k], location='bottom')

    fig.suptitle(r'Species concentration / temperature field /$\Sigma_i{Y_i}$')
    plt.tight_layout()
