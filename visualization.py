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

def plot_velocity_image(u,v,u0,v0,axs=None):

    if axs is None:
        fig, axs=plt.subplots(2,2, figsize=(6,6), sharex=True, sharey=True)

    U  = np.array([u,v])
    U0 = np.array([u0,v0])
    umax = np.max(np.abs(U))

    for i in [0,1]:
        [ax1, ax2] = axs[i,:]

        im1=ax1.imshow(U0[i].T, cmap='seismic', vmax=umax, vmin=-umax, origin='lower') # first frame
        im2=ax2.imshow(U[i].T , cmap='seismic', vmax=umax, vmin=-umax, origin='lower') # last frame

        s = '$u$' if i==0 else '$v$'
        fig.colorbar(im1, ax=ax1)
        ax1.set_title("{} at $n=0$".format(s))
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        #im2=ax2.imshow(np.abs(vres[numplot]).T)
        fig.colorbar(im2, ax=ax2)
        ax2.set_title(r"after simulation")
        fig.suptitle(r'velocity fields')
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

def plot_velocity_vector_field(u, v, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_title('velocity field')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

    ax.quiver(u.T, v.T)
