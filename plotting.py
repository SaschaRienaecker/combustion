import string

def set_size(width='thesis', fraction=1, aspect_r=None):
    """
    This code ( set_size ) fragment is copied from
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    aspect_r: float, optional
            Aspect ratio, determines the height based on the width. Default is golden ratio.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 452.9679
    elif width == 'TP':
        width_pt = 418.25
    elif width == 'article':
        width_pt = 250.0
    # elif width == 'beamer':
    #     width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if aspect_r is None:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        aspect_r = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / fraction * aspect_r

    return (fig_width_in, fig_height_in)

def annotate_subplots(axs, hpos=0.5, vpos=1.1, horizontalalignment='center', verticalalignment='top'):

    for n, ax in enumerate(axs):
        ax.text(hpos, vpos, '({})'.format(string.ascii_lowercase[n]), transform=ax.transAxes, horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment)



if __name__ == '__main__':
    """A little example"""
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('./tex.mplstyle') # do not forget to set the path correctly
    fsize = set_size(width=250)
    fig, ax = plt.subplots(figsize=fsize)

    X = np.linspace(0,10,50)
    Y = np.sin(X)
    ax.plot(X,Y)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('This is a custom, publication-quality figure.')
    fig.savefig('../figures/example_fig.pdf')
    plt.tight_layout()
    plt.show()
