import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

plt.style.use('ggplot')
plt.rcParams['font.family']='monospace'

PATTERN = re.compile(r'feature_[0-9]+')
COLORMAP = plt.get_cmap('plasma')

def make_legend(colors:list, labels:list, marker:str='o')->list:
    """
    Creates legend handles from colors and labels lists.
    """
    handles = []
    for col, label in zip(colors, labels):
        handles.append(
            Line2D(
                [], [],
                color=col, label=label,
                marker=marker
            )
        )
    return handles

def visualize_latent(
    df:pd.DataFrame,
    color_column:str='numbreaks',
    ignore_dim:bool=True,
    title:str='Title',
    savedir:str='./tmp',
    show:bool=False):
    
    """
    Visualization Utility.
    Only 2D or 3D data can be trivially visualized
    using these function so the larger latent spaces
    are projected on their first three components subpsace.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of the model output, optionally
        together with external labels.
        Extracted / transformed features must be
        named 'feature_0', 'feature_1', and so on.
    color_column : str, default='numbreaks'
        A column to infer colors from.
    ignore_dim : bool, default=True
        If True, >=3D latent representations
        are truncated to the their 3D subspaces.
        Otherwise ValueError is raised in the case
        of too many latent features found.
    title : str, default='Title'
        Title of the resulting figure.
    savedir : str, default='./tmp'
        The directory to save the plot in.
    show : bool, default=False
        Whether to show the plot in the notebook.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If too many ot too few
        latent dimensions found in 
        dataframe and this is not 
        suppressed by `ignore_dim`=True.
    """
    
    columns = [col for col in df.columns if re.match(PATTERN, col)]
    dim = len(columns)

    if ((dim>=4) or (dim<2)) and (not ignore_dim):
        raise ValueError(f'Unable to visualize {dim}D data')
    elif (dim>=4) and ignore_dim:
        print(f'Warning: {dim}D data truncated to 3D')
        columns = columns[:3]
        dim = 3
    
    fig = plt.figure()
    handles = []
    c = 'dimgray'

    if color_column is not None and color_column in df.columns:
        
        if df[color_column].dtype == 'object':
            c, labels = pd.factorize(df[color_column])
            n_colors = len(labels)
            colors = [COLORMAP(i) for i in range(n_colors)]
            handles.append(make_legend(colors, labels))

        elif np.issubdtype(df[color_column].dtype, np.number):
            c = df[color_column].values
    
    
    if dim==3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(
            df['feature_0'].values,
            df['feature_1'].values,
            df['feature_2'].values,
            c=c, cmap=COLORMAP, s=32, marker='o'
        )

    if dim==2:
        ax = fig.add_subplot()
        ax.scatter(
            df['feature_0'].values,
            df['feature_1'].values,
            c=c, cmap=COLORMAP, s=32, marker='o'
        )
    
    ax.set_title(title)

    if len(handles) > 0:
        ax.legend(handles, bbox_to_anchor=(1.1, 0.5), loc='center left')

    plt.tight_layout() 
    plt.savefig(f'{savedir}/{title}.pdf', format='pdf', bbox_inches='tight')

    if show:
        plt.gcf().set_dpi(300)
        plt.show()
    else:
        plt.close()