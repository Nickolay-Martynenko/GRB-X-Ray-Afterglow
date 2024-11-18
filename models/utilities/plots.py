import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from data_downloader import download_single_LC

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

def make_errorbar(ax:Axes,
	x:np.ndarray, y:np.ndarray,
	yerr:np.ndarray, xerr:np.ndarray,
	color:str,
	fmt='s', markersize=6, capsize=3, 
	)->Axes:
	"""
	Adjusts errobar to axes
	"""
	ax.errorbar(x, y, yerr, xerr,
		color=color,
		fmt=fmt,
		markersize=markersize,
		capsize=capsize)
	return ax

def errorbar_plot_collection(
	event_names:list,
	colors:list=None,
	fmts:list=None,
	title:str='Title',
    savedir:str='./tmp',
    show:bool=False):
	"""
	Draws errorbar from the list of event_names

	Parameters
	----------
	event_names : list 
		List of event_names to
		be downloaded and visualized
	colors : list, default=None
		The list of colors to be assigned
		to the errobar plots. Must be of 
		the same length as event_names. 
		If None, a unique color is assigned
		to each plot.
	fmts : list, default=None
		The list of fmts to be assigned
		to the errobar plots. Must be of 
		the same length as event_names. 
		If None, a default fmt 's' is 
		assigned to each plot.
	title : str, default='Title'
        Title of the resulting figure.
    savedir : str, default='./tmp'
        The directory to save the plot in.
    show : bool, default=False
        Whether to show the plot in the notebook.

    Returns
    -------
    None
	"""

	fig, ax = plt.subplots()

	num_events = len(events)

	if colors is None:
		colors = [COLORMAP((i+1)/num_events) for i in range(num_events)]
	if fmts is None:
		fmts = ['s',]*num_events

	for event_name, color, fmt in zip(event_names, colors, fmts):
		x, y, yerr, xerr = download_single_LC(event_name)
		ax = make_errorbar(ax, x, y, yerr, xerr, color, fmt)

	plt.tight_layout() 
    plt.savefig(f'{savedir}/{title}.pdf', format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def visualize_latent(
    df:pd.DataFrame,
    color_column:str='numbreaks',
    ignore_dim:bool=True,
    title:str='Title',
    savedir:str='./tmp',
    show:bool=False):
    
    """
    Visualization Utility.
    Only 1D, 2D or 3D data can be trivially visualized,
    so the other datasets are truncated.
    
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

    if (dim>=4 and not ignore_dim):
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

    if dim==1:
        ax = fig.add_subplot()
        np.random.seed(42)
        ax.scatter(
            df['feature_0'].values,
            np.random.uniform(size=df['feature_0'].values.shape),
            c=c, cmap=COLORMAP, s=32, marker='o'
        )
    
    ax.set_title(title)

    if len(handles) > 0:
    	ax.legend(handles, bbox_to_anchor=(1.1, 0.5), loc='center left')

    plt.tight_layout() 
    plt.savefig(f'{savedir}/{title}.pdf', format='pdf', bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()