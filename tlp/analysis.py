import datetime
import json
import os
import subprocess
import tempfile
import typing

import joblib
import matplotlib.axes
import matplotlib.axis
import matplotlib.ticker
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sklearn.metrics
import seaborn as sns
from tqdm.auto import tqdm

from .pipeline import SPLIT_FRACTION
from .features import TIME_STRATEGIES, Strategy, Experiment

def plot_datetime(
  datetimes: pd.Series, *, 
  t_min: typing.Optional[pd.Timestamp] = None,
  t_split: typing.Optional[pd.Timestamp] = None,
  t_max: typing.Optional[pd.Timestamp] = None
  ) -> None:
  """Plot the maturing and probing interval.

  Args:
    datetimes: pd.Series containing pd.Date
    t_min, t_split, t_max: Optional; Dates corresponding to the start of the
      maturing interval, end of the maturing interval and end of the probing
      interval, respectively.

  Usage:
   tlp.analysis.plot_datetime(edgelist['datetimes'])
  """
  t_min_, t_split_, t_max_ = datetimes.quantile([0,SPLIT_FRACTION,1])
  if t_min is None: t_min = t_min_
  if t_split is None: t_split = t_split_
  if t_max is None: t_max = t_max_
  with plt.rc_context({'xtick.top': True, 'ytick.right': True, 
                       'figure.figsize': (20, 4)}):
    datetimes.value_counts(normalize=True).sort_index().cumsum().plot(
      xlabel='Year', ylabel='CDF', xlim=(datetimes.min(), datetimes.max()),
      ylim=(0,1), grid=True, legend=False, label='cumulative distribution')
    plt.fill_between([t_min, t_split], 0, 1, color='C1', alpha=.5, 
                     label='maturing interval')
    plt.fill_between([t_split, t_max], 0, 1, color='C2', alpha=.5, 
                     label='probing interval')
    plt.legend()
    
def class_imbalance(targets: np.ndarray) -> pd.DataFrame:
  """Report the class imbalance in the targets."""
  return pd.DataFrame({
    'absolute': pd.Series(targets).value_counts(),
    'relative': pd.Series(targets).value_counts(normalize=True)})
 
def _teexmaster(g: nx.Graph, fraction=.01):
  """Usage: average_path_length, diameter = teexmaster(graph)"""
  with tempfile.NamedTemporaryFile() as tmp:
    nx.write_edgelist(nx.convert_node_labels_to_integers(g), tmp.name, 
                      delimiter='\t', data=False)
    cp = subprocess.run(
      ['./teexgraph/teexgraph'], 
      input=f'load_undirected {tmp.name}\nest_dist_distri {fraction}', 
      encoding='ascii', stdout=subprocess.PIPE)
  path_distribution = np.array([int(item)
                                for line in cp.stdout.split('\n')
                                for item in line.split('\t') if item != ''
                               ]).reshape(-1,2)
  return path_distribution
  
def _diameter(g: nx.Graph):
  """Usage: average_path_length, diameter = teexmaster(graph)"""
  with tempfile.NamedTemporaryFile() as tmp:
    nx.write_edgelist(nx.convert_node_labels_to_integers(g), tmp.name, 
                      delimiter='\t', data=False)
    cp = subprocess.run(
      ['./teexgraph/teexgraph'], 
      input=f'load_undirected {tmp.name}\ndiameter', 
      encoding='ascii', stdout=subprocess.PIPE)
  return int(cp.stdout.split()[0])

def _gc(g: nx.Graph) -> nx.Graph:
  """Return giant component."""
  return g.subgraph(max(nx.connected_components(g), key=len)).copy()
  
def _weighted_mean(array: np.ndarray) -> float: 
  return np.average(np.array(range(len(array))), weights=array)
    
class NetworkStats(typing.NamedTuple):
  path_distribution: pd.Series
  stats: pd.Series
    
def network_stats(edgelist_dict: dict[str, pd.DataFrame], *,
                  fraction: float = 1, 
                  path: typing.Optional[str] = None,
                  verbose: bool = False
                  ) -> dict[str, NetworkStats]:
  """Report some statistics for the given graphs. 
  
  Args:
    edgelist_dict: Dictionary keyed by label and with edgelist as value.
    fraction: Optional; Use only this fraction of nodes to determine the path
      distribution and hence metrics like diameter and average_path_length.
    path: Store intermediate results in this directory. Will create directory if
      it does not exists.
    verbose: Optional; If true show tqdm progressbar.
      
  Returns:
    NetworkStats: Dict-in-dict, with the outer dict keyed by label of the graph
      and inner dict containing keys path_distribution and statistics.  
  
  Usage:
    path_distribution, stats = tlp.analysis.network_stats(edgelist)
  """
  if path is not None: os.makedirs(path)
  results = dict()
  iterator = tqdm(edgelist_dict.items(), disable=not verbose, unit='graph')
  for name, edgelist in iterator:
    iterator.set_description(name)
    multigraph = nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph)
    simplegraph = nx.from_pandas_edgelist(edgelist)
    simplegraph_gc = _gc(simplegraph)
    multigraph_gc = _gc(multigraph)
    path_distribution = _teexmaster(simplegraph, fraction=fraction)[:,1]
    average_path_length = _weighted_mean(path_distribution)
    diameter = len(path_distribution)
    n_e = multigraph.number_of_edges()
    s_n = simplegraph.number_of_edges()
    result = NetworkStats(
      path_distribution=pd.Series(path_distribution, name=name),
      stats = pd.Series(
        {'nodes': simplegraph.number_of_nodes(), 
         'nodes (GC)': multigraph_gc.number_of_nodes(), 
         'edges': n_e,
         'edges (GC)': multigraph_gc.number_of_edges(), 
         'pairs of nodes connected': s_n,
         'pairs of nodes connected (GC)': simplegraph_gc.number_of_edges(),
         'average number of edges between connected node pairs': n_e/s_n,
         'density (multigraph)': nx.density(multigraph),
         'density (multigraph, GC)': nx.density(multigraph_gc),
         'density (simplegraph)': nx.density(simplegraph),
         'density (simplegraph, GC)': nx.density(simplegraph_gc),
         'assortativity (multigraph)': (
           nx.degree_assortativity_coefficient(multigraph)),
         'assortativity (simplegraph)': (
           nx.degree_assortativity_coefficient(simplegraph)),
         'diameter': diameter, 
         'average path length': average_path_length,
         'average clustering coefficient (simplegraph)': (
           nx.average_clustering(simplegraph)),
         'average clustering coefficient (multigraph)': (
           nx.average_clustering(multigraph))},
        name=name))
    if path is not None:
      joblib.dump(result, os.path.join(path, name + '.pkl'))
    results[name] = result
  return results
  
def plot_path_distributions(df: pd.DataFrame, smooth: bool = True) -> None:
  """Plot the path distributions in one plot.
  
  Args:
    df: pd.DataFrame containing the path distributions for various graphs.
    smooth: Optional; If true, make the graph more smooth.
    
  Typical usage is with the function tlp.analysis.network_stats:
    edgelist_dict = {
      'complete_graph': joblib.load(f'{dataset_id}/edgelist.pkl'),
      'mature_graph': joblib.load(f'{dataset_id}/edgelist_mature.pkl')}
    df = tlp.analysis.network_stats(edgelist_dict, path=stats')
    tlp.analysis.plot_path_distributions(df)
  """
  df = df.apply(lambda x: x/x.sum(), axis='index')
  if smooth:
    df = (
      df
      .reindex(labels=np.arange(0, df.index.max(), .1))
      .interpolate('akima', limit_area='inside')
      .dropna(how='all'))

  with plt.rc_context({'xtick.top': True, 'ytick.right': True}):
    ax = plt.gca()
    df.plot(kind='area', ax=ax, xlim=(1), ylim=(0), stacked=False, alpha=.5, 
            title='Path distribution', xlabel='Path length', 
            ylabel='Probability')
    assert isinstance(ax, matplotlib.axis.Axis)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())

def plot_datetime_strategies(
  datetimes: pd.Series, *,
  time_strategies: dict[str, Strategy] = TIME_STRATEGIES):
  """Plot the mapping from datetime with the various time strategies.
  
  Args:
    datetimes: pd.Series containing all the considered datetimes (one per edge)
    time_strategies: Optional; A list containing functions that can map the
      datetime column of the edgelist to a float. See 
      tlp.feature.TIME_STRATEGIES.
  
  Usage:
    tlp.analysis.plot_datetime_strategies(edgelist['datetime'])
  """
  index = pd.to_datetime(
    np.linspace(datetimes.min().value, datetimes.max().value))
  with plt.rc_context({'xtick.top': True, 'ytick.right': True}):
    _, ax = plt.subplots(figsize=(4,4))
    df = pd.DataFrame(
      {str: func(index) for str, func in time_strategies.items()}, 
      index=index)
    df.plot(ax=ax, xlabel='Year', ylabel='Proportion',  
            xlim=datetimes.agg(['min', 'max']), ylim=(0,1))
    
    ax.legend(title='Time strategy') # type: ignore
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1)) # type: ignore
    
def plot_datetime_distribution(
  datetimes: pd.Series, *,
  time_strategies: dict[str, Strategy] = TIME_STRATEGIES) -> None:
  """Plot the distribution of the transformed datetime values by the various
  time strategies.
  
  Args:
    datetimes: pd.Series containing all the considered datetimes (one per edge).
    time_strategies: Optional; A list containing functions that can map the
      datetime column of the edgelist to a float. See 
      tlp.feature.TIME_STRATEGIES.
  
  Usage:
    tlp.analysis.plot_datetime_distribution(edgelist['datetime'])
  """
  data = pd.concat(
    {time_str: time_func(datetimes) 
     for time_str, time_func in time_strategies.items()}, 
    names=['Time strategy', 'index']
    )
  data = data.reset_index('Time strategy').reset_index(drop=True)
  rc = {'xtick.top': True, 'ytick.right': True, 'figure.figsize': (4,4)}
  with plt.rc_context(rc), sns.axes_style('ticks'):  
    ax = sns.ecdfplot(data=data, x='datetime', hue='Time strategy')
    ax.set_xlim((0,1)) # type: ignore
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1)) # type: ignore
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1)) # type: ignore
    
def plot_score(feature_dict: dict[Experiment, pd.Series], targets: np.ndarray
               ) -> None:
  """Plot the scores as obtained in feature construction.
  
  Args:
    feature_dict: A dict such as obtained from all methods in tlp.features.
    targets: np.array containing the targets belonging to the instances used in
      feature construction.
  
  Usage:
    tlp.analysis.plot_score(feature_dict, targets)
  """
  data = pd.concat(
    [pd.DataFrame(
      {'Ground truth': targets, 'Score': scores, **experiment._asdict()}) 
      for experiment, scores in feature_dict.items()],
    ignore_index=True)
  data['Feature'] = (
    data['feature'] + ' ' +
    data['time_aware'].replace({True: '(time-aware)', False: '(time-agnostic)'})
    + ' ' + data['time_strategy'] + ' ' + data['aggregation_strategy'])
  
  g = sns.displot(
    data=data, x='Score', hue='Ground truth', col='Feature', kind='kde', 
    palette=['red', 'green'], height=4.5, col_wrap=6,
    rug=True, rug_kws=dict(height=-.01, clip_on=False), 
    facet_kws=dict(sharex=False, sharey=False, despine=False))
  g.set_titles('{col_name}')
  
def plot_roc_auc(feature_dict: dict[Experiment, pd.Series], targets: np.ndarray
                 ) -> None:
  """Plot the ROC curves.

  Args:
      feature_dict (dict[Experiment, pd.Series]): A dict such as obtained from 
        all methods in tlp.features.
      targets (np.ndarray): The targets belonging to the instances used in 
        feature construction.
  """
  data = dict()
  no_positives = np.sum(targets)
  for experiment, scores in feature_dict.items():
    fpr, tpr, _ = (
      sklearn.metrics.roc_curve(y_true=targets, y_score=scores)) # type: ignore 

    rocauc = pd.Series(tpr*no_positives, 
                      index=pd.Index(fpr*no_positives, name='False positives'))
    rocauc = rocauc[~rocauc.index.duplicated()]
    data[experiment] = rocauc
    
  data = pd.DataFrame(data).interpolate().reset_index().melt(
    id_vars='False positives',
    var_name=list(Experiment._fields), # type: ignore
    value_name='True positives')
  
  g = sns.relplot(
    data=data, x='False positives', y='True positives', hue='time_aware', 
    col='feature', palette={True: 'green', False: 'red'}, aspect=1, kind='line', 
    height=4, ci=100)
  g.map(plt.axline, xy1=(0,0), xy2=(1,1), c='black') # type: ignore
  g.set(xlim=(0,no_positives), ylim=(0,no_positives), xlabel='False Positives', 
        ylabel='True positives')
  
def get_auc(feature_dict: dict[Experiment, pd.Series], targets: np.ndarray
            ) -> pd.Series:
  """Get the auc performance of the given features.

  Args:
    feature_dict (dict[Experiment, pd.Series]): A dict such as obtained from 
      all methods in tlp.features.
    targets (np.ndarray): The targets belonging to the instances used in 
      feature construction.

  Returns:
      pd.DataFrame: AUC performance. The columns are the same as the
        attributes of the Experiment class, except for one added column 'auc'.
  """  
  data = pd.Series(
    [sklearn.metrics.roc_auc_score(targets, scores) # type: ignore
      for scores in feature_dict.values()], 
    index=pd.Index(feature_dict, name=Experiment._fields), # type: ignore
    name='auc') 
  return data
    
def cheap_statistics(edgelist_file: str, output_path: str, verbose: bool = False
  ) -> None:
  """Print some statistics that are relatively cheap to compute.
  This method is made for graphs that have multiple edges between
  nodes (nx.MultiGraph).
  """
  result = dict()
  
  if verbose: print(f'{datetime.datetime.now()} Get edgelist')
  edgelist = joblib.load(edgelist_file)
  
  if verbose: print(f'{datetime.datetime.now()} Get nx.MultiGraph')
  multigraph = nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph)
  result['edges'] = multigraph.number_of_edges()
  # if verbose: print(f'{datetime.datetime.now()} Get degree assortativity')
  # result['degree assortativity (nx.MultiGraph)'] = (
  #   nx.degree_assortativity_coefficient(multigraph))
  if verbose: print(f'{datetime.datetime.now()} Get density')
  result['density (nx.MultiGraph)'] = nx.density(multigraph)
  
  if verbose: print(f'{datetime.datetime.now()} Get GC of nx.MultiGraph')
  multigraph_gc = _gc(multigraph)
  del multigraph
  multigraph_gc_number_of_edges = multigraph_gc.number_of_edges()
  result['fraction edges in GC'] = (
    multigraph_gc_number_of_edges / result['edges'])
  del multigraph_gc
  
  if verbose: print(f'{datetime.datetime.now()} Get nx.Graph')
  simplegraph = nx.from_pandas_edgelist(edgelist)
  del edgelist
  result['nodes'] = simplegraph.number_of_nodes()
  result['avg events per pair'] = (
    result['edges'] / simplegraph.number_of_edges())
  if verbose: print(f'{datetime.datetime.now()} Get density')
  result['density (nx.Graph)'] = nx.density(simplegraph)
  if verbose: print(f'{datetime.datetime.now()} Get degree assortativity')
  result['degree assortativity (nx.Graph)'] = (
    nx.degree_assortativity_coefficient(simplegraph))
  if verbose: print(f'{datetime.datetime.now()} Get clustering coefficient')
  result['average clustering coefficient'] = (
    nx.average_clustering(simplegraph))
  
  if verbose: print(f'{datetime.datetime.now()} Get GC of nx.Graph')
  simplegraph_gc = _gc(simplegraph)
  del simplegraph
  result['fraction nodes in GC'] = (
    simplegraph_gc.number_of_nodes() / result['nodes'])
  result['avg events per pair in GC'] = (
    multigraph_gc_number_of_edges / simplegraph_gc.number_of_edges())
  
  if verbose: print(f'{datetime.datetime.now()} Store results')
  with open(os.path.join(output_path, 'stats.json'), 'w') as file:
    json.dump(result, file)