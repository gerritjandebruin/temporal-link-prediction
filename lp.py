import collections
import subprocess
import tempfile
import typing

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

LOWER_BOUND = 0.2
SPLIT_FRACTION = 2/3

TIME_STRATEGIES = {
  'lin': lambda x: rescale(
    rescale(x.astype(int)), lower_bound=LOWER_BOUND), 
  'exp': lambda x: rescale(
    exp_time(rescale(x.astype(int))), lower_bound=LOWER_BOUND), 
  'sqrt': lambda x: rescale(
    np.sqrt(rescale(x.astype(int))), lower_bound=LOWER_BOUND)}

AGGREGATION_STRATEGIES = {
  'mean': np.mean, 'sum': np.sum, 'max': np.max, 'median': np.median}

class Experiment(typing.NamedTuple):
  feature: str
  scores : np.array
  aggregation_strategy: str = np.nan
  time_strategy: str = np.nan
  
  def __repr__(self) -> str:
    return f'<Experiment {self.feature}, {self.aggregation_strategy}, {self.time_strategy}>'
    
  def get_parameters(self) -> dict:
    """Return all the parameters of the experiment in a dict."""
    return dict(feature=self.feature, 
                aggregation_strategy=self.aggregation_strategy, 
                time_strategy=self.time_strategy)

def exp_time(x: np.array) -> np.array:
  """Apply y=3*exp(x) and normalize it between (0,1)."""
  return np.exp(3*x) / np.exp(3)

def rescale(x: np.array, *, lower_bound: float = 0) -> np.array:
  """Rescale the provided array.

  Args:
    lower_bound: Instead of normalizing between 0 and 1, normalize between 
      lower_bound and 1.
  """
  lowest, highest = np.quantile(x, [0, 1])
  return lower_bound + (1-lower_bound)*(x-lowest)/(highest-lowest)

def konect_to_pandas(file: str, sep: str, datetime_with_space: bool = False
                     ) -> pd.DataFrame:
  """Convert the file in the format as obtained from KONECT to pd.DataFrame with
  the columns u, v, datetime.

  Args:
    file: File as downloaded from KONECT. Usual name is out.*.
    sep: Character used to split the data into columns.
    time_with_space: Optional; If true, expect that the datetime column is
      separated with a space instead of sep.
  """
  edgelist = pd.read_csv(file, sep=sep, comment='%', 
                         names=['u', 'v', 'weight', 'datetime'])
  edgelist = edgelist[edgelist['datetime'] != 0]
  
  # Check of both u->v and v->u are present for every edge.
  edgeset = {(u,v) for u, v in edgelist[['u', 'v']].itertuples(index=False)}
  assert np.all(
    [edge in edgeset for edge in edgelist[['u', 'v']].itertuples(index=False)])
  
  if datetime_with_space: 
    edgelist['datetime'] = (
      edgelist['weight'].str.split(expand=True)[1].astype(int))
    edgelist['weight'] = (
      edgelist['weight'].str.split(expand=True)[0].astype(int))

  assert (edgelist['weight'] == 1).all()
  
  edgelist['datetime'] = pd.to_datetime(edgelist['datetime'], unit='s')
  return edgelist.drop(columns=['weight'])

def plot_interval(datetimes: pd.Series, *, 
                  t_min: typing.Optional[pd.Timestamp] = None,
                  t_split: typing.Optional[pd.Timestamp] = None,
                  t_max: typing.Optional[pd.Timestamp] = None) -> None:
  """Plot the maturing and probing interval.

  Args:
    datetimes: pd.Series containing pd.Date
    t_min, t_split, t_max: Optional; Dates corresponding to the start of the
      maturing interval, end of the maturing interval and end of the probing
      interval, respectively.

  Usage:
   plot_interval(edgelist['datetimes'])
  """
  t_min_, t_split_, t_max_ = datetimes.quantile([0,SPLIT_FRACTION,1])
  if t_min is None: t_min = t_min_
  if t_split is None: t_split = t_split_
  if t_max is None: t_max = t_max_

  with plt.rc_context({'xtick.top': True, 'ytick.right': True, 
                     'figure.figsize': (20, 4)}):
    plt.fill_between((t_min, t_split), 0, 1, color='C1', alpha=.5, 
                     label='maturing interval')
    plt.fill_between((t_split, t_max), 0, 1, color='C2', alpha=.5, 
                     label='probing interval')
    ax = (datetimes.value_counts(normalize=True).sort_index().cumsum()
          .plot(xlabel='Year', ylabel='CDF', 
                xlim=(datetimes.min(), datetimes.max()), ylim=(0,1),
                grid=True, legend=False, label='cumulative distribution'))
    plt.legend()

def plot_time_strategies(datetimes: pd.Series) -> None:
  """Plot the mapping from datetime with the various time strategies."""
  index = pd.to_datetime(
    np.linspace(datetimes.min().value, datetimes.max().value))
  with plt.rc_context({'xtick.top': True, 'ytick.right': True}):
    ax = pd.DataFrame(
      {time_str: time_func(np.array(index))
      for time_str, time_func in TIME_STRATEGIES.items()}, 
      index=index
    ).plot(xlabel='Year', ylabel='Proportion', figsize=(4,4), 
           xlim=datetimes.agg(['min', 'max']), ylim=(0,1))
    ax.legend(title='Time strategy')
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1))
    
def normalize(x: pd.Series) -> pd.Series:
  """Normalize a pd.Series by subtracting the minimum and dividing by the range. 
  Works also with datetime values.
  """
  after_subtraction = (x - x.min())
  return (after_subtraction/after_subtraction.max())

def not_connected(graph: nx.Graph, *, 
                  cutoff: typing.Optional[int] = None, verbose=False
                 ) -> np.array:
  """Get all pairs of nodes that are not connected.
  
  Args:
    graph
    cutoff: Optional; Return only unconnected pairs of nodes with at most this 
      distance in the graph.
    verbose: Optional; If true, show tqdm progressbar.
    
  Returns:
    A 2d np.array with shape (n, 2) containing the unconnected pairs of nodes.
  """
  return np.array([
    (node, neighborhood)
    for node 
    in tqdm(graph, disable=not verbose, 
            desc='Collecting unconnected pairs of nodes', leave=False)
    for neighborhood, distance 
    in nx.single_source_shortest_path_length(graph, node, cutoff=cutoff).items() 
    if distance > 1 and node < neighborhood])

def split_in_intervals(
  edgelist: pd.DataFrame, *, 
  datetime_col: str = 'datetime', 
  t_min: typing.Optional[pd.Timestamp] = None,
  t_split: typing.Optional[pd.Timestamp] = None,
  t_max: typing.Optional[pd.Timestamp] = None
  ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
  """Split the provided edgelist into the edges belonging to the maturing and
  probing interval.
  
  Args:
    edgelist: DataFrame containing the edges.
    datetime_col: Optional; The key of the datetime column in the edgelist.
  
  Usage:
    edgelist_maturing, edgelist_probing = split_intervals(edgelist)
  """
  t_min_, t_split_, t_max_ = (
    edgelist[datetime_col].quantile([0,SPLIT_FRACTION,1]))
  if t_min is None: t_min = t_min_
  if t_split is None: t_split = t_split_
  if t_max is None: t_max = t_max_
  return (edgelist.loc[edgelist[datetime_col].between(t_min, t_split)],
          edgelist.loc[edgelist[datetime_col].between(t_split, t_max)])

def gc(g: nx.Graph) -> nx.Graph:
  """Return giant component."""
  return g.subgraph(max(nx.connected_components(g), key=len)).copy()

def weighted_mean(d: np.ndarray) -> float: 
  return np.average(d[:,0], weights=d[:,1])

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

def report(graph: nx.Graph, fraction=.01) -> typing.Tuple[np.array, dict]:
  """Report some statistics of the given graph. Fraction is passed to 
  teexmaster. 
  
  Usage:
    path_distribution, stats = report(graph)"""
  assert type(graph) is nx.MultiGraph
  simplegraph = nx.Graph(graph)
  simplegraph_gc = gc(simplegraph)
  graph_gc = gc(graph)
  path_distribution = _teexmaster(graph, fraction=fraction)
  average_path_length = weighted_mean(path_distribution)
  diameter = path_distribution[:,0].max()
  n_e = graph.number_of_edges()
  s_n = simplegraph.number_of_edges()
  return (
    path_distribution, 
    {'nodes': graph.number_of_nodes(), 
     'nodes (GC)': graph_gc.number_of_nodes(), 
     'edges': n_e,
     'edges (GC)': graph_gc.number_of_edges(), 
     'edges (simplegraph)': s_n,
     'edges (simplegraph, GC)': simplegraph_gc.number_of_edges(),
     'average number of edges between connected node pairs': s_n/n_e,
     'density': nx.density(graph),
     'density (GC)': nx.density(graph_gc),
     'density (simplegraph)': nx.density(simplegraph),
     'density (simplegraph, GC)': nx.density(simplegraph_gc),
     'assortativity': nx.degree_assortativity_coefficient(graph),
     'assortativity (simplegraph)': (
       nx.degree_assortativity_coefficient(simplegraph)),
     'diameter': diameter, 
     'average path length': average_path_length,
     'average clustering coefficient': nx.average_clustering(simplegraph)})

def get_instances(*,
  edgelist_mature: pd.DataFrame, edgelist_probe: pd.DataFrame,
  cutoff: typing.Optional[int] = None,
  verbose: bool = False) -> typing.Tuple[np.array, np.array]:
  """Get the instances along with the targets. The instances are the pairs of
  nodes that are not (yet) connected in the maturing interval. The targets
  indicate whether the pairs of nodes will connect in the probing interval.
  
  Args:
    edgelist
    datetime_col: Optional; The key of the datetime column in the edgelist.
    cutoff: Optional; Return only unconnected pairs of nodes with at most this 
      distance in the graph.
    verbose: Optional; If true, show tqdm progressbar.
    
  Returns:
    instances: np.array with size (n,2)
    targets: np.array with size n
  
  Usage:
    instances, targets = get_instances(edgelist)
  """ 
  graph_mature = (
    nx.from_pandas_edgelist(edgelist_mature, source='u', target='v', 
                            edge_attr=True, create_using=nx.MultiGraph))
  instances = not_connected(graph_mature, cutoff=cutoff, verbose=verbose)
  edgeset_probing = {
    (edge.u, edge.v) 
    for edge in edgelist_probe[['u', 'v']].itertuples(index=False)}
  targets = np.array(
    [(u, v) in edgeset_probing 
     for u, v in tqdm(instances, desc='Determine targets', 
                      disable=not verbose, leave=False)])
  
  return (instances, targets)

def _sample(array: np.array, size: int) -> np.array:
  """Take a sample (with replacement) of a given size n along the first axis of 
  array.
  """
  return array[np.random.randint(len(array), size=size)]

def balanced_sample(index: np.array, y: np.array, *, size: int
                    ) -> typing.Tuple[np.array, np.array]:
  """Take n positive and n negative samples from the index.
  
  Args:
    index: np.array with shape (m,2). From this array the samples are taken.
    y: np.array with shape (m).
    size: Take this number of positive and this number of negative samples.
  
  Returns:
    index: np.array with size (n,2)
    y: np.array with size n
    
  Usage:
    index_sampled, index_y = lp.balanced_sample(index, y, size=10000)
  """
  positives = _sample(index[y], size)
  negatives = _sample(index[~y], size)
  return (
    np.concatenate([negatives, positives]),
    np.concatenate([np.zeros(10000, dtype=bool), np.ones(10000, dtype=bool)]))
  
def adamic_adar_time_agnostic(edgelist: pd.DataFrame, 
                              samples: collections.abc.Iterable,
                              name: str = 'AA (time agnostic)') -> Experiment:
  """Returns the time agnostic Adamic Adar feature for the given samples based
   on the simplegraph.
  
  Args:
    edgelist: A pd.DataFrame with at least columns 'u' and 'v'
    samples: An iterable yielding pairs of nodes.
    name: Optional; The returned pd.Series will have this name.
  
  Returns:
    A named tuple containing string attribute feature and np.array attribute 
      score.
  """
  scores = np.array(
    [p 
     for u, v, p 
     in nx.adamic_adar_index(
      G=nx.from_pandas_edgelist(edgelist, source='u', target='v'), 
      ebunch=samples)])
  return Experiment(name, scores)
  
def _adamic_adar_time_aware_score(
  edgelist: pd.DataFrame, samples: collections.abc.Iterable, *, 
  aggregation_strategy: collections.abc.Callable = AGGREGATION_STRATEGIES['mean'], 
  time_strategy: collections.abc.Callable = TIME_STRATEGIES['lin'], 
  verbose: bool = False,
  **kwargs
  ) -> np.array:
  """Returns the time aware Adamic Adar index for the given pair of nodes on the 
  provided graph, using the specified agg_func and time_func.
  
  Args:
    edgelist: A pd.DataFrame with at least columns 'u', 'v' and 'datetime' used
      to make the graph where the Adamic Adar index is determined from.
    samples: An iterable yielding pairs of nodes.
    aggregation_strategy: Optional; Function used to aggregate datetime info 
      when multiple edges are present between two nodes. See 
      AGGREGATION_STRATEGIES.
    time_strategy: Optional; Function used to map datetime values to output. See 
      TIME_STRATEGIES.
    verbose: Optional
    **kwargs: Optional; Passed to the tqdm iterator.
    
    Returns:
      A np.array containing the scores.
  """
  edgelist['datetime_transformed'] = time_strategy(edgelist['datetime'])
  graph = nx.from_pandas_edgelist(
    edgelist, source='u', target='v', edge_attr=True, 
    create_using=nx.MultiGraph)
  return np.array([
    sum(
      [aggregation_strategy(
        [edge_attributes['datetime_transformed']
         for edge_attributes in graph.get_edge_data(u, z).values()]) *
       aggregation_strategy(
         [edge_attributes['datetime_transformed'] 
          for edge_attributes in graph.get_edge_data(v, z).values()]) /
       np.log(len(list(graph.neighbors(z))))
       for z in nx.common_neighbors(graph, u, v)]
    ) for u, v in tqdm(samples, disable=not verbose, **kwargs)])

def adamic_adar_time_aware(
  edgelist: nx.MultiGraph, samples: collections.abc.Iterable, *, 
  aggregation_strategies: 
    dict[str, collections.abc.Callable] = AGGREGATION_STRATEGIES,
  time_strategies: 
    dict[str, collections.abc.Callable] = TIME_STRATEGIES,
  verbose: bool = False,
  name: str = 'AA (time aware)'
) -> list[Experiment]:
  """Returns the time aware Adamic Adar index for the given pair of nodes on the 
    provided graph, using the specified aggregation and time strategies.
    
    Args:
      graph
      samples: An iterable yielding pairs of nodes.
      aggregation_strategies: Optional; Function used to aggregate datetime info 
        when multiple edges are present between two nodes. See 
        lp.AGGREGATION_STRATEGIES.
      time_strategies: Optional; Function used to map datetime values to output. 
        See lp.TIME_STRATEGIES.
      verbose: Optional
      name: Optional; Used to specify the experiment feature name.
      
      Returns:
        A list of named tuples Experiment containing string attributes feature, 
        time_strategy and aggregation_strategy and np.array attribute score.
  """  
  return [
    Experiment(
      feature=name, time_strategy=time_str, aggregation_strategy=agg_str,
      scores=_adamic_adar_time_aware_score(
        edgelist, samples, aggregation_strategy=agg_func, time_strategy=time_func, 
        verbose=verbose, position=2, leave=False))
    for time_str, time_func in tqdm(time_strategies.items(), position=0, 
                                    disable=not verbose)
    for agg_str, agg_func in tqdm(aggregation_strategies.items(), position=1, 
                                  leave=False, disable=not verbose)]