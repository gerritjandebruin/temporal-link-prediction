import collections.abc
import os
import typing

import joblib
import networkx as nx
from networkx.classes.function import get_edge_attributes
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# All transformed datetime values are mapped between LOWER_BOUND and 1.
LOWER_BOUND = 0.2

# All strategies used to transform the datetime values.
def _exp_time(x: np.ndarray) -> np.ndarray:
  """Apply y=3*exp(x) and normalize it between (0,1)."""
  return np.exp(3*x) / np.exp(3)

def _rescale(x: np.ndarray, *, lower_bound: float = 0) -> np.ndarray:
  """_rescale the provided array.

  Args:
    lower_bound: Instead of normalizing between 0 and 1, normalize between 
      lower_bound and 1.
  """
  lowest, highest = np.quantile(x, [0, 1])
  return lower_bound + (1-lower_bound)*(x-lowest)/(highest-lowest)

TIME_STRATEGIES = {
  'lin': lambda x: _rescale(
    _rescale(x.astype(int)), lower_bound=LOWER_BOUND), 
  'exp': lambda x: _rescale(
    _exp_time(_rescale(x.astype(int))), lower_bound=LOWER_BOUND), 
  'sqrt': lambda x: _rescale(
    np.sqrt(_rescale(x.astype(int))), lower_bound=LOWER_BOUND)}

# All strategies used to aggregate multiple edges between two nodes in case of
# graphs with discrete event data.
AGGREGATION_STRATEGIES = {
  'mean': np.mean, 'sum': np.sum, 'max': np.max, 'median': np.median}
  
class Experiment(typing.NamedTuple):
  """Class containing all information to unique identify a given experiment.
  An experiment is the combination of a feature name, a time_aware bool and 
    aggregation and time strategy, if available. Iteration over this object is 
    allowed and yields the values of the attributes. This class can conveniently
    be turned in a dict using the functin _asdict().  
  """
  feature: str
  time_aware: bool
  aggregation_strategy: typing.Optional[str] = None
  time_strategy: typing.Optional[str] = None

def adamic_adar_time_agnostic(edgelist: pd.DataFrame, 
                              instances: collections.abc.Iterable, *,
                              path: typing.Optional[str] = None
                              ) -> dict[Experiment, pd.Series]:
  """Returns the time agnostic Adamic Adar feature for the given instances based
   on the provided edgelist.
  
  Args:
    edgelist: A pd.DataFrame with at least columns 'source' and 'target'.
    instances: An iterable yielding pairs of nodes.
    path: Optional; If provided, store the result at path/AA_time_agnostic.pkl.
  
  Returns:
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """
  if path is None:
    file = None
  else: 
    file = os.path.join(path, 'AA_time_agnostic.pkl')
    os.makedirs(path, exist_ok=True)
    if os.path.isfile(file):
      return joblib.load(file)
    
  result = {
    Experiment('AA', time_aware=False): (
      pd.Series(
        [p 
         for u, v, p 
         in nx.adamic_adar_index(G=nx.from_pandas_edgelist(edgelist), 
                                 ebunch=instances)]))}
  if file is not None: joblib.dump(result, file)
  return result
                                         
Strategy = collections.abc.Callable

def adamic_adar_time_aware(
  edgelist: pd.DataFrame, 
  instances: collections.abc.Iterable, *,
  aggregation_strategies: dict[str, Strategy] = AGGREGATION_STRATEGIES,
  time_strategies: dict[str, Strategy] = TIME_STRATEGIES,
  path: typing.Optional[str] = None,
  verbose: bool = False
  ) -> dict[Experiment, np.ndarray]:
  """Returns the time aware Adamic Adar feature for the given instances based
   on the provided edgelist. This feature is calculated for all the possible
   combinations of the aggregation_strategies and time_strategies.
  
  Args:
    edgelist: A pd.DataFrame with at least columns 'source' and 'target'.
    instances: An iterable yielding pairs of nodes.
    aggregation_strategies: Optional; A list containing functions that can 
      aggregate multiple observed events between two nodes to a single value.
      See AGGREGATION_STRATEGIES.
    time_strategies: Optional; A list containing functions that can map the
      datetime column of the edgelist to a float. See 
      tlp.feature.TIME_STRATEGIES.
    path: Optional; If provided, store the result at path/AA_time_aware.pkl.
    verbose: Optional; If true, show tqdm progressbar.
  
  Returns:
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """
  if path is None:
    file = None
  else: 
    file = os.path.join(path, 'AA_time_aware.pkl')
    os.makedirs(path, exist_ok=True)
    if os.path.isfile(file):
      return joblib.load(file)
  
  result = dict()
  for time_str, time_func in tqdm(time_strategies.items(), 
                                  desc='time strategies', disable=not verbose):
    edgelist['datetime_transformed'] = time_func(edgelist['datetime']) 
    graph = nx.from_pandas_edgelist(edgelist, edge_attr=True, 
                                    create_using=nx.MultiGraph)
    for agg_str, agg_func in tqdm(aggregation_strategies.items(), 
                                  desc='aggregation strategies', leave=False,
                                  disable=not verbose):
      experiment = Experiment('AA', time_aware=True, 
                              aggregation_strategy=agg_str, 
                              time_strategy=time_str)
      result[experiment] = np.array(
        [sum(
          [agg_func(
            [edge_attributes['datetime_transformed']
              for edge_attributes in graph.get_edge_data(u, z).values()]) *
           agg_func(
             [edge_attributes['datetime_transformed'] 
              for edge_attributes in graph.get_edge_data(v, z).values()]) /
           np.log(len(list(graph.neighbors(z))))
           for z in nx.common_neighbors(graph, u, v)])
         for u, v 
         in tqdm(instances, leave=False, disable=not verbose, unit='instances')]
        )
  if file is not None: joblib.dump(result, file)
  return result
  