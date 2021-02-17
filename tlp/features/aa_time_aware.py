import collections.abc
import os
import typing

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .strategies import AGGREGATION_STRATEGIES, TIME_STRATEGIES, Strategy
from .experiment import Experiment

def aa_time_aware(
  edgelist: pd.DataFrame, 
  instances: collections.abc.Iterable, *,
  output_path: str,
  aggregation_strategies: dict[str, Strategy] = AGGREGATION_STRATEGIES,
  time_strategies: dict[str, Strategy] = TIME_STRATEGIES,
  verbose: bool = False
  ) -> None:
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
    output_path: Optional; If provided, store the result at path/AA_time_aware.pkl.
    verbose: Optional; If true, show tqdm progressbar.
  
  Stores at os.path.join(output_path, 'AA_time_aware.pkl'):
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """
  file = os.path.join(output_path, 'AA_time_aware.pkl')
  os.makedirs(output_path, exist_ok=True)
  if os.path.isfile(file):
    return None
  
  result = dict()
  for time_str, time_func in tqdm(time_strategies.items(), 
                                  desc='time strategies', 
                                  disable=not verbose, 
                                  leave=False):
    edgelist['datetime_transformed'] = time_func(edgelist['datetime']) 
    graph = nx.from_pandas_edgelist(
      edgelist, edge_attr=True, create_using=nx.MultiGraph)
    for agg_str, agg_func in tqdm(aggregation_strategies.items(), 
                                  desc='aggregation strategies', 
                                  leave=False,
                                  disable=not verbose):
      experiment = Experiment(
          'AA', time_aware=True, aggregation_strategy=agg_str, 
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
  joblib.dump(result, file)
