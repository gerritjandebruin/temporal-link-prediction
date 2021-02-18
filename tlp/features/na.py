import collections.abc
import os
import typing

import joblib
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

from .core import Experiment, get_edgelist_and_instances
from .strategies import (AGGREGATION_STRATEGIES, NODEPAIR_STRATEGIES,
                         TIME_STRATEGIES, Strategies, Strategy)
from ..helpers import file_exists, print_status


def _node_attributes_single_strategy(
  G: nx.Graph, 
  instances: typing.Iterable[tuple[int, int]], 
  nodepair_strategy: Strategy, 
  aggregation_strategy: typing.Callable, 
  verbose: bool
  ):  

  result = list()

  iterator = tqdm(instances, leave=False, disable=not verbose, unit='nodepair')
  
  for u, v in iterator:
    activity_u = aggregation_strategy(
      [
        edge_attributes['datetime_transformed'] 
        for nb in G[u]
        for edge_attributes in G.get_edge_data(u, nb).values()
      ]
    )
    activity_v = aggregation_strategy(
      [
        edge_attributes['datetime_transformed'] 
        for nb in G[v]
        for edge_attributes in G.get_edge_data(v, nb).values()
      ]
    )
    result.append(nodepair_strategy([activity_u, activity_v]))
  return result

def na(
  path: str, *, 
  nodepair_strategies: Strategies = NODEPAIR_STRATEGIES,
  aggregation_strategies: Strategies = AGGREGATION_STRATEGIES,
  time_strategies: Strategies = TIME_STRATEGIES,
  verbose: bool = False
) -> None:
  """Determine some features based on node activity. This feature is calculated
  over all possible combinations of the aggregation_strategies, 
  nodepair_strategies, and time_strategies. The result is stored at path/na.pkl.

  Args:
    path
    nodepair_strategies: Optional; How to merge the values found for node u and
      v. See tlp.NODEPAIR_STRATEGIES for default strategies.
    aggregation_strategies: Optional; How to aggregate multiple observed events 
      between two nodes to a single value. See tlp.AGGREGATION_STRATEGIES for
      default strategies. Note that this should not make any difference when
      there is only maximum one event per pair of nodes available.
    time_strategies: Optional; A list containing functions that can map the
      datetime column of the edgelist to a float. See tlp.TIME_STRATEGIES for
      default strategies.
  """
  if verbose: print_status('Start na(...')   
                           
  file = os.path.join(path, 'na.pkl')
  if file_exists(file, verbose=verbose): return
  
  os.makedirs(path, exist_ok=True)

  # Read in 
  edgelist, instances = get_edgelist_and_instances(
    path, check_for_datetime=False, verbose=verbose)
  
  result = dict()

  for time_str, time_func in tqdm(time_strategies.items(), leave=False, 
                                  disable=not verbose, desc='Time strategy'):
    edgelist['datetime_transformed'] = time_func(edgelist['datetime'])

    G = nx.from_pandas_edgelist(
      edgelist, edge_attr=True, create_using=nx.MultiGraph)

    for agg_str, agg_func in tqdm(aggregation_strategies.items(), leave=False, 
                                  disable=not verbose,
                                  desc='Aggregation strategy'):
      for nodepair_str, nodepair_func in tqdm(nodepair_strategies.items(), 
                                              leave=False, disable=not verbose,
                                              desc='Nodepair strategy'):
        # Calculation
        experiment = Experiment(
          feature='NA',
          time_aware=True,
          aggregation_strategy=agg_str,
          time_strategy=time_str,
          nodepair_strategy=nodepair_str
        )
        result[experiment] = (
          _node_attributes_single_strategy(
            G=G, 
            instances=instances, 
            nodepair_strategy=nodepair_func,
            aggregation_strategy=agg_func,
            verbose=verbose
          )
        )

  # Store results
  print_status('Store results')
  joblib.dump(result, file)
  