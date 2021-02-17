import collections.abc
import os
import typing

import joblib
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

from . import AGGREGATION_STRATEGIES, TIME_STRATEGIES, NODEPAIR_STRATEGIES
from .experiment import Experiment

def diff(x): return x[1] - x[0]

def _node_attributes_single_strategy(
  graph: nx.Graph, 
  instances: typing.Iterable[tuple[int, int]], 
  nodepair_strategy: typing.Callable, 
  aggregation_strategy: typing.Callable, 
  verbose: bool):  
  result = list()
  iterator = tqdm(instances, leave=False, disable=not verbose, unit='nodepair')
  for u, v in iterator:
    activity_u = aggregation_strategy(
      [
        edge_attributes['datetime_transformed'] 
        for nb in graph[u]
        for edge_attributes in graph.get_edge_data(u, nb).values()
      ]
    )
    activity_v = aggregation_strategy(
      [
        edge_attributes['datetime_transformed'] 
        for nb in graph[v]
        for edge_attributes in graph.get_edge_data(v, nb).values()
      ]
    )
    result.append(nodepair_strategy([activity_u, activity_v]))
  return result

def na(
  path: str, *, 
  nodepair_strategies: dict[str, typing.Callable] = NODEPAIR_STRATEGIES,
  aggregation_strategies: dict[str, typing.Callable] = AGGREGATION_STRATEGIES,
  time_strategies: dict[str, typing.Callable] = TIME_STRATEGIES,
  verbose: bool = False
) -> None:
  """Determine some features based on node activity. This feature is calculated
  over all possible combinations of the aggregation_strategies, 
  nodepair_strategies, and time_strategies.

  Args:
    path
  """
  file = os.path.join(path, 'NA.pkl')
  os.makedirs(path, exist_ok=True)
  if os.path.isfile(file):
    return
  
  result = dict()
  for time_str, time_func in tqdm(time_strategies.items(), 
                                  leave=False, 
                                  disable=not verbose,
                                  desc='Time strategy'):
    edgelist['datetime_transformed'] = time_func(edgelist['datetime'])
    graph = nx.from_pandas_edgelist(edgelist, edge_attr=True, 
                                    create_using=nx.MultiGraph)
    for nodepair_str, nodepair_func in tqdm(nodepair_strategies.items(), 
                                            leave=False, 
                                            disable=not verbose,
                                            desc='Nodepair strategy'):
      for agg_str, agg_func in tqdm(aggregation_strategies.items(), 
                                    leave=False, 
                                    disable=not verbose,
                                    desc='Aggregation strategy'):
        experiment = Experiment(
          feature='NA',
          time_aware=True,
          aggregation_strategy=agg_str,
          time_strategy=time_str,
          nodepair_strategy=nodepair_str
        )
        result[experiment] = (
          _node_attributes_single_strategy(
            graph=graph, 
            instances=instances, 
            nodepair_strategy=nodepair_func,
            aggregation_strategy=agg_func,
            verbose=verbose
          )
        )
  joblib.dump(result, file)
  