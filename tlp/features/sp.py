import collections
import collections.abc
import os
import typing

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .strategies import AGGREGATION_STRATEGIES, TIME_STRATEGIES
from .experiment import Experiment

def single_source_number_paths_length_2(graph: nx.Graph, source):
  result = collections.Counter()
  for nb in graph[source]: 
    for nnb in graph[nb]: 
      result[nnb] += 1
  return result

def sp(
    edgelist: pd.DataFrame, 
    instances: np.ndarray, 
    output_path: str, 
    index: typing.Optional[int] = None):
  """Determine the number of shortest paths available for each nodepair in 
  instances.

  Args:
      edgelist (pd.DataFrame): [description]
      instances (np.ndarray): [description]
      output_path (str): [description]
      verbose (bool, optional): [description]. Defaults to False.
  """
  filename = os.path.join(output_path, 'sp.pkl')

  # If file already exists, stop.
  if os.path.isfile(filename): 
    tqdm.write(f'{index:02} is skipped')
    return
  
  graph = nx.from_pandas_edgelist(edgelist)
  # Slow method, but providing also shortest paths at greater distance:
#   [len(list(nx.all_shortest_paths(graph, *sample))) 
#    for sample in tqdm(instances, disable=not verbose, leave=False)]
  
  paths_of_length_2 = {
    node: single_source_number_paths_length_2(graph, node) 
    for node 
    in tqdm(instances[:,0], disable=not index, unit='node', desc=f'{index:02}', 
            leave=False, position=index)
  }
  
  experiment = Experiment('Number of shortest paths', time_aware=False)
  result = {experiment: [paths_of_length_2[u][v] for u, v in instances]}
  joblib.dump(result, filename=filename)