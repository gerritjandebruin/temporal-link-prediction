import os
import pickle

import networkx as nx
import numpy as np

from .core import get_edgelist_and_instances, Experiment
from ..helpers import file_exists

def aa_time_agnostic(path:str, verbose: bool = False) -> None:
  """Calculate the time agnostic Adamic Adar feature. The result is a dict keyed
  on a NamedTuple ()
  
  Args:
    path: The path should contain edgelist_mature.pkl and 
      instances_sampled.npy. Result is stored at path/aa_time_agnostic.pkl.
  
  Result:
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """  
  # Check if result already present. If so, quit.
  feature_path = os.path.join(path, 'features')
  file = os.path.join(feature_path, 'aa_time_agnostic.pkl')
  if file_exists(file, verbose=verbose): return
  
  # Create folder if it does not exist.
  os.makedirs(feature_path, exist_ok=True)

  # Read in
  edgelist_mature, instances = get_edgelist_and_instances(
    path, check_for_datetime=False, verbose=verbose)

  graph_mature = nx.from_pandas_edgelist(edgelist_mature)

  # Check
  for node in instances.flat:
    assert node in graph_mature

  # Calculation
  scores = np.array(
    [p for u, v, p in nx.adamic_adar_index(graph_mature, instances)])
  result = {Experiment('aa', time_aware=False): scores}

  # Store
  with open(file, 'wb') as file:
    pickle.dump(result, file)
