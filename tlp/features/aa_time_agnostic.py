import collections.abc
import os

import joblib
import networkx as nx
import pandas as pd

from . import Experiment, get_edgelist_and_instances

def aa_time_agnostic(path:str, *, verbose: bool = False) -> None:
  """Calculate the time agnostic Adamic Adar feature.
  
  Args:
    path: The path should contain edgelist_mature.pkl and 
      instances_sampled.npy. Result is stored at path/AA_time_agnostic.pkl.
    verbose: Optional; Defaults to False.
  
  Stores at os.path.join(output_path, 'AA_time_agnostic.pkl'):
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """
  file = os.path.join(path, 'AA_time_agnostic.pkl')
  os.makedirs(path, exist_ok=True)

  if os.path.isfile(file):
    return

  edgelist_mature, instances_sampled = get_edgelist_and_instances(
    path, check_for_datetime=False)

  G = nx.from_pandas_edgelist(edgelist_mature)
  scores = pd.Series(
    [p for u, v, p in nx.adamic_adar_index(G, instances_sampled)])
  result = {Experiment('AA', time_aware=False): scores}
  joblib.dump(result, file)
