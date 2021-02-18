import collections.abc
import os

import joblib
import networkx as nx
import pandas as pd

from .core import get_edgelist_and_instances, Experiment

def aa_time_agnostic(path:str, *, verbose: bool = False) -> None:
  """Calculate the time agnostic Adamic Adar feature. The result is a dict keyed
  on a NamedTuple ()
  
  Args:
    path: The path should contain edgelist_mature.pkl and 
      instances_sampled.npy. Result is stored at path/aa_time_agnostic.pkl.
    verbose: Optional; Defaults to False.
  
  Result:
    A dict with as key a NamedTuple (Experiment) and as value a np.array
      containing the scores.
  """
  file = os.path.join(path, 'aa_time_agnostic.pkl')
  os.makedirs(path, exist_ok=True)

  if os.path.isfile(file):
    return

  edgelist, instances = get_edgelist_and_instances(
    path, check_for_datetime=False)

  G = nx.from_pandas_edgelist(edgelist)

  scores = pd.Series([p for u, v, p in nx.adamic_adar_index(G, instances)])
  result = {Experiment('aa', time_aware=False): scores}

  joblib.dump(result, file)
