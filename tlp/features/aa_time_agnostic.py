import os

import joblib
import networkx as nx
import pandas as pd

from .core import get_edgelist_and_instances, Experiment
from ..helpers import print_status, file_exists

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
  if verbose: print_status('Start aa_time_agnostic(...)')
  
  feature_path = os.path.join(path, 'features')
  file = os.path.join(feature_path, 'aa_time_agnostic.pkl')
  if file_exists(file, verbose=verbose): return
  
  os.makedirs(feature_path, exist_ok=True)

  # Read in
  edgelist, instances = get_edgelist_and_instances(
    path, check_for_datetime=False, verbose=verbose)

  if verbose: print_status('Create graph')
  G = nx.from_pandas_edgelist(edgelist)

  # Calculation
  if verbose: print_status('Calculate scores.')
  scores = pd.Series([p for u, v, p in nx.adamic_adar_index(G, instances)])
  result = {Experiment('aa', time_aware=False): scores}

  # Store
  if verbose: print_status('Store result')
  joblib.dump(result, file)
