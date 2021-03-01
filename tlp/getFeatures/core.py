import os
import typing

import joblib
import numpy as np
import pandas as pd

from ..helpers import load, print_status

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
  nodepair_strategy: typing.Optional[str] = None
  
def recursive_lookup(path: str) -> dict[Experiment, np.ndarray]:
  result = dict()
  for file in os.listdir(path):
    filepath = os.path.join(path, file)
    if os.path.isfile(filepath):
      result.update(joblib.load(filepath))
  return result

def get_edgelist_and_instances(
  path: str, *, check_for_datetime: bool = True, verbose: bool = False
  ) -> tuple[pd.DataFrame, np.ndarray]:
  """Return the edgelist_mature and instances_samples stored in path. Test 
  various properties the objects should have.

  Args:
    path
    check_for_datetime: Optional; If true, check if datetime column exists in
      edgelist_mature. Defaults to True.
    verbose: Optional; Defaults to False.
  
  Usage:
  edgelist_mature, instances = get_edgelist_and_instances(path)"""
  # Read in
  edgelist_mature_file = os.path.join(path, 'edgelist_mature.pkl')
  if verbose: print_status(f'Read {edgelist_mature_file}')
  edgelist_mature = pd.read_pickle(edgelist_mature_file)

  instances_file = os.path.join(path, 'instances_sampled.npy')
  if verbose: print_status(f'Read {instances_file}')  
  instances_sampled = np.load(instances_file)

  if check_for_datetime:
    cols = ['source', 'target', 'datetime']
  else:
    cols = ['source', 'target']
  assert all(col in edgelist_mature.columns for col in cols)

  assert instances_sampled.ndim == 2
  assert instances_sampled.shape[1] == 2

  return edgelist_mature, instances_sampled