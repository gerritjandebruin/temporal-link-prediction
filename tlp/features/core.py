import os
import typing

import joblib
import numpy as np
import pandas as pd

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

def get_edgelist_and_instances(path: str, *, check_for_datetime: bool = True
  ) -> tuple[pd.DataFrame, np.ndarray]:
  """Return the edgelist_mature and instances_samples stored in path. Test 
  various properties the objects should have.

  Args:
    path
    check_for_datetime: Optional; If true, check if datetime column exists in
      edgelist_mature. Defaults to True.
  
  Usage:
  edgelist_mature, instances_sampled = get_edgelist_and_instances(path)"""
  edgelist_mature_path = os.path.join(path, 'edgelist_mature.pkl')
  instances_sampled_path = os.path.join(path, 'instances_sampled.npy')
  assert os.path.isfile(edgelist_mature_path)
  assert os.path.isfile(instances_sampled_path)

  edgelist_mature = joblib.load(edgelist_mature_path)
  instances_sampled = np.load(instances_sampled_path)

  if check_for_datetime:
    cols = ['source', 'target', 'datetime']
  else:
    cols = ['source', 'target']
  assert all(col in edgelist_mature.columns for col in cols)

  assert instances_sampled.ndim == 2
  assert instances_sampled.shape[1] == 2

  return edgelist_mature, instances_sampled