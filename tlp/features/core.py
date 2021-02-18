import os
import typing

import joblib
import numpy as np
import pandas as pd

from .aa_time_agnostic import aa_time_agnostic
from .aa_time_aware import aa_time_aware
from .na import na
from .strategies import (AGGREGATION_STRATEGIES, NODEPAIR_STRATEGIES,
                         TIME_STRATEGIES, Strategies)


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

def feature_construction(
  path: str, 
  aggregation_strategies: Strategies = AGGREGATION_STRATEGIES,
  time_strategies: Strategies = TIME_STRATEGIES,
  nodepair_strategies: Strategies = NODEPAIR_STRATEGIES,
  verbose: bool = False
  ) -> None:
  """Calculate the following features:
  
  * AA (time agnostic)
  * AA (time aware)
  * Node Activity
  * Shortest Paths

  Args:
    path
    verbose: Optional; Defaults to False.

  The following files should be present in the path:
  - edgelist_mature.pkl, which should contain the columns source, target and 
      datetime. The graph constructed from this edgelist is used to determine
      the features from.
  - instances_sampled.npy, which should be a np.ndarray with shape (n,2). The 
      features are only calculated for each instance in this array.
  """

  # Feature 1
  aa_time_agnostic(path, verbose=verbose)
  
  # Feature 2
  aa_time_aware(
    path, 
    aggregation_strategies=aggregation_strategies,
    time_strategies=time_strategies,
    verbose=verbose)

  # Feature 3
  na(
    path,
    nodepair_strategies=nodepair_strategies,
    aggregation_strategies=aggregation_strategies,
    time_strategies=time_strategies,
    verbose=verbose
  )

  # Feature 4
