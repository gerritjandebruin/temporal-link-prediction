from .aa_time_agnostic import aa_time_agnostic
from .aa_time_aware import aa_time_aware
from .na import na
from .sp import sp
from .strategies import (AGGREGATION_STRATEGIES, NODEPAIR_STRATEGIES,
                         TIME_STRATEGIES, Strategies)
from ..helpers import print_status, file_exists

import os

def construction(
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
  if verbose: print_status('Collect all features.')
  
  # First check if we can begin.
  if not file_exists(os.path.join(path, 'edgelist_mature.pkl'), 
                     verbose=verbose): 
    return
  if not file_exists(os.path.join(path, 'instances_sampled.npy'),
                     verbose=verbose):
    return
  
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
  sp(path, verbose=verbose)