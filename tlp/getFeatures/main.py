from .aa_time_agnostic import aa_time_agnostic
from .aa_time_aware import aa_time_aware
from .na import na
from .sp import sp
from .strategies import (AGGREGATION_STRATEGIES, NODEPAIR_STRATEGIES,
                         TIME_STRATEGIES, Strategies)
from ..helpers import print_status, file_exists

import os

from tqdm.auto import tqdm

def get_features(
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
  index = path.split('/')[1]
  print_status(f'#{index} Start')

  # First check if we can begin.
  required_files = [
    os.path.join(path, file)
    for file in ['edgelist_mature.pkl', 'instances.npy']
  ]
  for required_file in required_files:
    if not os.path.isfile(required_file):
      print_status(f'#{index}: {required_file} does not exist.')
      return
  
  # Feature 1
  print_status(f'#{index} aa_time_agnostic')
  aa_time_agnostic(path, verbose=verbose)
  
  # Feature 2
  print_status(f'#{index} aa_time_aware')
  aa_time_aware(
    path, 
    aggregation_strategies=aggregation_strategies,
    time_strategies=time_strategies,
    verbose=verbose)

  # Feature 3
  print_status(f'#{index} na')
  na(
    path,
    nodepair_strategies=nodepair_strategies,
    aggregation_strategies=aggregation_strategies,
    time_strategies=time_strategies,
    verbose=verbose
  )

  # Feature 4
  print_status(f'#{index} sp')
  sp(path, verbose=verbose)
  
  print_status(f'#{index} done')