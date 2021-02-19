from .sampling import balanced_sample
import typing

import pandas as pd

from .splitting import split_in_intervals
from .instances import get_instances
from .targets import get_targets
from .sampling import balanced_sample

SPLIT_FRACTION = 2/3

def data_preparation(
  path: str, *,
  split_fraction: typing.Optional[float] = SPLIT_FRACTION,
  t_min: typing.Optional[pd.Timestamp] = None,
  t_split: typing.Optional[pd.Timestamp] = None,
  t_max: typing.Optional[pd.Timestamp] = None,
  cutoff: typing.Optional[int] = 2,
  sample_size: typing.Optional[int] = 10000,
  verbose: bool = False
  ) -> None: 
  """Apply the entire data preparation pipeline, consisting of the following
  steps:
  
  1. Splitting the edgelist into a maturing and probing interval.
  2. Collect the unconnected pairs of nodes from the maturing interval. These
    are the instances used for the link prediction problem.
  3. Determine whether the instances connect in the probing interval (targets).
  4. Sample to balance from the step 3.
  
  Args:
    path: In this path the file edgelist.pkl should be present. This should be
      a pd.DataFrame containing the columns source, target and datetime.
    split_fraction: Optional; If t_split is not provided, the split into the 
      maturing and probing interval will happen such that split_fraction of 
      edges will end up in the maturing interval.
    t_min, t_split, t_max: Optional; Timestamps used to mark the beginning of 
      the maturing interval, the end of the maturing interval and the end of the
      probing interval, respectively.
    cutoff: Optional; Return only unconnected pairs of nodes with at most this 
      distance in the graph. Defaults to 2. If None, return all unconnected
      pairs of nodes.
    sample_size: Optional; Take this number of positive and this number of
      negative samples from the instances. Defaults to 10000. If None, use all
      instances.
    verbose: Optional; If True, show tqdm progressbars. Defaults to False.
    
  """  
  # Step 1
  split_in_intervals(
    path, split_fraction=split_fraction, t_min=t_min, t_split=t_split, 
    t_max=t_max, verbose=verbose
  )
  
  # Step 2
  get_instances(path, cutoff=cutoff, verbose=verbose)
  
  # Step 3
  get_targets(path, verbose=verbose)
  
  # Step 4
  balanced_sample(path, sample_size=sample_size, verbose=verbose)