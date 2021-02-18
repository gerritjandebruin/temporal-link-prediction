import joblib
import os
import typing

import pandas as pd

from ..helpers import print_status, load, file_exists

# Split the edgelist at this quantile into the mature and probe edgelist.
SPLIT_FRACTION = 2/3  

def split_in_intervals(
  path: str, *,
  split_fraction: typing.Optional[float],
  t_min: typing.Optional[pd.Timestamp] = None,
  t_split: typing.Optional[pd.Timestamp] = None,
  t_max: typing.Optional[pd.Timestamp] = None,
  verbose: bool = False
  ) -> None:
  """Split the edgelist into the edges belonging to the maturing and probing 
  interval. The edgelist should be present as a pickled pd.DataFrame at 
  path/edgelist.pkl. The results (two pd.DataFrames) are stored at 
  path/edgelist_{mature/probe}.pkl.
  
  Args:
    path
    split_fraction: Optional; If t_split is not provided, the split into the 
      maturing and probing interval will happen such that split_fraction of 
      edges will end up in the maturing interval.
    t_min, t_split, t_max: Optional; Timestamps used to mark the beginning of 
      the maturing interval, the end of the maturing interval and the end of the
      probing interval, respectively.
    verbose: Optional; Defaults to False.
  """
  if verbose: 
    print_status('Start split_in_intervals(...). Read edgelist.pkl.')
    
  edgelist = load(os.path.join(path, 'edgelist.pkl'), verbose=verbose)
  
  edgelist_mature_file = os.path.join(path, 'edgelist_mature.pkl')
  edgelist_probe_file = os.path.join(path, 'edgelist_probe.pkl')
  if file_exists([edgelist_mature_file, edgelist_probe_file], verbose=verbose): 
    return
  
  if verbose: print_status('Determine split times.')
  if t_min is None:
    t_min = edgelist['datetime'].min()
  if t_split is None:
    assert split_fraction is not None, (
        "Either split_fraction or t_split should be provided.")
    t_split = edgelist['datetime'].quantile(split_fraction)
  if t_max is None:
    t_max = edgelist['datetime'].max()
  
  assert isinstance(t_min, pd.Timestamp)
  assert isinstance(t_split, pd.Timestamp)
  assert isinstance(t_max, pd.Timestamp)
  
  if verbose: print_status('Perform splitting')
  edgelist_mature = edgelist.loc[edgelist['datetime'].between(t_min, t_split)]
  edgelist_probe = edgelist.loc[edgelist['datetime'].between(t_split, t_max)]
  
  if verbose: print_status('Store results')
  edgelist_mature.to_pickle(edgelist_mature_file)
  edgelist_probe.to_pickle(edgelist_probe_file)