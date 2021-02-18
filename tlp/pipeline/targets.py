import os

import joblib
import numpy as np
from tqdm.auto import tqdm

from ..helpers import print_status, load, file_exists

def get_targets(path: str, *, verbose: bool = False) -> None:
  """Get the targets for the provided instances. These targets indicate whether
  the given pairs of nodes (instances) connect in the graph after the probing 
  interval.
  
  In path the files edgelist_probe.pkl and instances.npy should be present. The
  provided edgelist should be a pd.DataFrame containing the columns 'source', 
  'target'. The provided instances.npy should be of shape (n,2). The result is 
  a boolean np.ndarray with shape (n) stored at path/targets.npy.
  
  Args:
    path
    verbose: Optional; If True, show tqdm progressbar.
  """
  if verbose: print_status('Start get_targets(...).')
  
  output_file = os.path.join(path, 'targets.npy')
  if file_exists(output_file, verbose=verbose): return 
  
  edgelist_probe = load(os.path.join(path, 'edgelist_probe.pkl'))
  
  instances = load(os.path.join(path, 'instances.npy'))  
  
  if verbose: 
    print_status('Create a set containing all edges from the probing interval.')
  edgeset_probing = {
    (u, v) 
    for u, v in edgelist_probe[['source', 'target']].itertuples(index=False)
  }
  
  if verbose:
    print_status('Check if instances connect in probing interval.')
  output = [
    (u, v) in edgeset_probing 
    for u, v in tqdm(
      instances, desc='Determine targets', disable=not verbose, leave=False)
  ]
  
  if verbose: print_status('Store result')
  output = np.array(output)
  np.save(output_file, output)