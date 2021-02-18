import os
import typing

import joblib
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from .core import file_exists

# Only not connected node pairs within cutoff distance in the graph of the 
# maturing interval are used.
CUTOFF = 2 

def get_instances(
  path: str, *, 
  cutoff: typing.Optional[int] = CUTOFF,
  verbose: bool = False
  ) -> None:
  """Get the instances, which are the pairs of nodes that are not (yet) 
  connected in the maturing interval. The edges in the maturing interval should
  be present at path/edgelist_mature.pkl. 
  
  The result is a np.ndarray with shape (n,2) stored at path/instances.npy.
  
  Args:
    path
    cutoff: 
    verbose: Optional; If true, show tqdm progressbar.
  """ 
  output_file = os.path.join(path, 'instances.npy')
  if file_exists(output_file, verbose=verbose): return
  
  edgelist_mature_file = os.path.join(path, 'edgelist_mature.pkl')
  assert os.path.isfile(edgelist_mature_file), (
    f'{edgelist_mature_file} does not exist')
  edgelist_mature = joblib.load(edgelist_mature_file)
  
  graph_mature=nx.from_pandas_edgelist(
    edgelist_mature, edge_attr=True, create_using=nx.MultiGraph)
  
  instances = [
    (node, neighborhood)
    for node 
    in tqdm(graph_mature, disable=not verbose, 
            desc='Collecting unconnected pairs of nodes', leave=False)
    for neighborhood, distance 
    in nx.single_source_shortest_path_length(graph_mature, node, 
                                            cutoff=cutoff).items() 
    if distance > 1 and node < neighborhood
  ]
  
  instances = np.array(instances)
  np.save(output_file, instances)