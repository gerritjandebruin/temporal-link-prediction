import itertools
import os
import typing

import joblib
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from ..helpers import ProgressParallel, file_exists, print_status
from ..constants import CUTOFF

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def get_neighborhood(
  graph: nx.Graph, *, nodes: list, cutoff: typing.Optional[int]
  ):
  return [
    (node, neighborhood)
    for node in nodes if node is not None
    for neighborhood, distance
    in nx.single_source_shortest_path_length(graph, node, cutoff=cutoff).items()
    if distance > 1 and node < neighborhood
  ]

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
  # Mark beginning
  if verbose: 
    print_status('Start get_instances(...). Read in edgelist_mature.pkl')

  # Quit if result already exists.
  output_file = os.path.join(path, 'instances.npy')
  if file_exists(output_file, verbose=verbose): return
  
  # Read in edgelist.
  edgelist_mature_file = os.path.join(path, 'edgelist_mature.pkl')
  assert os.path.isfile(edgelist_mature_file), (
    f'{edgelist_mature_file} does not exist')
  edgelist_mature = joblib.load(edgelist_mature_file)
  
  # Create graph
  if verbose: print_status('Create nx.Graph object.')
  graph_mature=nx.from_pandas_edgelist(edgelist_mature)
  
  # Collect all pairs of nodes at a maximum distance of cutoff.
  if verbose: print_status('Collect instances')
  # Singlecore
  instances = [
    (node, neighborhood)
    for node 
    in tqdm(graph_mature, desc=f"#{path.split('/')[1]} collect instances", mininterval=10)
    for neighborhood, distance 
    in nx.single_source_shortest_path_length(graph_mature, node, 
                                             cutoff=cutoff).items() 
    if distance > 1 and node < neighborhood
  ]
  
  if verbose: print_status('Store result')
  instances = np.array(instances)
  np.save(output_file, instances)
