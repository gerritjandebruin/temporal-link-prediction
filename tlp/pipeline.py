import os
import requests
import tarfile
import tempfile
import typing

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .helpers import download

# Only not connected node pairs within cutoff distance in the graph of the 
# maturing interval are used.
CUTOFF = 2 

# The number of samples taken from both the positive and negative class.
SAMPLE_SIZE = 10000

# Split the edgelist at this quantile into the mature and probe edgelist.
SPLIT_FRACTION = 2/3

def _extract_tar(tar_file: typing.IO, output_path: str) -> None:
  """Download and extract the KONECT dataset. Store temporary files in path. It
  has some special way of doing this: it will ignore the first level directory,
  such that the contents of this directory will be stored at path. 
  
  Args:
    tar_file: File that needs to be extracted.
    path: Location where the contents of the .tar.bz2 will be stored.
  """
  assert hasattr(tar_file, 'name')
  
  with tarfile.open(tar_file.name) as tar:
    # I assume that there is only one directory in the tar-archive.
    dir_names = [member.name for member in tar.getmembers() if member.isdir()]
    assert len(dir_names) == 1
    dir_name = dir_names[0]

    tar.extractall(output_path)
    
  # Move all files from the one directory in the archive to the output_path.
  # On top of that, rename them such that the filename contains only what occurs
  # before the first period. 
  # E.g. ./dblp_coauthor/out.dblp_coauthor -> ./out
  with os.scandir(os.path.join(output_path, dir_name)) as it:
    for entry in it:
      os.replace(entry.path, os.path.join(output_path, entry.name.split('.')[0]))
  os.rmdir(os.path.join(output_path, dir_name))
  
def get_edgelist_from_konect(url: str, *, path: str, verbose: bool = False
                             ) -> None:
  """Download and extract the KONECT dataset. Store extracted files in path. If
  the temporary files are already present in path, the file is not again
  downloaded or extracted. The final edgelist, which is an pd.DataFrame with 
  columns 'source', 'target', 'datetime' is stored in output_path/edgelist.pkl.
  
  Args:
    url: The url pointing to KONECT download file. Usual format: 
      'http://konect.cc/files/download.*.tar.bz2'.
    output_path: Optional; Store the extracted dataset in this directory.
    verbose: Optional; Show tqdm when downloading.
  """
  os.makedirs(path, exist_ok=True)
  output_file = os.path.join(path, 'edgelist.pkl')
  
  # Check if output file not already present.
  assert not os.path.isfile(output_file), "file already exists"
  
  # Edgelist is stored in the out.* file contained in the tar archive.
  out_location = os.path.join(path, 'out') 
  if not os.path.isfile(out_location): # Check if extraction took already place.
    with tempfile.NamedTemporaryFile() as download_location:
      download(url, dst=download_location.name, verbose=verbose)
      _extract_tar(tar_file=download_location, output_path=path) 
  
  # CSV file to pd.DataFrame
  edgelist = pd.read_csv(
    os.path.join(out_location, 'out'), delim_whitespace=True, engine='python', 
    comment='%', names=['source', 'target', 'weight', 'datetime'])
  edgelist = edgelist[edgelist['datetime'] != 0]
  
  # Check for signed network
  if -1 in edgelist['weight'].unique():
    print("""\
This is likely a signed network (weight equals -1). 
Only positive weights will be used.
          """)
    edgelist = edgelist[edgelist['weight'] > 0]
  
  # Check of both u->v and v->u are present for every edge.
  edgeset = {
    (u,v) for u, v in edgelist[['source', 'target']].itertuples(index=False)}
  assert np.all(
    [edge in edgeset 
     for edge in edgelist[['source', 'target']].itertuples(index=False)])
  
  # Convert UNIX datetime to datetime object.
  edgelist['datetime'] = pd.to_datetime(edgelist['datetime'], unit='s')

  # Check for weights
  if not (edgelist['weight'] == 1).all():
    print('This is a weighted network. However, weights will be discarded.')
  
  # Drop weight column
  edgelist.drop(columns=['weight'], inplace=True)
  
  # Store
  edgelist.to_pickle(output_file)     
    
def split_in_intervals(
  path: str, *,
  t_min: typing.Optional[pd.Timestamp] = None,
  t_split: typing.Optional[pd.Timestamp] = None,
  t_max: typing.Optional[pd.Timestamp] = None
  ) -> None:
  """Split the edgelist into the edges belonging to the maturing and probing 
  interval. The edgelist should be present as a pickled pd.DataFrame at 
  path/edgelist.pkl. The results (two pd.DataFrames) are stored at 
  path/edgelist_{mature/probe}.pkl.
  
  Args:
    path
    t_min, t_split, t_max: Optional; Timestamps used to mark the beginning of 
      the maturing interval, the end of the maturing interval and the end of the
      probing interval, respectively.
  """
  edgelist_file = os.path.join(path, 'edgelist.pkl')
  assert os.path.isfile(edgelist_file), f'{edgelist_file} does not exists'
  edgelist = joblib.load(edgelist_file)
  
  edgelist_mature_file = os.path.join(path, 'edgelist_mature.pkl')
  edgelist_probe_file = os.path.join(path, 'edgelist_probe.pkl')
  assert not os.path.isfile(edgelist_mature_file), (
    f'{edgelist_mature_file} already exists')
  assert not os.path.isfile(edgelist_probe_file), (
    f'{edgelist_probe_file} already exists')
  
  t_min_, t_split_, t_max_ = (
    edgelist['datetime'].quantile([0,SPLIT_FRACTION,1]))
  if t_min is None: t_min = t_min_
  if t_split is None: t_split = t_split_
  if t_max is None: t_max = t_max_
  assert isinstance(t_min, pd.Timestamp)
  assert isinstance(t_split, pd.Timestamp)
  assert isinstance(t_max, pd.Timestamp)
  
  edgelist_mature = edgelist.loc[edgelist['datetime'].between(t_min, t_split)]
  edgelist_probe = edgelist.loc[edgelist['datetime'].between(t_split, t_max)]
  
  edgelist_mature.to_pickle(edgelist_mature_file)
  edgelist_probe.to_pickle(edgelist_probe_file)

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
    cutoff: Optional; Return only unconnected pairs of nodes with at most this 
      distance in the graph.
    verbose: Optional; If true, show tqdm progressbar.
  """ 
  output_file = os.path.join(path, 'instances.npy')
  assert not os.path.isfile(output_file), f'{output_file} already exists'
  
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
  output_file = os.path.join(path, 'targets.npy')
  assert not os.path.isfile(output_file), f'{output_file} already exists'  
  
  edgelist_probe_file = os.path.join(path, 'edgelist_probe.pkl')
  assert os.path.isfile(edgelist_probe_file), (
    f'{edgelist_probe_file} does not exist')
  edgelist_probe = joblib.load(edgelist_probe_file)  
  
  instances_file = os.path.join(path, 'instances.pkl')
  assert os.path.isfile(instances_file), f'{instances_file} does not exist'
  instances = np.load(instances_file)    
  
  edgeset_probing = {
    (u, v) 
    for u, v in edgelist_probe[['source', 'target']].itertuples(index=False)
  }
  
  output = [
    (u, v) in edgeset_probing 
    for u, v in tqdm(
      instances, desc='Determine targets', disable=not verbose, leave=False)
  ]
  output = np.array(output)
  np.save(output_file, output)

def _sample(array: np.ndarray, size: int) -> np.ndarray:
  """Take a sample (with replacement) of a given size n along the first axis of 
  array.
  """
  return array[np.random.randint(len(array), size=size)]

def balanced_sample(path: str, *, size: int = SAMPLE_SIZE) -> None:
  """Take n positive and n negative samples from the provided instances. Whether
  a instance is positive, is determined by the targets.
  The instances should be a np.ndarray of shape (n,2) provided at 
  path/instances.npy and the targets a np.ndarray of shape (n) provided at 
  path/targets.npy. The result are two files; path/instances_sampled.npy is a 
  np.ndarray with shape (size, 2) and path/targets_sampled.npy is a np.ndarray
  with shape (size).
  
  Args:
    path: str
    size: Optional; Take this number of positive and this number of negative 
      samples.
  """
  instances_sampled_file = os.path.join(path, 'instances_sampled.npy')
  assert not os.path.isfile(instances_sampled_file), (
    f'{instances_sampled_file} already exists')
  
  targets_sampled_file = os.path.join(path, 'targets_sampled.npy')
  assert not os.path.isfile(targets_sampled_file), (
    f'{targets_sampled_file} already exists')  
  
  instances_file = os.path.join(path, 'instances.pkl')
  assert os.path.isfile(instances_file), f'{instances_file} does not exist'
  instances = np.load(instances_file)   
  
  targets_file = os.path.join(path, 'targets.pkl')
  assert os.path.isfile(targets_file), f'{targets_file} does not exist'
  targets = np.load(targets_file)       
  
  positives = _sample(instances[targets], size)
  negatives = _sample(instances[~targets], size)
  
  instances_sampled = np.concatenate([negatives, positives])
  np.save(instances_sampled_file, instances_sampled)
  
  targets_sampled = np.concatenate(
    [np.zeros(size, dtype=bool), np.ones(size, dtype=bool)])
  np.save(targets_sampled_file, targets_sampled)