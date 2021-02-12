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

# Only not connected node pairs within cutoff distance in the graph of the 
# maturing interval are used.
CUTOFF = 2 

# The number of samples taken from both the positive and negative class.
SAMPLE_SIZE = 10000

# Split the edgelist at this quantile into the mature and probe edgelist.
SPLIT_FRACTION = 2/3

def download_from_url(url: str, dst: str, verbose: bool = False):
    """
    @param: url to download file
    @param: dst place to put the file
    @param: if verbose, show tqdm
    
    Source: https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1],
        disable=not verbose)
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

def _download_and_extract(url: str, path: str, verbose: bool = False) -> None:
  """Download and extract the KONECT dataset. Store temporary files in path.
  
  Args:
    url: The url pointing to KONECT download file. Usual format: 
      'http://konect.cc/files/download.*.tar.bz2'.
    path: Optional; Store the extracted dataset in this directory.
  """
  os.makedirs(path, exist_ok=True)
  with tempfile.NamedTemporaryFile() as file:
    download_from_url(url, file.name, verbose=verbose)
    with tarfile.open(file.name) as tar:
      dir_names = [member.name for member in tar.getmembers() if member.isdir()]
      assert len(dir_names) == 1
      dir_name = dir_names[0]
      
      tar.extractall(path)
  with os.scandir(os.path.join(path, dir_name)) as it:
    for entry in it:
      os.replace(entry.path, os.path.join(path, entry.name.split('.')[0]))
  os.rmdir(os.path.join(path, dir_name))
  
def get_edgelist(
  url: str, path: str, verbose: bool = False) -> pd.DataFrame:
  """Download and extract the KONECT dataset. Store temporary files in path. If
  the temporary files are already present in path, the file is not again
  downloaded or extracted.
  
  Args:
    url: The url pointing to KONECT download file. Usual format: 
      'http://konect.cc/files/download.*.tar.bz2'.
    path: Optional; Store the extracted dataset in this directory.
    verbose: Optional; Show tqdm when downloading.
    
  Returns:
    edgelist: A pd.DataFrame containing the columns source, target and datetime.
  """
  if not os.path.isfile(os.path.join(path, 'out')):
    _download_and_extract(url, path, verbose=verbose) 
  
  edgelist = pd.read_csv(os.path.join(path, 'out'), delim_whitespace=True,
                         engine='python', comment='%', 
                         names=['source', 'target', 'weight', 'datetime'])
  edgelist = edgelist[edgelist['datetime'] != 0]
  
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
  
  edgelist['datetime'] = pd.to_datetime(edgelist['datetime'], unit='s')

  if not (edgelist['weight'] == 1).all():
    print('This is a weighted network. However, weights will be discarded.')
  return edgelist.drop(columns=['weight'])     
    
def split_in_intervals(
  edgelist: pd.DataFrame, *, 
  t_min: typing.Optional[pd.Timestamp] = None,
  t_split: typing.Optional[pd.Timestamp] = None,
  t_max: typing.Optional[pd.Timestamp] = None
  ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
  """Split the provided edgelist into the edges belonging to the maturing and
  probing interval.
  
  Args:
    edgelist: DataFrame containing the edges.
    t_min, t_split, t_max: Optional; Timestamps used to mark the beginning of 
      the maturing interval, the end of the maturing interval and the end of the
      probing interval, respectively.
  
  Usage:
    edgelist_maturing, edgelist_probing = split_intervals(edgelist)
  """
  t_min_, t_split_, t_max_ = (
    edgelist['datetime'].quantile([0,SPLIT_FRACTION,1]))
  if t_min is None: t_min = t_min_
  if t_split is None: t_split = t_split_
  if t_max is None: t_max = t_max_
  assert isinstance(t_min, pd.Timestamp)
  assert isinstance(t_split, pd.Timestamp)
  assert isinstance(t_max, pd.Timestamp)
  return (edgelist.loc[edgelist['datetime'].between(t_min, t_split)], 
          edgelist.loc[edgelist['datetime'].between(t_split, t_max)])

def get_instances(edgelist_mature: pd.DataFrame,*,
                  cutoff: typing.Optional[int] = CUTOFF,
                  verbose: bool = False
                  ) -> np.ndarray:
  """Get the instances, which are the pairs of nodes that are not (yet) 
  connected in the maturing interval. 
  
  Args:
    edgelist
    cutoff: Optional; Return only unconnected pairs of nodes with at most this 
      distance in the graph.
    verbose: Optional; If true, show tqdm progressbar.
    
  Returns:
    instances: np.array with size (n,2)
  """ 
  graph_mature=nx.from_pandas_edgelist(edgelist_mature, edge_attr=True, 
                                       create_using=nx.MultiGraph)
  return np.array([
    (node, neighborhood)
    for node 
    in tqdm(graph_mature, disable=not verbose, 
            desc='Collecting unconnected pairs of nodes', leave=False)
    for neighborhood, distance 
    in nx.single_source_shortest_path_length(graph_mature, node, 
                                             cutoff=cutoff).items() 
    if distance > 1 and node < neighborhood])  
  
def get_targets(
  *,instances: np.ndarray, edgelist_probe: pd.DataFrame, verbose: bool = False
  ) -> np.ndarray:
  """Get the targets for the provided instances. These targets indicate whether
  the given pairs of nodes (instances) connect in the graph after the probing 
  interval.
  
  Args:
    instances: np.array with shape (n,2) where n indicate the number of 
      instances.
    edgelist_probe: pd.DataFrame containing the edges belonging to the probing
      interval. In this dataframe at least the columns source and target should 
      be present.
    verbose: Optional; If True, show tqdm progressbar.
    
  Returns:
    targets: np.array with shape (n)
  """
  edgeset_probing = {
    (u, v) 
    for u, v in edgelist_probe[['source', 'target']].itertuples(index=False)}
  return np.array(
    [(u, v) in edgeset_probing 
     for u, v in tqdm(instances, desc='Determine targets', disable=not verbose, 
                      leave=False)])

def _sample(array: np.ndarray, size: int) -> np.ndarray:
  """Take a sample (with replacement) of a given size n along the first axis of 
  array.
  """
  return array[np.random.randint(len(array), size=size)]

def balanced_sample(
  instances: np.ndarray, targets: np.ndarray, *, size: int = SAMPLE_SIZE
                    ) -> tuple[np.ndarray, np.ndarray]:
  """Take n positive and n negative samples from the index.
  
  Args:
    instances: np.array with shape (m,2). From this array the samples are taken.
    targets: np.array with shape (m).
    size: Optional; Take this number of positive and this number of negative 
      samples.
  
  Returns:
    instances: np.array with size (n,2)
    size: np.array with size n
    
  Usage:
    instances_sampled, targets_sampled = tlp.balanced_sample(instances, targets)
  """
  positives = _sample(instances[targets], size)
  negatives = _sample(instances[~targets], size)
  return (
    np.concatenate([negatives, positives]),
    np.concatenate([np.zeros(size, dtype=bool), np.ones(size, dtype=bool)]))