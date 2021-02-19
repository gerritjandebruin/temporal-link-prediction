import os
import tarfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..helpers import download, print_status, file_exists

def _extract_tar(tar_file: str, output_path: str) -> None:
  """Download and extract the KONECT dataset. Store temporary files in path. It
  has some special way of doing this: it will ignore the first level directory,
  such that the contents of this directory will be stored at path. 
  
  Args:
    tar_file: File that needs to be extracted.
    path: Location where the contents of the .tar.bz2 will be stored.
  """  
  with tarfile.open(tar_file) as tar:
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
  if verbose: print_status('Started getting edgelist from konect.')
  
  os.makedirs(path, exist_ok=True)
  output_file = os.path.join(path, 'edgelist.pkl')
  
  # Check if output file not already present.
  if file_exists(output_file, verbose=verbose): return
  
  # Edgelist is stored in the out.* file contained in the tar archive.
  out_location = os.path.join(path, 'out') 
  if not os.path.isfile(out_location): # Check if extraction took already place.
    download_location = os.path.join(path, 'download')
    if verbose: print_status('Start download')
    download(url, dst=download_location, verbose=verbose)
    
    if verbose: print_status('Start extracting')
    _extract_tar(tar_file=download_location, output_path=path) 
  
  # CSV file to pd.DataFrame
  if verbose: print_status('Start reading csv.')
  edgelist = pd.read_csv(
    out_location, delim_whitespace=True, engine='python', comment='%', 
    names=['source', 'target', 'weight', 'datetime'])
  edgelist = edgelist[edgelist['datetime'] != 0]
  
  # Check for signed network
  if -1 in edgelist['weight'].unique():
    print("""\
This is likely a signed network (weight equals -1). 
Only positive weights will be used.
          """)
    edgelist = edgelist[edgelist['weight'] > 0]
  
  # Check of both u->v and v->u are present for every edge.
  if verbose: print_status('Check for directionality.')
  edgeset = {
    (u,v) for u, v in edgelist[['source', 'target']].itertuples(index=False)}
  assert np.all(
    [edge in edgeset 
     for edge in edgelist[['source', 'target']].itertuples(index=False)])
  
  # Convert UNIX datetime to datetime object.
  if verbose: print_status('Convert datetime column.')
  edgelist['datetime'] = pd.to_datetime(edgelist['datetime'], unit='s')

  # Check for weights
  if verbose: print_status('Check for weights.')
  if not (edgelist['weight'] == 1).all():
    print('This is a weighted network. However, weights will be discarded.')
  
  # Drop weight column
  edgelist.drop(columns=['weight'], inplace=True)
  
  # Store
  if verbose: print_status('Store edgelist')
  edgelist.to_pickle(output_file) 
  if verbose: print_status('Done') 