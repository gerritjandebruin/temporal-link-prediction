import os
import requests
import tarfile
import tempfile
import typing

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

def _download(url: str, dst: str, verbose: bool = False):
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
  assert not os.path.isfile(output_file), f"{output_file} already exists"
  
  # Edgelist is stored in the out.* file contained in the tar archive.
  out_location = os.path.join(path, 'out') 
  if not os.path.isfile(out_location): # Check if extraction took already place.
    with tempfile.NamedTemporaryFile() as download_location:
      _download(url, dst=download_location.name, verbose=verbose)
      _extract_tar(tar_file=download_location, output_path=path) 
  
  # CSV file to pd.DataFrame
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