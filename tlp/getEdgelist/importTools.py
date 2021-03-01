import os
import tarfile

import requests
from tqdm.auto import tqdm

def download(url: str, dst: str, verbose: bool = False):
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
    total=file_size, initial=first_byte, unit='B', unit_scale=True, 
    desc=url.split('/')[-1], disable=not verbose)
  req = requests.get(url, headers=header, stream=True)
  with(open(dst, 'ab')) as f:
    for chunk in req.iter_content(chunk_size=1024):
      if chunk:
        f.write(chunk)
        pbar.update(1024)
  pbar.close()

def extract_tar(tar_file: str, output_path: str) -> None:
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
      os.replace(
        entry.path, os.path.join(output_path, entry.name.split('.')[0])
      )
  os.rmdir(os.path.join(output_path, dir_name))