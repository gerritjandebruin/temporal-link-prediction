import os
import zipfile

import pandas as pd

from .importTools import download
from ..helpers import file_exists, print_status

def aminer(
  path: str, 
  url: str = 'https://lfs.aminer.cn/lab-datasets/dynamicinf/coauthor.zip',
  verbose: bool = False
  ):
  if verbose: print_status(f'Start {__file__} with {path=}.')

  # Create directory if it does not exist.
  os.makedirs(path, exist_ok=True)

  # Check if output file not already present.
  output_file = os.path.join(path, 'edgelist.pkl')
  if file_exists(output_file, verbose=verbose): return
    
  # If not yet downloaded, download.
  download_location = os.path.join(path, 'coauthor.zip')
  if not os.path.isfile(download_location):
    download(url, download_location, verbose=verbose)
    
  # If not yet extracted, extract.
  if verbose: print_status('Extract')
  extract_location = os.path.join(path, 'coauthor')
  if not os.path.isdir(extract_location):
    with zipfile.ZipFile(download_location, 'r') as zip_ref:
      zip_ref.extractall(path)
    
  # Read in files. 
  if verbose: print_status('Read in')
  with open(os.path.join(extract_location, 'filelist.txt'), 'r') as f:
    filelist = f.read().splitlines()

  # Convert file to pandas dataframe.
  if verbose: print_status('Convert to pd.DataFrame')
  edgelist = pd.concat(
    {
      pd.Timestamp(int(file.split('.')[0]), 1, 1): (
        pd.read_csv(
          os.path.join(
            extract_location, file), sep='\t', names=['source', 'target']))
      for file in filelist
    },
    names=['datetime', 'index']
  )
  edgelist.reset_index(level='datetime', inplace=True)
  edgelist.reset_index(drop=True, inplace=True)

  # Store result
  if verbose: print_status('Store result.')
  edgelist.to_pickle(output_file)