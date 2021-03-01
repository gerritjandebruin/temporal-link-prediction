import os
import pathlib
import tarfile

import pandas as pd

from .importTools import download
from ..helpers import print_status, file_exists

def act_mooc(
  path: str, 
  url: str = 'https://snap.stanford.edu/data/act-mooc.tar.gz', 
  verbose=False
  ) -> None:
  if verbose: print_status(f'Start {__file__} with {path=}.')

  # Create directory if it does not exist.
  os.makedirs(path, exist_ok=True)

  # Check if output file not already present.
  output_file = os.path.join(path, 'edgelist.pkl')
  if file_exists(output_file, verbose=verbose): return

  # Download file
  edgelist_location = os.path.join(path, 'edgelist.pkl')
  download_location = os.path.join(path, url.split('/')[-1])
  if os.path.isfile(edgelist_location): return
  if not os.path.isfile(download_location): 
    download(url, download_location, verbose=verbose)
  
  # Extract file
  extract_location = os.path.join(path, 'act-mooc')
  if not os.path.isdir(extract_location):
    with tarfile.open(download_location, "r:gz") as tar: tar.extractall(path)
  
  # Convert file to pandas dataframe.
  df = pd.read_csv(
    os.path.join(extract_location, 'mooc_actions.tsv'), 
    delim_whitespace=True)
  df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
  df.rename(
    columns={'USERID': 'source', 'TARGETID': 'target', 'TIMESTAMP': 'datetime'},
    inplace=True)

  # Store result
  df.to_pickle(edgelist_location)