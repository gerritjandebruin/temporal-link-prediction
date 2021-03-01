import os

import pandas as pd

from .importTools import download
from ..helpers import file_exists, print_status

url = 'https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz'
path = 'data/30/'

def email_EU(
  path: str, 
  url: str = 'https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz',
  verbose: bool = False
  ) -> None:
  if verbose: print_status(f'Start {__file__} with {path=}.')

  # Create directory if it does not exist.
  os.makedirs(path, exist_ok=True)

  # Check if output file not already present.
  edgelist_location = os.path.join(path, 'edgelist.pkl')
  if file_exists(edgelist_location, verbose=verbose): return

  # If not yet downloaded, download.
  download_location = os.path.join(path, url.split('/')[-1])
  if not os.path.isfile(download_location): 
    download(url=url, dst=download_location, verbose=verbose)
  
  # Convert file to pandas dataframe.
  if verbose: print_status('Convert to pd.DataFrame')
  df = pd.read_csv(
    download_location, 
    delim_whitespace=True, 
    index_col=False, 
    names=['source', 'target', 'datetime'])
  df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

  # Store result
  if verbose: print_status('Store result.')
  df.to_pickle(edgelist_location)