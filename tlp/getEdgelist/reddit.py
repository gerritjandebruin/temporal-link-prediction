import os

import pandas as pd

from .importTools import download
from ..helpers import file_exists, print_status

def reddit(path: str, url: str, verbose: bool = False) -> None:
  if verbose: print_status(f'Start {__file__} with {path=}.')
  
  # Create directory if it does not exist.
  os.makedirs(path, exist_ok=True)

  # Check if output file not already present.
  output_file = os.path.join(path, 'edgelist.pkl')
  if file_exists(output_file, verbose=verbose): return

  # If not yet downloaded, download.
  download_location = os.path.join(path, 'soc-redditHyperlinks-body.tsv')
  if not os.path.isfile(download_location): 
    download(url=url, dst=download_location, verbose=verbose)
  
  # Convert file to pandas dataframe.
  if verbose: print_status('Convert to pd.DataFrame')
  df = pd.read_csv(
    download_location, sep='\t', index_col=False, parse_dates=['TIMESTAMP'])
  df = df[df['LINK_SENTIMENT'] == 1]
  df.rename(
    columns={
      'SOURCE_SUBREDDIT': 'source', 
      'TARGET_SUBREDDIT': 'target', 
      'TIMESTAMP': 'datetime'
    }, 
    inplace=True)
  df = df[['source', 'target', 'datetime']]

  # Store result
  if verbose: print_status('Store result.')
  df.to_pickle(output_file)