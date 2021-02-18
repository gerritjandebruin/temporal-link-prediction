import os

import pandas as pd

import tlp

url = 'https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz'
path = 'data/30/'

def get_edgelist(path: str, url: str) -> None:
  os.makedirs(path, exist_ok=True)
  edgelist_location = os.path.join(path, 'edgelist.pkl')
  download_location = os.path.join(path, url.split('/')[-1])
  if os.path.isfile(edgelist_location): return
  if not os.path.isfile(download_location): 
    tlp.download(url, download_location, verbose=True)
  
  df = pd.read_csv(
    download_location, delim_whitespace=True, index_col=False, 
    names=['source', 'target', 'datetime'])
  df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
  df.to_pickle(edgelist_location)
  
if __name__ == "__main__":
    get_edgelist(path, url)