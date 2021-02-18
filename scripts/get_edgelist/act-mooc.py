import os
import tarfile

import pandas as pd

import tlp

url = 'https://snap.stanford.edu/data/act-mooc.tar.gz'
path = 'data/31/'

def get_edgelist(path: str, url: str) -> None:
  os.makedirs(path, exist_ok=True)
  edgelist_location = os.path.join(path, 'edgelist.pkl')
  download_location = os.path.join(path, url.split('/')[-1])
  if os.path.isfile(edgelist_location): return
  if not os.path.isfile(download_location): 
    tlp.download(url, download_location, verbose=True)
  
  extract_location = os.path.join(path, 'act-mooc')
  if not os.path.isdir(extract_location):
    with tarfile.open(download_location, "r:gz") as tar: tar.extractall(path)
  
  df = pd.read_csv(
    os.path.join(extract_location, 'mooc_actions.tsv'), 
    delim_whitespace=True)
  df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
  df.rename(
    columns={'USERID': 'source', 'TARGETID': 'target', 'TIMESTAMP': 'datetime'},
    inplace=True)
  df.to_pickle(edgelist_location)
  
if __name__ == "__main__":
  get_edgelist(path, url)