import importlib
import os
import urllib.request
import zipfile

import joblib
import pandas as pd

import tlp

url = 'https://lfs.aminer.cn/lab-datasets/dynamicinf/coauthor.zip'
path = 'data/07'
filepath = os.path.join(path, 'edgelist.pkl')

download_location = os.path.join(path, 'coauthor.zip')
extract_location = os.path.join(path, 'coauthor')

if __name__ == "__main__":
  if not os.path.isfile(filepath):
    os.makedirs(path, exist_ok=True)
    
    if not os.path.isfile(download_location):
      urllib.request.urlretrieve(url, download_location)
    
    if not os.path.isdir(extract_location):
      with zipfile.ZipFile(download_location, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    with open(os.path.join(extract_location, 'filelist.txt'), 'r') as f:
      filelist = f.read().splitlines()

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
    
    edgelist = edgelist.reset_index(level='datetime').reset_index(drop=True)
    edgelist.to_pickle(filepath)