import os

import pandas as pd

import tlp

urls = {
  28: 'https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv',
  29: 'https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv'
}

def get_edgelist(path: str, url: str) -> None:
  os.makedirs(path, exist_ok=True)
  edgelist_location = os.path.join(path, 'edgelist.pkl')
  download_location = os.path.join(path, 'soc-redditHyperlinks-body.tsv')
  if os.path.isfile(edgelist_location): return
  if not os.path.isfile(download_location): 
    tlp.download(url, download_location)
  
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
  df.to_pickle(edgelist_location)
  
if __name__ == "__main__":
  for index, url in urls.items():
    get_edgelist(path=f'data/{index}', url=url)