import os

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import typer

from .logger import logger 

app = typer.Typer()

@app.command()
def single(in_edgelist: str, out_sampled_file: str, cutoff: int = 2):
  if os.path.isfile(out_sampled_file):
    logger.debug(f'{out_sampled_file} already exists')
    return
  
  edgelist = pd.read_pickle(in_edgelist)
  graph_mature = nx.from_pandas_edgelist(
    edgelist.loc[lambda x: x['phase'] == 'mature'])
  
  instances = [
    (node, neighborhood)
    for node 
    in tqdm(graph_mature, mininterval=10)
    for neighborhood, distance 
    in nx.single_source_shortest_path_length(graph_mature, node, 
                                             cutoff=cutoff).items() 
    if distance > 1 and node < neighborhood
  ]
  
  np.save(out_sampled_file, instances)
  
@app.command()
def all():
  folders = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 18, 19, 
             20, 21, 22, 23, 24, 25, 28, 29]
  for folder in folders:
    path = os.path.join('data', f'{folders:02}')
    logger.debug(f'{folder=}')
    single(in_edgelist=os.path.join(path, 'edgelist.pkl'), cutoff=2,
           out_sampled_file=os.path.join(path, 'all_instances.npy'))
  
if __name__ == '__main__':
  app()