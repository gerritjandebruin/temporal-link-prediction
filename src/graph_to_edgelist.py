import itertools, os, random
import joblib
import networkx as nx
import numpy as np
import typer

from .progress_parallel import ProgressParallel, delayed
from .get_edgelist import add_phase
from .logger import logger

app = typer.Typer()

@app.command()
def single(network_index: int, nswap_perc: int, method: str = None, 
           verbose: bool = False):
  directory = f'data/{network_index:02}/{nswap_perc:+04.0f}{"" if method is None else method}'
  filepath_in = os.path.join(directory, 'graph.pkl')
  filepath_out = os.path.join(directory, 'edgelist.pkl')
  if os.path.isfile(filepath_in) and not os.path.isfile(filepath_out):
    if verbose: logger.debug(filepath_in)
    try:
      G = joblib.load(filepath_in)
    except:
      logger.error(network_index, nswap_perc, 'failed!')
      return
    edgelist = add_phase(nx.to_pandas_edgelist(G))
    edgelist.to_pickle(filepath_out)

@app.command()
def all(n_jobs: int = -1, 
        shuffle: bool = True, 
        seed: int = 42,
        verbose: bool = False):
    iterator = [(network, nswap_perc, None if nswap_perc == 0 else method)
                for network in np.arange(1, 31) if not network in [15, 17, 26, 27]
                for nswap_perc in np.arange(-100, 101, 20)
                for method in ['a', 'b'] 
                if not (method == 'b' and nswap_perc == 0)]
    if shuffle:
        random.seed(seed)
        random.shuffle(iterator)
    ProgressParallel(n_jobs=n_jobs, total=len(iterator))(
        delayed(single)(network_index, nswap_perc, method, verbose) 
        for network_index, nswap_perc, method in iterator
    )
  
@app.command()
def check():
    iterator = list(
        itertools.product(
            [network for network in np.arange(1, 31) 
             if network not in [15, 17, 26, 27]], 
            np.arange(-100, 101, 20)
        )
    )
    for n, nswap_perc in iterator:
        if not os.path.isfile(f'data/{n:02}/{nswap_perc:+04.0f}/edgelist.pkl'):
            print(n, nswap_perc)


if __name__ == '__main__':
  app()