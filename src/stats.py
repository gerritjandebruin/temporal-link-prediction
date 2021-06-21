import multiprocessing, os, random, logging
import pandas as pd
import networkx as nx
import typer
import joblib
import numpy as np
from tqdm.auto import tqdm

from .progress_parallel import ProgressParallel, delayed
from .logger import logger

app = typer.Typer()

def get_network(network_index: int, multi_edge=True, edge_attr='datetime'):
    edgelist = (
        pd.read_pickle(f'data/{network_index:02}/+000/edgelist.pkl')
        .loc[lambda x: x['source'] != x['target']]
    )
    if multi_edge:
        return nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph,
                                       edge_attr=edge_attr)
    else:
        return nx.from_pandas_edgelist(edgelist)

@app.command()
def single(network: int, nswap_perc: int, method: str = None, 
           verbose: bool = True):
    assert nswap_perc != 0 or method is None, f'got {nswap_perc=} and {method=}'
    if nswap_perc == 0:
        directory = f'data/{network:02}/{nswap_perc:+04.0f}'
    else:
        directory = f'data/{network:02}/{nswap_perc:+04.0f}{method}'
    filepath_in = os.path.join(directory, 'graph.pkl')
    files = ['assortativity.float', 'connected_pairs.int', 'edges.int', 
             'nodes.int', 'triangles.int']
    if not os.path.isfile(filepath_in):
        return
    if np.all([os.path.isfile(os.path.join(directory, f)) for f in files]):
        return

    if not verbose:
        logger.setLevel(logging.INFO)
    if nswap_perc == 0:
        logger.debug(f'Get network {network=:02}')
        G = get_network(network, edge_attr=None)
    else:
        try:
            logger.debug(f'Get network {filepath_in=}')
            G = joblib.load(filepath_in)
        except:
            tqdm.write(f'{network=}, {nswap_perc=} failed!')
            return
    out_directory = os.path.join(directory, 'properties')
    os.makedirs(out_directory, exist_ok=True)
    if not os.path.isfile(os.path.join(out_directory, 'nodes.int')):
        logger.debug('Get node count.')
        nodes = G.number_of_nodes()
        with open(os.path.join(out_directory, 'nodes.int'), 'w') as file:
            file.write(str(nodes))
    if not os.path.isfile(os.path.join(out_directory, 'edges.int')):
        logger.debug('Get edge count.')
        edges = G.number_of_edges()
        with open(os.path.join(out_directory, 'edges.int'), 'w') as file:
            file.write(str(edges))
    logger.debug('Get simple graph.')
    simple_graph = nx.Graph(G)
    if not os.path.isfile(os.path.join(out_directory, 'connected_pairs.int')):
        logger.debug('Get connected pairs.')
        connected_pairs = simple_graph.number_of_edges()
        with open(os.path.join(out_directory, 'connected_pairs.int'), 'w') as file:
            file.write(str(connected_pairs))
    if not os.path.isfile(os.path.join(out_directory, 'assortativity.float')):
        logger.debug('Get assortativity.')
        assortativity = nx.degree_assortativity_coefficient(G)
        with open(os.path.join(out_directory, 'assortativity.float'), 'w') as file:
            file.write(str(assortativity))
    if not os.path.isfile(os.path.join(out_directory, 'triangles.int')):
        logger.debug('Get triangle count.')
        triangles = sum(nx.triangles(simple_graph).values()) // 3
        with open(os.path.join(out_directory, 'triangles.int'), 'w') as file:
            file.write(str(triangles))    

        
@app.command()
def all(network: int = None, 
        n_jobs: int = -1, 
        method: str = None,
        shuffle: bool = True, 
        seed: int = 42, 
        verbose: bool = False):
    if not network:
        networks = np.arange(1, 31)
    else:
        networks = [network]
    if method is None:
        methods = ['a', 'b']
    else:
        assert method in ['a', 'b']
        methods = [method]
    iterations = [(network, nswap_perc, method)
                  for network in networks
                  for nswap_perc in np.arange(-100, 101, 20)
                  for method in methods
                  if not nswap_perc == 0 and method == 'b']
    if shuffle:
        random.seed(seed)
        random.shuffle(iterations)
    if multiprocessing.cpu_count() > len(iterations):
        n_jobs = len(iterations)
    ProgressParallel(n_jobs=n_jobs, total=len(iterations))(
        delayed(single)(
            network=network, 
            nswap_perc=nswap_perc,
            method=method,
            verbose=verbose
        ) for network, nswap_perc, method in iterations
    )

if __name__ == '__main__':
    app()   