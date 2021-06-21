import itertools, os, json, logging, random
from networkx.utils import py_random_state
from tqdm.auto import tqdm
import numpy as np
import networkx as nx
import pandas as pd
import joblib

import typer

from .progress_parallel import ProgressParallel, delayed
from .logger import logger

app = typer.Typer()

def get_network(network_index: int, multi_edge=True):
    edgelist = (
        pd.read_pickle(f'data/{network_index:02}/+000/edgelist.pkl')
        .loc[lambda x: x['source'] != x['target']]
    )
    if multi_edge:
        return nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph,
                                       edge_attr='datetime')
    else:
        return nx.from_pandas_edgelist(edgelist)


@py_random_state(3)
def double_edge_swap(network_index, nswap_perc=100, max_tries=1_000_000_000,
                     seed=None, assortative=True, verbose=False):    
    G = get_network(network_index)
    connected_pairs = nx.Graph(G).number_of_edges()
    nswap = int(connected_pairs * nswap_perc / 100)
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence
    with tqdm(total=max_tries, disable=not verbose, desc='tries', position=0) as pbar1, tqdm(total=nswap, disable=not verbose, desc='swaps', position=1) as pbar2:
        while swapcount < nswap and tries < max_tries:
            if tries >= max_tries:
                return None, {'tries': tries, 'swapcount': swapcount}
            
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ui == xi:
                continue  # same source, skip
            u = keys[ui]  # convert index to label
            x = keys[xi]
            # choose target uniformly from neighbors
            v = seed.choice(list(G[u]))
            y = seed.choice(list(G[x]))
            if v == y:
                continue  # Same target, skip
            # Get list of (deg,nodes) tuples, sorted on degree.
            degs = sorted([(G.degree(n), n)
                          for n in (u, v, x, y)], reverse=True)
            if (x not in G[u]) and (y not in G[v]):
                if not ((u == degs[0][1] and v == degs[1][1]) or 
                        (x == degs[0][1] and y == degs[1][1])):
                    datetimes_uv = [edge['datetime'] for edge in G[u][v].values()]
                    datetimes_xy = [edge['datetime'] for edge in G[x][y].values()]
                    for _ in datetimes_uv:
                        G.remove_edge(u, v)
                    for _ in datetimes_xy:
                        G.remove_edge(x, y)
                    if assortative:
                        for datetime_ in datetimes_uv:
                            # Connect two nodes with largest degree
                            G.add_edge(degs[0][1], degs[1][1], datetime=datetime_)
                        for datetime_ in datetimes_xy:
                            # Connect two nodes with smallest degree
                            G.add_edge(degs[2][1], degs[3][1], datetime=datetime_)
                    else:
                        for datetime_ in datetimes_uv:
                            # Connect smallest with largest
                            G.add_edge(degs[0][1], degs[3][1], datetime=datetime_)
                        for datetime_ in datetimes_xy:
                            G.add_edge(degs[1][1], degs[2][1],
                                    datetime=datetime_)  # Connect other
                    swapcount += 1
                    pbar2.update(1)
            tries += 1
            pbar1.update(1)
        else:
            return G, {'tries': tries, 'swapcount': swapcount}


def replace_edges(G, from_nodepair, to_nodepair):
    u,v = from_nodepair
    x,y = to_nodepair
    datetimes_uv = [edge['datetime'] for edge in G[u][v].values()]
    G.remove_edges_from([(u,v)])
    for datetime_ in datetimes_uv:
        G.add_edge(x, y, datetime=datetime_)
    return G

@py_random_state(3)
def double_edge_swap_triangle_preserve(network_index, 
                                       nswap_perc=100, 
                                       max_tries=1_000_000_000,
                                       seed=None, 
                                       assortative=True, 
                                       verbose=False):    
    G = get_network(network_index)
    connected_pairs = nx.Graph(G).number_of_edges()
    nswap = int(connected_pairs * nswap_perc / 100)
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    tries = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence
    with tqdm(total=max_tries, disable=not verbose, desc='tries', 
              position=0) as pbar1:
        with tqdm(total=nswap, disable=not verbose, desc='swaps', 
                  position=1) as pbar2:
            while swapcount < nswap and tries < max_tries:
                if tries >= max_tries:
                    return None, {'tries': tries, 'swapcount': swapcount}
                tries += 1
                pbar1.update(1)
                # pick two random edges without creating edge list
                # choose source node indices from discrete distribution
                ui = discrete_sequence(1, cdistribution=cdf, seed=seed)[0]
                u = keys[ui]  # convert index to label
                if len(G[u]) < 2:
                    continue # Else you can't sample two neighbours
                v, x = seed.sample(list(G[u]), k=2)
                # Make y such that is a neighbor of x, but not of u or v.
                nbs_x = list(G[x])
                nbs_u = list(G[u])
                nbs_v = list(G[v])
                seed.shuffle(nbs_x)
                for nb_x in nbs_x:
                    if nb_x not in nbs_u and nb_x not in nbs_v:
                        y = nb_x
                        break
                else:
                    continue
                deg_diff_u_v = abs(len(G[u]) - len(G[v]))
                deg_diff_v_y = abs(len(G[v]) - len(G[y]))
                deg_diff_x_y = abs(len(G[x]) - len(G[y]))
                if assortative:
                    extreme_deg_diff = max(deg_diff_u_v, deg_diff_v_y, deg_diff_x_y)
                else:
                    extreme_deg_diff = min(deg_diff_u_v, deg_diff_v_y, deg_diff_x_y)
                
                if deg_diff_v_y == extreme_deg_diff: 
                    continue # Change nothing
                elif deg_diff_u_v == extreme_deg_diff:
                    replace_edges(G, (u,v), (v,y))
                else:
                    assert deg_diff_x_y == extreme_deg_diff
                    replace_edges(G, (x,y), (v,y))
                    
                swapcount += 1
                pbar2.update(1)
            else:
                return G, {'tries': tries, 'swapcount': swapcount}


def check_common_neighbor(G, u, v):
    for w in G[u]:
        if w in G[v] and w not in (u, v):
            return True
    else:
        return False

@app.command()
def single(network: int, 
           nswap_perc: int, 
           method: str,
           max_tries: int = 1_000_000_000, 
           verbose: bool = False):
    # Check if result is cached.
    if not verbose:
        logger.setLevel(logging.INFO)
    if nswap_perc == 0:
        filedir = f'data/{network:02}/{nswap_perc:+04.0f}'
    else:
        filedir = f'data/{network:02}/{nswap_perc:+04.0f}{method}'
    filepath_graph = os.path.join(filedir, 'graph.pkl')
    filepath_rewire_stats = os.path.join(filedir, 'rewire_stats.json')
    os.makedirs(filedir, exist_ok=True)
    if os.path.isfile(filepath_graph):
        return
    logger.debug(f'Do {network=}, {nswap_perc=}')
    
    if method == 'a':
        G, stats = double_edge_swap(network, 
                                    nswap_perc=abs(nswap_perc), 
                                    max_tries=max_tries, 
                                    assortative=nswap_perc>0,
                                    verbose=verbose)
    else:
        assert method == 'b', f'{method=} not implemented'
        G, stats = double_edge_swap_triangle_preserve(
            network,
            nswap_perc=abs(nswap_perc),
            assortative=nswap_perc>0,
            verbose=verbose)
    if G is not None:
        logger.debug(f'Store {network=}, {nswap_perc=}')
        joblib.dump(G, filepath_graph)
        logger.debug(f'Done {network=}, {nswap_perc=}')
    else:
        logger.debug(f'No result {network=}, {nswap_perc=}')
    with open(filepath_rewire_stats, 'w') as file:
        json.dump(stats, file)


@app.command()
def all(network: int = None, 
        n_jobs: int = -1, 
        shuffle: bool = True, 
        seed: int = 42, 
        verbose: bool = False):
    if not network:
        networks = np.arange(1, 31)
    else:
        networks = [network]
    iterations = [(network, nswap_perc, method)
                  for network in networks
                  for nswap_perc in np.arange(-100, 101, 20)
                  for method in ['a', 'b']
                  if not nswap_perc == 0 and method == 'b']
    if shuffle:
        random.seed(seed)
        random.shuffle(iterations)
    ProgressParallel(n_jobs=n_jobs, total=len(iterations))(
        delayed(single)(
            network=network, 
            nswap_perc=nswap_perc,
            method=method,
            verbose=verbose
        ) for network, nswap_perc, method in iterations
    )
    
@app.command()
def clean_invalid_pickles():
    iterator = [
        os.path.join(subdirectory.path, 'graph.pkl')
        for directory in os.scandir('data')
        for subdirectory in os.scandir(directory.path)
        if os.path.isfile(os.path.join(subdirectory.path, 'graph.pkl'))
    ]
    
    def remove_if_invalid(filepath):
        try:
            joblib.load(filepath)
        except:
            logger.info(f'Delete {filepath}')
            os.remove(filepath)
    
    ProgressParallel(n_jobs=-1, total=len(iterator))(
        delayed(remove_if_invalid)(filepath) for filepath in iterator
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
        if not os.path.isfile(f'data/{n:02}/{nswap_perc:+04.0f}/graph.pkl'):
            print(n, nswap_perc)

if __name__ == '__main__':
    app()