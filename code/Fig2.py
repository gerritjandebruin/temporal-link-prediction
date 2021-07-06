import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
    
networks = [network for network in np.arange(1, 31) if network not in [15, 17, 26, 27]]
    
def get_stats(network: int):
    properties_dir = f'data/{network:02}/+000/properties/'
    with open(properties_dir + 'nodes.int') as file:
        number_of_nodes = int(file.read())
    with open(properties_dir + 'edges.int') as file:
        number_of_edges = int(file.read())
    return {
        'Number of nodes': number_of_nodes, 
        'Number of edges': number_of_edges,
        'Domain': pd.read_json('networks.jsonl', lines=True).set_index('index').loc[network, 'category']
    }

data = pd.DataFrame(map(get_stats, networks)) #type: ignore

fontsize=8
rc = {
  'xtick.top': True, 'ytick.right': True, 'figure.figsize': (3.30,2.475), 
  'axes.titlesize': fontsize, 
  'axes.labelsize': fontsize, 
  'xtick.labelsize': fontsize, 
  'ytick.labelsize': fontsize, 
  'legend.fontsize': fontsize, 
  'legend.title_fontsize': fontsize,
  'lines.linewidth': 2,
#   'lines.markersize': 4,
  'legend.handlelength': .4,
  'font.family': 'sans-serif',
  'font.sans-serif': 'Helvetica',
  'savefig.transparent': True
}

with plt.rc_context(rc):
    sns.scatterplot(data=data, 
                    x='Number of nodes', 
                    y='Number of edges', 
                    hue='Domain', 
                    style='Domain')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of edges')
    plt.legend(loc='upper left', title='Domain')
    plt.tight_layout()
    plt.savefig('figures/2.pdf')