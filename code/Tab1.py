import numpy as np
import pandas as pd

networks = [network for network in np.arange(1, 31)
            if network not in [15, 17, 26, 27]]

def read_file(path):
    extension = path.split('.')[1]
    if extension == 'int':
        with open(path) as file:
            return int(file.read())
    elif extension == 'float':
        with open(path) as file:
            return float(file.read())
    else:
        raise Exception(f'{extension=}')

def get_stats(network: int):
    properties_dir = f'data/{network:02}/+000/properties/'
    properties = {prop.split('.')[0]: read_file(properties_dir + prop) 
                  for prop 
                  in ['nodes.int', 'edges.int', 'connected_pairs.int', 'edges.int', 
                      'assortativity.float', 'average_clustering.float', 'diameter.int']}
    info = pd.read_json('networks.jsonl', lines=True).set_index('index').loc[network]
    return {
        'Label': info['label'],
        'Domain': info['category'],
        '\\bar e': properties['edges'] / properties['connected_pairs'],
        'Nodes': properties['nodes'], 
        'Edges': properties['edges'],
        'Density': 2*properties['connected_pairs'] / (properties['nodes']*(properties['nodes'] - 1)),
        'D.a.': properties['assortativity'],
        'A.c.c': properties['average_clustering'],
        'Diameter': properties['diameter'],
        '': '\cite{' + info['source'] + '}' #type: ignore
    }
    
def scientific_notation(x): 
    x = f'{x:.0e}'
    coefficient = x[0]
    if '-' in x:
        exponent = '-' + x[-1]
    else:
        exponent = x[1]
    return f'${coefficient}\!\cdot\!10^{{{exponent}}}$' #type: ignore
    
table = pd.DataFrame([get_stats(network) for network in networks]).sort_values('Nodes')

latex_table = table.to_latex(
    formatters={
        '\\bar e': lambda x: f'{x:.1f}',
        'Nodes': lambda x: f'{x:,}', 
        'Edges': lambda x: f'{x:,}',
        'Density': scientific_notation,
        'D.a.': lambda x: f'{x:.2f}',
        'A.c.c': lambda x: f'{x:.2f}'
    },
    column_format='l@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}r@{\hspace{1em}}r@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{1em}}c@{\hspace{0.5em}}c',
    index=False,
    caption=(
        "Networks used in this work. "
        "The following abbreviations are used in the columns; "
        "d.a.: degree assortativity, acc: average clustering coefficient, diam.: diameter. "
        "In the column `domain', Technological is abbreviated to Tech. and Information to Inf."
    ),
    label='table:datasets',
    escape=False,
    multicolumn=False,
)
print(latex_table)