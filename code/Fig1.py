import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
    
dblp_edgelist = pd.read_pickle('data/01/+000/edgelist.pkl')['datetime']
index = pd.to_datetime(np.linspace(dblp_edgelist.min().value, dblp_edgelist.max().value))
index_unix = index.astype(np.int64) / (365*24*3600e9)
index_unix = np.linspace(start=min(index_unix), stop=max(index_unix))

def lin(x):
  return .2 + .8 * (x-min(x)) / (max(x)-min(x))

def exp(x):
  return .2 + 0.8 * (np.exp(3*(x-min(x))/(max(x)-min(x)))-1) / (np.exp(3)-1)

def sqrt(x):
  return .2 + 0.8 * np.sqrt((x-min(x))/(max(x)-min(x)))

time_strategies = {'linear': lin, 'exponential': exp, 'square root': sqrt}

linestyles = ['solid', 'dotted', 'dashed']
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
#   'legend.handlelength': .1,
  'font.family': 'sans-serif',
  'font.sans-serif': 'Helvetica',
  'savefig.transparent': True
}

with plt.rc_context(rc):
    _, ax = plt.subplots()
    for (strategy_str, strategy_func), linestyle in zip(time_strategies.items(), linestyles):
        ax.plot(index, strategy_func(index_unix), label=strategy_str, linestyle=linestyle)
    ax.legend(title='Temporal\nweighting')
    ax.set_xlim((min(index), pd.Timestamp('01-01-2020')))
    ax.set_ylim((0,1))
    ax.set_xlabel('Year')
    ax.set_ylabel('Weight')
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(20))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(matplotlib.dates.YearLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.1))
    plt.tight_layout()
    plt.savefig('figures/1.pdf')