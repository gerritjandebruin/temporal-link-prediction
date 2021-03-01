import matplotlib.pyplot as plt

from getData import get_data #type: ignore

if __name__ == '__main__':
    df = get_data()
    
    rc = {'xtick.top': True, 'ytick.right': True, 'figure.figsize': (4, 4)}
    with plt.rc_context(rc):
        df.plot.scatter(x='score', y='score time-agnostic', ls='--')
        plt.axline((0, 0), (1,1), c='black') #type: ignore
        plt.xlim(.5, 1)
        plt.ylim(.5, 1)
        plt.savefig('figures/figure_scores.eps')