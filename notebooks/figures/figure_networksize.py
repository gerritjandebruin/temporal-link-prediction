import matplotlib.pyplot as plt
import seaborn as sns

from getData import get_data # type: ignore

if __name__ == "__main__":
    df = get_data()

    with plt.rc_context(
        {'xtick.top': True, 'ytick.right': True, 'figure.figsize': (4, 4)}
        ):
        sns.scatterplot(
            data=df, 
            x='nodes', 
            y='edges', 
            hue='categories', 
            style='categories'
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('# nodes')
        plt.ylabel('# edges')
        plt.savefig('figures/figure_networksize.eps')
