import matplotlib.pyplot as plt
import seaborn as sns

from getData import get_data # type: ignore

if __name__ == "__main__":
    df = get_data()
    sns.heatmap(
        df.corr(), square=True, vmin=-1, vmax=1, annot=True, fmt='.1f', 
        cmap='RdBu'
    )
    # Alleen onderste twee rijen
    plt.savefig('figures/figure_heatmap.eps')
