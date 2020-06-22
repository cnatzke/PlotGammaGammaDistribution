import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc


def plot_chi_square():
    print(f'Opening file: {sys.argv[1]}')
    columns = ["mixing_angle_1", "mixing_angle_2", "rcs"]
    df = pd.read_csv(sys.argv[1], header=0,
                     names=columns, skiprows=1, sep='\t')
    print(df.head())

    fig, ax = plt.subplots()

    rc('text', usetex=True)
    label_size = 20
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    title_font = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    axis_fontx = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'top'}
    axis_fonty = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}

    # Set context to `"paper"`
    sns.set_context("paper")
    ax.set(yscale='log')
    ax.set_title('$^{60}$Co J$_i = 4$', **title_font)

    sns.lineplot(x="mixing_angle_1", y="rcs", data=df, ax=ax)

    ax.set_xlabel("atan($\delta$)", **axis_fontx)
    ax.set_ylabel('$\chi^2$/(NDF)', **axis_fonty)
    ax.tick_params(direction='inout', labelsize=label_size, length=14)

    plt.tight_layout()
    plt.show(fig)

    min_df = df[df['rcs'] == min(df['rcs'])]
    print(f'Minimum values:\n{min_df}')


def main():
    plot_chi_square()


if __name__ == "__main__":
    main()
