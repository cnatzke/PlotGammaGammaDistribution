import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc


def get_asymmetic_error_index(df, min_rcs_value):
    '''Finds the index for error bounds on the minimum value of reduced chi-square

    Inputs:
        df DataFrame containing minimized values
        min_rcs_value Minimum reduced chi-square value

    Returns:
        i_low Lower error bound
        i_up Upper error bound
    '''

    e_range = 1  # error values found at min chi2/ndf plus minus 1 (approx)
    e_bound = min_rcs_value + e_range

    i_low = df[df['rcs'] < e_bound].mixing_angle_1.idxmin()
    i_up = df[df['rcs'] < e_bound].mixing_angle_1.idxmax()

    return i_low, i_up


def plot_chi_square(find_minimum):
    #print(f'Opening file: {sys.argv[1]}')
    columns = ["mixing_angle_1", "mixing_angle_2", "rcs"]
    df_0 = pd.read_csv(sys.argv[1], header=0,
                       names=columns, skiprows=1, sep='\t')
    '''
    df_0 = pd.read_csv("./chi2_values_4_2_0.dat", header=0,
                     names=columns, skiprows=1, sep='\t')
    df_1 = pd.read_csv("./chi2_values_3_2_0.dat", header=0,
                     names=columns, skiprows=1, sep='\t')
    df_2 = pd.read_csv("./chi2_values_2_2_0.dat", header=0,
                     names=columns, skiprows=1, sep='\t')
    '''
    # print(df.head())

    fig, ax = plt.subplots()

    rc('text', usetex=True)
    label_size = 20
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    title_font = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    axis_fontx = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'top'}
    axis_fonty = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}

    # Set context to `"paper"`
    sns.set_context("paper")
    ax.set(yscale='log')
    sns.lineplot(x="mixing_angle_1", y="rcs", data=df_0, ax=ax)
    '''
    sns.lineplot(x="mixing_angle_1", y="rcs", data=df_1, ax=ax)
    sns.lineplot(x="mixing_angle_1", y="rcs", data=df_2, ax=ax)
    '''

    #ax.legend(['$J_i = 4$', '$J_i = 3$','$J_i = 2$'])
    ax.set_title('$^{207}$Bi $\chi^2$/NDF', **title_font)
    ax.set_xlabel("atan($\delta$)", **axis_fontx)
    ax.set_ylabel('$\chi^2$/(NDF)', **axis_fonty)
    ax.tick_params(direction='inout', labelsize=label_size, length=14)

    if find_minimum:
        # Finding minimum values
        min_index = df_0['rcs'].idxmin()
        min_angle = df_0.iloc[min_index].mixing_angle_1
        min_rcs = df_0.iloc[min_index].rcs

        # Finding error bounds
        i_low, i_up = get_asymmetic_error_index(df_0, min_rcs)
        e_angle_low = df_0.iloc[i_low].mixing_angle_1
        e_angle_up = df_0.iloc[i_up].mixing_angle_1

        min_delta = np.tan(min_angle)
        e_delta_up = abs(min_delta - np.tan(e_angle_up))
        e_delta_low = abs(min_delta - np.tan(e_angle_low))
        #print(f'Minimum values:\n {df_0.iloc[min_index]}')
        # print(df_0.iloc[min_index-5:min_index+5])

        print(f'--------- MINIMUM VALUE ----------')
        print(f'Mixing Angle: {min_angle:.6f} +/- ({(e_angle_up - min_angle):.6f} / {(min_angle - e_angle_low):.6f})')
        print(f'Mixing Ratio: {min_delta:.6f} +/- ({e_delta_up:.6f} / {e_delta_low:.6f})')
        print(f'Reduced Chi-Squared: {df_0.iloc[min_index].rcs:.2f}')
        print(f'----------------------------------')

    plt.tight_layout()
    plt.show()


def main():
    plot_chi_square(True)


if __name__ == "__main__":
    main()
