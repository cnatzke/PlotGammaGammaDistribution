#!/home/cnatzke/anaconda3/envs/py3/bin/python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from lmfit import Model, Parameters

import physics_functions as physics
import chi_square as cs


def plot_distribution():
    verbose = 0

    print(f'Opening file: {sys.argv[1]}')
    df = pd.read_csv(sys.argv[1], header=0, names=[
        'Index', 'Angle', 'Corr_Area', 'Corr_Area_Err', 'Uncorr_Area', 'Uncorr_Area_Err'], skiprows=1)
    if verbose > 0:
        print(df.head())

    df["Cosine_Angle"] = np.cos(np.radians(df['Angle']))
    df["Normalized_Area"] = df['Corr_Area'] / df['Uncorr_Area']
    df["Normalized_Area_Err"] = df['Normalized_Area'] * \
        np.sqrt((df['Corr_Area_Err'] / df['Corr_Area']) ** 2 +
                (df['Uncorr_Area_Err'] / df['Uncorr_Area']) ** 2)
    if verbose > 0:
        print(f'Normalized counts: \n {df}')
    df.to_csv('./event_mixed_counts.dat')

    # plt.show()

    # ------------------------------------------------------------
    # CURVE FITTING
    # ------------------------------------------------------------

    gmodel = Model(physics.gamma_gamma_dist_func)
    params = Parameters()
    params.add('a_0', value=1.0)
    params.add('a_2', value=0.5)
    params.add('a_4', value=0.5)
    # params.add('a_4', value = 0.5, vary=False) #fixed parameter

    result = gmodel.fit(df['Normalized_Area'], params, x=df['Cosine_Angle'])
    print('---------------------------------')
    print('FIT RESULTS')
    print('---------------------------------')
    print(result.fit_report())
    print('---------------------------------')

    # calculating residuals (val - fit value)
    df['Fit_Values'] = result.eval(x=df['Cosine_Angle'])
    df['Residuals'] = df['Normalized_Area'] - df['Fit_Values']
    chi2 = cs.get_chi_square(df['Normalized_Area'],
                             df['Normalized_Area_Err'], df['Fit_Values'])
    chi2_ndf = cs.get_reduced_chi_square(
        df['Normalized_Area'], df['Normalized_Area_Err'], df['Fit_Values'])

    a2_fitted = result.params['a_2'].value
    a2_err_fitted = result.params['a_2'].stderr
    a4_fitted = result.params['a_4'].value
    a4_err_fitted = result.params['a_4'].stderr
    if verbose > 0:
        print(f'Residuals: \n {df.head()}')

    # ------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------
    label_size = 20
    gamma_1 = 569.7
    gamma_2 = 1770.2

    rc('text', usetex=True)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    title_font = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    axis_fontx = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'top'}
    axis_fonty = {'size': label_size, 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    text_font = {'size': label_size, 'color': 'black',
                 'horizontalalignment': 'center', 'verticalalignment': 'center'}
    error_plot_format = {'fmt': '.k', 'ecolor': 'gray', 'lw': 1, 'capsize': 3}

    fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={
                             'height_ratios': [2, 1]}, figsize=(10, 10))
    plt.subplots_adjust(wspace=0, hspace=0)

    axes[0].errorbar(df['Cosine_Angle'], df['Normalized_Area'],
                     yerr=df['Normalized_Area_Err'],  **error_plot_format)
    axes[0].plot(df['Cosine_Angle'], df['Fit_Values'], 'k--')
    axes[0].set_title(
        f'Angular Distribution of {gamma_1} keV and {gamma_2} keV', **title_font)
    axes[0].set_ylabel("Normalized Counts", **axis_fonty)

    # text
    a2_params = f'$a_2$ = {a2_fitted:.6f}$\pm${a2_err_fitted:.6f}'
    a4_params = f'$a_4$ = {a4_fitted:.6f}$\pm${a4_err_fitted:.6f}'
    chi2_params = f'$\chi^2/NDF$ = {chi2_ndf:.2f}'

    axes[0].text(0.5, 0.9, a2_params, transform=axes[0].transAxes, **text_font)
    axes[0].text(0.5, 0.8, a4_params, transform=axes[0].transAxes, **text_font)
    axes[0].text(0.5, 0.7, chi2_params,
                 transform=axes[0].transAxes, **text_font)

    # residual plot
    axes[1].errorbar(df['Cosine_Angle'], df['Residuals'],
                     yerr=df['Normalized_Area_Err'], **error_plot_format)
    axes[1].axhline(y=0.0, color='black', linestyle='--')
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].set_xlabel(r'Cos($\theta$)', **axis_fontx)
    axes[1].set_ylabel("Residuals", **axis_fonty)

    for ax in axes.flatten():
        ax.tick_params(direction='inout', labelsize=label_size, length=14)

    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig("./gg_corr.pdf")
    plt.savefig("./gg_corr.png")
    plt.draw()
    plt.pause(1)  # needed to show plot

    input("Press any key to continue ...")
    plt.close()


def main():
    plot_distribution()


if __name__ == "__main__":
    main()
