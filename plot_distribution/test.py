#!/home/cnatzke/anaconda3/envs/py3/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from lmfit import Model, Parameters

import chi_square as cs
import physics_functions as physics


def test_chi2_minimization():
    verbose = 0

    df = pd.read_csv("./207Bi/fits.txt", header=0, names=[
                     "Index", "Angle", "Corr_Area", "Corr_Area_Err", "Uncorr_Area", "Uncorr_Area_Err"], skiprows=1)
    # print(df.head())

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

    test_values = cs.minimize_mixing_chi2(df, 7, 5, 1)

    print(test_values)

    fig, ax = plt.subplots()
    ax.plot(test_values['mixing_angle_1'], test_values['mixing_angle_2'])
    plt.show()


def main():
    test_chi2_minimization()


if __name__ == "__main__":
    main()
