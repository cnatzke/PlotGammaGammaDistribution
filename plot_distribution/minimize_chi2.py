#!/home/cnatzke/anaconda3/envs/py3/bin/python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from lmfit import Model, Parameters

import chi_square as cs
import physics_functions as physics


def test_chi2_minimization():
    verbose = 0

    print(f'Opening file: {sys.argv[1]}')
    df = pd.read_csv(sys.argv[1], header=0, names=[
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

    print("Starting chi2 minimization ... ")
    # Arguments should be double the physics value (e.g 7/2->7, 2->4, etc)
    j1 = 2
    j2 = 2
    j3 = 0
    chi2_df = cs.minimize_mixing_chi2(df, 2 * j1, 2 * j2, 2 * j3)
    print("Starting chi2 minimization ... [DONE]")

    output_csv_name = f'chi2_values_{j1}_{j2}_{j3}.dat'
    print(f'Writing data to output file: {output_csv_name}')

    chi2_df.to_csv(output_csv_name, columns=[
                   'mixing_angle_1', 'mixing_angle_2', 'rcs'], sep='\t', index=False)


def main():
    test_chi2_minimization()


if __name__ == "__main__":
    main()
