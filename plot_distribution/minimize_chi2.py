#!/home/cnatzke/anaconda3/envs/py3/bin/python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from lmfit import Model, Parameters

import chi_square as cs
import physics_functions as physics


def minimize_chi2():
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

    print("Starting chi2 minimization ... ")
    # Arguments should be double the physics value (e.g 7/2->7, 2->4, etc)
    j1 = 7/2
    j2 = 5/2
    j3 = 1/2
    delta_2_user = 0
    chi2_df = cs.minimize_mixing_chi2(df, 2 * j1, 2 * j2, 2 * j3, fix_a4=True, delta_2=delta_2_user)
    print("Starting chi2 minimization ... [DONE]")

    output_csv_name = f'chi2_values_{j1}_{j2}_{j3}.dat'
    print(f'Writing data to output file: {output_csv_name}')

    chi2_df.to_csv(output_csv_name, columns=[
                   'mixing_angle_1', 'mixing_angle_2', 'rcs'], sep='\t', index=False)


def main():
    minimize_chi2()


if __name__ == "__main__":
    main()
