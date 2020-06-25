import numpy as np
import pandas as pd
import physics_functions as physics
from lmfit import Model, Parameters


def get_chi_square(data, data_yerror, fit_values):
    '''Calculates the chi squared for input data arrays

    Inputs:
        data Array of data points
        data_yerror Array of the yerrors for data
        fit_values Array of fitted function values

    Returns:
        chi2 Chi-squared values
    '''
    z = (data - fit_values) / data_yerror
    chi2 = np.sum(z ** 2)
    return chi2


def get_reduced_chi_square(data, data_yerror, fit_values):
    '''Calculates the reduced chi squared for input data arrays

    Inputs:
        data Array of data points
        data_yerror Array of the yerrors for data
        fit_values Array of fitted function values

    Returns:
        rcs Reduced chi2
    '''
    chi2 = get_chi_square(data, data_yerror, fit_values)
    N = len(data) - 1
    #print(N)
    return chi2 / N

def minimize_mixing_chi2(data, j_high, j_mid, j_low, verbose=0, fix_a4=False, delta_2=-1):

    # Sample allowed a2/a4 space
    df = physics.sample_a2_a4_space(j_high, j_mid, j_low, fix_a4, delta_2)

    #------------------------------------------------------------
    # CURVE FITTING
    #------------------------------------------------------------
    gmodel = Model(physics.gamma_gamma_dist_func)
    params = Parameters()
    params.add('a_0', value = 1.0)

    # loop over all values of a2 & a4 to find the minimum chi2
    rcs_list = []
    for index, row in df.iterrows():
        params.add('a_2', value = row['a2'], vary=False)
        params.add('a_4', value = row['a4'], vary=False)
        #params.add('a_4', value = 0.5, vary=False) #fixed parameter

        result = gmodel.fit(data['Normalized_Area'], params, x=data['Cosine_Angle'])
        if verbose > 0:
            print('---------------------------------')
            print('FIT RESULTS')
            print('---------------------------------')
            print(result.fit_report())
            print('---------------------------------')

        # calculating residuals (val - fit value)
        data['Fit_Values'] = result.eval(x=data['Cosine_Angle'])
        data['Residuals'] = data['Normalized_Area'] - data['Fit_Values']
        chi2 = get_chi_square(data['Normalized_Area'], data['Normalized_Area_Err'], data['Fit_Values'])
        chi2_ndf = get_reduced_chi_square(data['Normalized_Area'], data['Normalized_Area_Err'], data['Fit_Values'])

        rcs_list.append(chi2_ndf)

    df['rcs'] = rcs_list
    return df
