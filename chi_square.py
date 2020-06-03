import numpy as np

def get_chi_square(data, data_yerror, fit_values):
    # compute the mean and the chi^2/dof
    z = (data - fit_values) / data_yerror
    chi2 = np.sum(z ** 2)
    return chi2

def get_reduced_chi_square(data, data_yerror, fit_values):
    chi2 = get_chi_square(data, data_yerror, fit_values)
    N = len(data) - 1
    print (N)
    return chi2 / N
