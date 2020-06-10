import numpy as np
import pynj3 as pj


def racah_w(a, b, c, d, e, f):
    return pow((-1), int(a+b+c+d))*pj.wigner6j(int(2*a),int(2*b),int(2*e),int(2*d),int(2*c),int(2*f))

def F(k, jf, l1, l2, ji):
    cg = pj.clebsch_gordon(l1, 1, l2, -1, k, 0)
    if cg == 0:
        return 0

    w = racah_w(ji, ji, l1, l2, k, jf)
    if w == 0:
        return 0

    return pow((-1), (jf-ji-1))*(pow((2*li+1)*(2*l2+1)*(2*ji+1),(1.0/2.0)))*cg*w
    # Reference: Tables of coefficients for angular distribution of gamma rays from aligned nuclei
	# T. Yamazaki. Nuclear Data A, 3(1):1?23, 1967.


def A(k, ji, jf, l1, l2, delta):
    f1 = F(k, ji, l1, l1, jf)
    f2 = F(k, ji, l1, l2, jf)
    f2 = F(k, ji, l2, l2, jf)
    return (1 / (1 + pow(delta, 2))) * (f1 + 2 * delta * f2 + pow(delta, 2) * f3)


def B(k, ji, jf, l1, l2, delta):
    f1 = F(k, ji, l1, l1, jf)
    f2 = F(k, ji, l1, l2, jf)
    f2 = F(k, ji, l2, l2, jf)
    return (1 / (1 + pow(delta, 2))) * (f1 + pow((-1), (l1 + l2)) * 2 * delta * f2 + pow(delta, 2) * f3)


def calculate_a2(j1, j2, j3, l1_lowest_allowed_spin, l1_mixing_spin, l2_lowest_allowed_spin, l2_mixing_spin, delta_1, delta_2):
    return B(2, j2, j1, l1_lowest_allowed_spin, l1_mixing_spin, delta_1) * A(2, j3, j2, l2_lowest_allowed_spin, l2_mixing_spin, delta2)


def calculate_a4(j1, j2, j3, l1_lowest_allowed_spin, l1_mixing_spin, l2_lowest_allowed_spin, l2_mixing_spin, delta_1, delta_2):
    return B(4, j2, j1, l1_lowest_allowed_spin, l1_mixing_spin, delta_1) * A(4, j3, j2, l2_lowest_allowed_spin, l2_mixing_spin, delta2)


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
    print(N)
    return chi2 / N


def mixing_ratio_chi_square_minimization(j_high, j_mid, j_low):
    '''Samples allowed mixing ratios and returns the mixing ratio with the smallest chi2 values

    Returns:

    Inputs:
        j_high Spin of the highest level
        j_mid Spin of the middle level
        j_low Spin of the lowest level
    '''

    # set to True for more output
    verbose = True

    if verbose:
        print("Staring chi2 minimization of mixing ratio")

    j1 = 0.5 * j_high
    j2 = 0.5 * j_mid
    j3 = 0.5 * j_low

    # l1 is the transition between j1 and j2
    # l2 is the transition between j2 and j3
    l1_lowest_allowed_spin = abs(j_high - j_mid) / 2
    l1_mixing_spin = l1_lowest_allowed_spin + 1
    l2_lowest_allowed_spin = abs(j_mid - j_low) / 2
    l2_mixing_spin = l2_lowest_allowed_spin + 1

    if l1_lowest_allowed_spin == 0:
        l1_lowest_allowed_spin = 1
    if l2_lowest_allowed_spin == 0:
        l2_lowest_allowed_spin = 1

    # Checks the values of the transition spin fit_values
    if (j_high == 0 and j_mid == 0) or (j_mid == 0 and j_low == 0):
        print(":::ERROR:::  I cannot handle a gamma transition between J=0 states ...")
    if l1_lowest_allowed_spin == abs(j_high + j_mid) / 2:
        print(
            f':::WARNING::: Only one angular momentum allowed for the high->middle ({j_high}->{j_mid}) transition.')
        print('The mixing ratio (delta1) will be fixed at zero')
        l1_mixing_spin = l1_lowest_allowed_spin
    if l2_lowest_allowed_spin == abs(j_mid + j_low) / 2:
        print(
            f':::WARNING::: Only one angular momentum allowed for the middle->low ({j_mid}->{j_low}) transition.')
        print('The mixing ratio (delta2) will be fixed at zero')
        l2_mixing_spin = l2_lowest_allowed_spin

    # -------------------------------------------------------------------
    #                       Constrained fitting
    # -------------------------------------------------------------------
    # Here's where you'll iterate over the physical quantities you don't
    # know, whether that be mixing ratios or spins or both. Here's an
    # example of two unknown mixing ratios but known spins.
    #
    # The basic idea is to select a particular set of physical quantities,
    # calculate a2/a4, fix the Method4Fit parameters, and fit the scaling
    # factor A0. Then output the specifications for that set of physical
    # quantities and the chi^2 for further analysis.
    # -------------------------------------------------------------------
    # delta runs from -infinity to infinity (unless constrained by known physics)
    # in this case, it then makes more sense to sample evenly from tan^{-1}(delta)
    # The next few lines are where you may want to include limits to significantly speed up calculations
    # mixing for the high-middle transition

    mixing_angle_min_1 = -np.pi / 2
    mixing_angle_max_1 = np.pi / 2
    steps_1 = 100
    step_size_1 = (mixing_angle_max_1 - mixing_angle_min_1) / steps_1
    # Mixing for middle-low transition
    # To constrain to zero, set mixing_angle_min_2 to 0, mixing_angle_min_2 to 1,
    # and steps_2 to 1
    mixing_angle_min_2 = -np.pi / 2
    mixing_angle_max_2 = np.pi / 2
    steps_2 = 100
    step_size_2 = (mixing_angle_max_2 - mixing_angle_min_2) / steps_2

    # Constraining the delta values (if appropriate)
    if l1_lowest_allowed_spin == l1_mixing_spin:
        mixing_angle_min_1 = 0
        steps_1 = 1
    if l2_lowest_allowed_spin == l2_mixing_spin:
        mixing_angle_min_2 = 0
        steps_2 = 1

    print(
        f'Sampling {steps_1} steps for mixing ratio one and {steps_2} for mixing ratio two')

    for i in range(steps_1):
        mix_angle_1 = mixing_angle_min_1 + i * step_size_1
        delta_1 = np.tan(mix_angle_1)
        if verbose:
            print(f'Mixing angle one: {mix_angle_1} Delta one: {delta_1}')

        for j in range(steps_2):
            mix_angle_2 = mixing_angle_min_2 + j * step_size_2
            delta_2 = np.tan(mix_angle_2)
            if verbose:
                print(f'Mixing angle two: {mix_angle_2} Delta two: {delta_2}')

    return 1
