import numpy as np
import pandas as pd
from scipy.special import legendre
import sympy.physics.wigner as wg

def gamma_gamma_dist_func(x, a_0, a_2, a_4):
    p_2 = legendre(2)
    p_4 = legendre(4)
    return a_0 * (1 + a_2 * p_2(x) + a_4 * p_4(x))

def clebsch_gordan(j1, m1, j2, m2, j, m):
    # safety checks
    if 2 * j1 != np.floor(2 * j1) or \
       2 * j2 != np.floor(2 * j2) or \
       2 * j != np.floor(2 * j) or \
       2 * m1 != np.floor(2 * m1) or \
       2 * m2 != np.floor(2 * m2) or \
       2 * m != np.floor(2 * m):
        print(":::clebsch_gordan::: All arguments must be integers or half-integers")
        return 0

        if((m1 + m2) != m3):
            printf("m1 + m2 must equal m.\n")
            return 0

        if((j1 - m1) != np.floor(j1 - m1)):
            printf("2*j1 and 2*m1 must have the same parity")
            return 0

        if((j2 - m2) != np.floor(j2 - m2)):
            printf("2*j2 and 2*m2 must have the same parity")
            return 0

        if(j3 - m3 != np.floor(j3 - m3)):
            printf("2*j and 2*m must have the same parity")
            return 0

        if(j3 > (j1 + j2) or j3 < abs(j1 - j2)):
            printf("j is out of bounds.")
            return 0

        if(abs(m1) > j1):
            printf("m1 is out of bounds.")
            return 0

        if(abs(m2) > j2):
            printf("m2 is out of bounds.")
            return 0

        if(abs(m3) > j3):
            printf("m is out of bounds.\n")
            return 0

    term1 = pow((((2 * j + 1) / np.math.factorial(j1 + j2 + j + 1)) * np.math.factorial(j2 + j - j1) * np.math.factorial(j + j1 - j2) * np.math.factorial(j1 + j2 - j) *
                 np.math.factorial(j1 + m1) * np.math.factorial(j1 - m1) * np.math.factorial(j2 + m2) * np.math.factorial(j2 - m2) * np.math.factorial(j + m) * np.math.factorial(j - m)), (0.5))

    sum = 0

    for k in range(100):
        if (j1 + j2 - j - k < 0) or (j1 - m1 - k < 0) or (j2 + m2 - k < 0):
            # No further tersm will contribute to the sum
            break
        elif (j - j1 - m2 + k < 0) or (j - j2 + m1 + k < 0):
            a1 = (j - j1 - m2)
            a2 = (j - j2 + m1)
            k = max(-min(a1, a2) - 1, k)
        else:
            term = np.math.factorial(j1 + j2 - j - k) * np.math.factorial(j - j1 - m2 + k) * np.math.factorial(
                j - j2 + m1 + k) * np.math.factorial(j1 - m1 - k) * np.math.factorial(j2 + m2 - k) * np.math.factorial(k)
            if ((k % 2) == 1):
                term = -1 * term
            sum = sum + 1 / term

    cg = term1 * sum
    return cg
    # Reference: An Effective Algorithm for Calculation of the C.G.
    # Coefficients Liang Zuo, et. al.
    # J. Appl. Cryst. (1993). 26, 302-304


def racah_w(a, b, c, d, e, f):
    return pow((-1), int(a + b + d + c)) * float(wg.wigner_6j(int(2 * a), int(2 * b), int(2 * e), int(2 * d), int(2 * c), int(2 * f)))


def F(k, jf, l1, l2, ji):

    verbose = False
    # cg_coeff_homebrew = float(clebsch_gordan(l1, 1, l2, -1, k, 0)) # works, but takes arguments in different order compared to the sympy version
    cg_coeff = float(wg.clebsch_gordan(l1, l2, k, 1, -1, 0))
    if cg_coeff == 0:
        return 0

    w = float(wg.racah(ji, ji, l1, l2, k, jf))
    # w = racah_w(ji, ji, l1, l2, k, jf) # returning strange values

    if w == 0:
        return 0

    if verbose:
        print(f'GC: {cg_coeff} W: {w}')

    return pow((-1), (jf - ji - 1)) * (pow((2 * l1 + 1) * (2 * l2 + 1) * (2 * ji + 1), (1.0 / 2.0))) * cg_coeff * w
    # Reference: Tables of coefficients for angular distribution of gamma rays from aligned nuclei
    # T. Yamazaki. Nuclear Data A, 3(1):1?23, 1967.


def A(k, ji, jf, l1, l2, delta):
    f1 = F(k, ji, l1, l1, jf)
    f2 = F(k, ji, l1, l2, jf)
    f3 = F(k, ji, l2, l2, jf)
    return (1 / (1 + pow(delta, 2))) * (f1 + 2 * delta * f2 + pow(delta, 2) * f3)


def B(k, ji, jf, l1, l2, delta):
    f1 = F(k, jf, l1, l1, ji)
    f2 = F(k, jf, l1, l2, ji)
    f3 = F(k, jf, l2, l2, ji)
    return (1 / (1 + pow(delta, 2))) * (f1 + pow((-1), (l1 + l2)) * 2 * delta * f2 + pow(delta, 2) * f3)


def calculate_a2(j1, j2, j3, l1_lowest_allowed_spin, l1_mixing_spin, l2_lowest_allowed_spin, l2_mixing_spin, delta_1, delta_2):
    return B(2, j2, j1, l1_lowest_allowed_spin, l1_mixing_spin, delta_1) * A(2, j3, j2, l2_lowest_allowed_spin, l2_mixing_spin, delta_2)


def calculate_a4(j1, j2, j3, l1_lowest_allowed_spin, l1_mixing_spin, l2_lowest_allowed_spin, l2_mixing_spin, delta_1, delta_2):
    return B(4, j2, j1, l1_lowest_allowed_spin, l1_mixing_spin, delta_1) * A(4, j3, j2, l2_lowest_allowed_spin, l2_mixing_spin, delta_2)


def sample_a2_a4_space(j_high, j_mid, j_low):
    '''Samples allowed mixing ratios for fitting

    Returns:
        mix_samples PANDAS dataframe containing sampled values

    Inputs:
        j_high Spin of the highest level
        j_mid Spin of the middle level
        j_low Spin of the lowest level
    '''

    # set to True for more output
    verbose = False

    if verbose:
        print("Staring chi2 minimization of mixing ratio")

    j1 = 0.5 * j_high
    j2 = 0.5 * j_mid
    j3 = 0.5 * j_low

    # l1 is the transition between j1 and j2
    l1_lowest_allowed_spin = abs(j_high - j_mid) / 2
    if l1_lowest_allowed_spin == 0:
        l1_lowest_allowed_spin = 1
    l1_mixing_spin = l1_lowest_allowed_spin + 1

    # l2 is the transition between j2 and j3
    l2_lowest_allowed_spin = abs(j_mid - j_low) / 2
    if l2_lowest_allowed_spin == 0:
        l2_lowest_allowed_spin = 1
    l2_mixing_spin = l2_lowest_allowed_spin + 1

    # Checks the values of the transition spin fit_values
    if (j_high == 0 and j_mid == 0) or (j_mid == 0 and j_low == 0):
        print(":::ERROR:::  I cannot handle a gamma transition between J=0 states ...")
    if l1_lowest_allowed_spin == abs(j_high + j_mid) / 2:
        print(
            f':::WARNING::: Only one angular momentum allowed for the high->middle ({j_high}->{j_mid}) transition.')
        print('The mixing ratio (delta1) will be fixed at zero\n')
        l1_mixing_spin = l1_lowest_allowed_spin
    if l2_lowest_allowed_spin == abs(j_mid + j_low) / 2:
        print(
            f':::WARNING::: Only one angular momentum allowed for the middle->low ({j_mid}->{j_low}) transition.')
        print('The mixing ratio (delta2) will be fixed at zero\n')
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

    sampled_values = []

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

            a2 = calculate_a2(j1, j2, j3, l1_lowest_allowed_spin, l1_mixing_spin,
                                      l2_lowest_allowed_spin, l2_mixing_spin, delta_1, delta_2)
            a4 = calculate_a4(j1, j2, j3, l1_lowest_allowed_spin, l1_mixing_spin,
                                      l2_lowest_allowed_spin, l2_mixing_spin, delta_1, delta_2)
            if verbose:
                print(f'Calculated a2: {a2:.6f} a4: {a4:.6f}')

            sample_value = [mix_angle_1, mix_angle_2, a2, a4]
            sampled_values.append(sample_value)

    columns = ['mixing_angle_1', 'mixing_angle_2', 'a2', 'a4']
    df = pd.DataFrame(sampled_values, columns=columns)
    return df
