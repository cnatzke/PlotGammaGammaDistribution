import numpy as np
import sympy.physics.wigner as wg


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
    #cg_coeff_homebrew = float(clebsch_gordan(l1, 1, l2, -1, k, 0)) # works, but takes arguments in different order compared to the sympy version
    cg_coeff = float(wg.clebsch_gordan(l1, l2, k, 1, -1, 0))
    if cg_coeff == 0:
        return 0

    w = float(wg.racah(ji, ji, l1, l2, k, jf))
    #w = racah_w(ji, ji, l1, l2, k, jf) # returning strange values

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
