#!/home/cnatzke/anaconda3/envs/py3/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.special import legendre
from scipy.optimize import curve_fit
from scipy.stats import chisquare

import chi_square as cs

#------------------------------------------------------------
def gamma_gamma_dist_func(x, a_0, a_2, a_4):
    p_2 = legendre(2)
    p_4 = legendre(4)
    return a_0 * (1 + a_2 * p_2(x) + a_4 * p_4(x))

#------------------------------------------------------------

df = pd.read_csv("./fits.txt", header=0, names=[
                 "Angle", "Corr_Area", "Corr_Area_Err", "Uncorr_Area", "Uncorr_Area_Err"], skiprows=1)
# print(df.head())

df["Cosine_Angle"] = np.cos(np.radians(df['Angle']))
df["Normalized_Area"] = df['Corr_Area'] / df['Uncorr_Area']
df["Normalized_Area_Err"] = df['Normalized_Area'] * \
    np.sqrt((df['Corr_Area_Err'] / df['Corr_Area']) ** 2 +
            (df['Uncorr_Area_Err'] / df['Uncorr_Area']) ** 2)
print(f'Normalized counts: \n {df.head()}')
df.to_csv('./event_mixed_counts.dat')

# plt.show()


#------------------------------------------------------------
# CURVE FITTING
#------------------------------------------------------------
popt, pcov = curve_fit(gamma_gamma_dist_func,
                       df['Cosine_Angle'], df['Normalized_Area'])
perr = np.sqrt(np.diag(pcov))

# calculating residuals (val - fit value)
df['Fit_Values'] = gamma_gamma_dist_func(df['Cosine_Angle'], *popt)
df['Residuals'] = df['Normalized_Area'] - df['Fit_Values']
chi2 = cs.get_chi_square(df['Normalized_Area'], df['Normalized_Area_Err'], df['Fit_Values'])
chi2_ndf = cs.get_reduced_chi_square(df['Normalized_Area'], df['Normalized_Area_Err'], df['Fit_Values'])

print(f'Fitted Parameters:\n {popt}')
print(f'Covariance Matrix:\n {pcov}')
print(f'Errors in paramters: {perr}')
print(f'Chi2: {chi2}')
print(f'Chi2/NDF: {chi2_ndf}')
print(f'Residuals: \n {df.head()}')

#------------------------------------------------------------
# PLOTTING
#------------------------------------------------------------
label_size = 20;
gamma_1 = 1172
gamma_2 = 1332

rc('text', usetex=True)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
title_font = {'size': label_size, 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}
axis_fontx = {'size': label_size, 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'top'}
axis_fonty = {'size': label_size, 'color': 'black', 'weight': 'normal',
              'verticalalignment': 'bottom'}
text_font = {'size': label_size, 'color': 'black', 'horizontalalignment': 'center', 'verticalalignment': 'center'}
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
a2_params = f'$a_2$ = {popt[1]:.6f}$\pm${perr[1]:.6f}'
a4_params = f'$a_4$ = {popt[2]:.6f}$\pm${perr[2]:.6f}'
chi2_params = f'$\chi^2/NDF$ = {chi2_ndf:.2f}'

axes[0].text(0.5, 0.7, a2_params, transform=axes[0].transAxes, **text_font)
axes[0].text(0.5, 0.6, a4_params, transform=axes[0].transAxes, **text_font)
axes[0].text(0.5, 0.5, chi2_params, transform=axes[0].transAxes, **text_font)

# residual plot
axes[1].errorbar(df['Cosine_Angle'], df['Residuals'], yerr=df['Normalized_Area_Err'], **error_plot_format)
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
