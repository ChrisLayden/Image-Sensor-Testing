import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.stats
import os
import time
from nonuniformity import ImageSequence

gain = 1.333
h = 6.62607015 * 10 ** -27 # erg s
c = 2.99792458 * 10 ** 10 # cm/s
photodiode_area = 1 # cm^2
pix_area = (3.76 * 10 ** -4) ** 2 # cm^2
exposure_time = 60 # s

wavelengths = np.array([640, 700, 750, 800, 980]) # nm
photosensitivities = np.array([334, 404, 464, 523, 697]) # mA/W
meas_currents = np.array([3.5, 4.0, 4.58, 4.9, 15.2]) * 10 ** -7 # mA

phot_energies = h * c / (wavelengths * 10 ** -7) * 10 ** -7 # J
phot_fluxes = meas_currents / (photosensitivities * phot_energies * photodiode_area) # phot/s/cm^2
mu_p_rates = phot_fluxes * pix_area # phot/s/pix

light_avgs = np.array([20000, 15000, 11700, 8300, 4000])
dark_avgs = np.array([100, 110, 110, 80, 80])
qes = (light_avgs - dark_avgs) / (mu_p_rates * exposure_time * gain)

print(qes)

# sup_dir = '/Volumes/KINGSTON/ASI2600Images/IMX571_Data/QE Tests/No Window'
# wavelengths = np.array([750]) # nm
# photosensitivities = np.array([397]) # mA/W
# meas_currents = np.array([0.15]) * 10 ** -6 # mA

# sup_dir = '/Volumes/KINGSTON/ASI2600Images/IMX571_Data/QE Tests/Window'
# wavelengths = np.array([640, 700, 750]) # nm
# photosensitivities = np.array([335, 369, 397]) # mA/W
# meas_currents = np.array([0.15, 0.15, 0.15]) * 10 ** -6 # mA

# phot_energies = h * c / (wavelengths * 10 ** -7) * 10 ** -7 # J
# phot_fluxes = meas_currents / (photosensitivities * phot_energies * photodiode_area) # phot/s/cm^2
# mu_p_rates = phot_fluxes * pix_area # phot/s/pix

# light_dirs = [sup_dir + '/' + str(wavelength) + ' nm/Light' for wavelength in wavelengths]
# dark_dirs = [sup_dir + '/' + str(wavelength) + ' nm/Dark' for wavelength in wavelengths]
# light_avgs = np.array([ImageSequence(light_dir).avg for light_dir in light_dirs])
# dark_avgs = np.array([ImageSequence(dark_dir).avg for dark_dir in dark_dirs])
# qes = (light_avgs - dark_avgs) / (mu_p_rates * exposure_time * gain)
# print(qes)

# # read fits file for expected imx571 data
# file = 'Data_Tables/imx571.fits'
# imx571_data = fits.open(file)[1].data
# hdr = fits.open(file)[1].header
# # convert to nanometers
# imx571_data['Wavelength'] = imx571_data['Wavelength'] / 10
# # plot the data

# plt.plot(imx571_data['Wavelength'], imx571_data['Throughput'], label='Online Data')

# window_wavelengths = np.array([640, 700, 750])
# window_qes = np.array([0.5069, 0.3722, 0.2732])
# no_window_wavelengths = np.array([750])
# no_window_qes = np.array([0.2775])
# plt.errorbar(window_wavelengths, window_qes, yerr=0.15, label='Window', fmt='o')
# plt.errorbar(no_window_wavelengths, no_window_qes, yerr=0.15, label='No Window', fmt='o')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Quantum Efficiency')
# plt.ylim(0,1)
# plt.legend()
# plt.show()