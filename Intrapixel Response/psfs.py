'''Functions for calculating PSFs and optimal apertures.

Functions
---------
airy_ensq_energy : float
    The fraction of the light that hits a square of half-width p
    centered on an Airy disk PSF.
gaussian_ensq_energy : float
    The fraction of the light that hits a square of half-width p
    centered on a Gaussian PSF.
gaussian_psf : array-like
    An x-y grid with a Gaussian disk evaluated at each point.
airy_disk : array-like
    An x-y grid with the Airy disk evaluated at each point.
optimal_aperture : array-like
    The optimal aperture for maximizing S/N.
'''

import numpy as np
from scipy import special, integrate


# Calculate the energy in a square of dimensionless half-width p
# centered on an Airy disk PSF. From Eq. 7 in Torben Anderson's paper,
# Vol. 54, No. 25 / September 1 2015 / Applied Optics
# http://dx.doi.org/10.1364/AO.54.007525
def airy_ensq_energy(half_width):
    '''Returns the energy in a square of half-width p centered on an Airy PSF.

    Parameters
    ----------
    half_width : float
        The half-width of the square, defined in the paper linked above.

    Returns
    -------
    pix_fraction : float
        The fraction of the light that hits the square.
    '''
    def ensq_int(theta):
        '''Integrand to calculate the ensquared energy'''
        return 4/np.pi * (1 - special.jv(0, half_width/np.cos(theta))**2
                            - special.jv(1, half_width/np.cos(theta))**2)
    pix_fraction = integrate.quad(ensq_int, 0, np.pi/4)[0]
    return pix_fraction


# Calculate the energy in a square of half-width p (in um)
# centered on an Gaussian PSF with x and y standard deviations
# sigma_x and sigma_y, respectively. Currently doesn't let you
# have any covariance between x and y.
def gaussian_ensq_energy(half_width, sigma_x, sigma_y):
    '''Returns the energy in square of half-width p centered on a Gaussian PSF.

    Parameters
    ----------
    half_width : float
        The half-width of the square, in units of um.
    sigma_x : float
        The standard deviation of the Gaussian in the x direction, in um.
    sigma_y : float
        The standard deviation of the Gaussian in the y direction, in um.

    Returns
    -------
    pix_fraction : float
        The fraction of the light that hits the square.
    '''
    arg_x = half_width / np.sqrt(2) / sigma_x
    arg_y = half_width / np.sqrt(2) / sigma_y
    pix_fraction = special.erf(arg_x) * special.erf(arg_y)
    return pix_fraction


def gaussian_psf(num_pix, resolution, pix_size, mu, sigma):
    '''Return an x-y grid with a Gaussian disk evaluated at each point.

    Parameters
    ----------
    num_pix : int
        The number of pixels in the subarray.
    resolution : int
        The number of subpixels per pixel in the subarray.
    pix_size : float
        The size of each pixel in the subarray, in microns.
    mu : array-like
        The mean of the Gaussian, in microns.
    Sigma : array-like
        The covariance matrix of the Gaussian, in microns^2.

    Returns
    -------
    gaussian : array-like
        The Gaussian PSF evaluated at each point in the subarray,
        normalized to have a total amplitude of the fractional energy
        ensquared in the subarray.
    '''
    grid_points = num_pix * resolution
    x = np.linspace(-num_pix / 2.0, num_pix / 2.0, grid_points) * pix_size
    y = np.linspace(-num_pix / 2.0, num_pix / 2.0, grid_points) * pix_size
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    sigma_inv = np.linalg.inv(sigma)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    arg = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)
    # Determine the fraction of the light that hits the entire subarray
    array_p = num_pix / 2 * pix_size
    subarray_fraction = gaussian_ensq_energy(array_p, np.sqrt(sigma[0][0]), np.sqrt(sigma[1][1]))
    # Normalize the PSF to have a total amplitude of subarray_fraction
    gaussian = np.exp(-arg / 2)
    normalize = subarray_fraction / gaussian.sum()
    return gaussian * normalize


def airy_disk(num_pix, resolution, pix_size, mu, fnum, lam):
    '''Return an x-y grid with the Airy disk evaluated at each point.

    Parameters
    ----------
    num_pix : int
        The number of pixels in the subarray.
    resolution : int
        The number of subpixels per pixel in the subarray.
    pix_size : float
        The size of each pixel in the subarray, in microns.
    mu : array-like
        The mean position of the Airy disk, in pixels.
    fnum : float
        The f-number of the telescope.
    lam : float
        The wavelength of the light, in Angstroms.

    Returns
    -------
    airy : array-like
        The Airy disk evaluated at each point in the subarray,
        normalized to have a total amplitude of the fractional energy
        ensquared in the subarray.
    '''
    lam /= 10 ** 4
    grid_points = num_pix * resolution
    x = np.linspace(-num_pix / 2.0, num_pix / 2.0, grid_points) * pix_size
    y = np.linspace(-num_pix / 2.0, num_pix / 2.0, grid_points) * pix_size
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (1,))
    pos[:, :, 0] = np.sqrt((x - mu[0]) ** 2 + (y - mu[1]) ** 2)
    pos = pos[:, :, 0]
    arg = np.pi / (lam) / fnum * pos
    # Avoid singularity at origin
    arg[arg == 0] = 10 ** -10
    airy = (special.jv(1, arg) / arg) ** 2 / np.pi
    # Determine the fraction of the light that hits the entire subarray
    array_p = num_pix / 2 * pix_size * np.pi / fnum / lam
    subarray_fraction = airy_ensq_energy(array_p)
    # Normalize the PSF to have a total amplitude of subarray_fraction
    normalize = subarray_fraction / airy.sum()
    return airy * normalize

def moffat_psf(num_pix, resolution, pix_size, mu, alpha, beta):
    '''Return an x-y grid with a Moffat distribution evaluated at each point.

    Parameters
    ----------
    num_pix : int
        The number of pixels in the subarray.
    resolution : int
        The number of subpixels per pixel in the subarray.
    pix_size : float
        The size of each pixel in the subarray, in microns.
    mu : array-like
        The mean of the Gaussian, in pixels.
    alpha: float
        The width of the Moffat distribution, in microns.
    beta: float
        The power of the Moffat distribution.

    Returns
    -------
    gaussian : array-like
        The Gaussian PSF evaluated at each point in the subarray,
        normalized to have a total amplitude of the fractional energy
        ensquared in the subarray.
    '''
    grid_points = num_pix * resolution
    x = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    y = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (1,))
    pos[:, :, 0] = 1 + np.sqrt((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / alpha ** 2
    arg = (beta - 1) / (np.pi * alpha ** 2) * (1 + pos[:, :, 0]) ** -beta
    return arg
    # This isn't done yet--need to do normalization
    