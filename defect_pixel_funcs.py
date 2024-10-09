import numpy as np
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
from cosmos_helper_funcs import *

def plot_defect_hist(med_img, clip_type='sigma', cutoff_level=5, cut_high=True, cut_low=True, num_bins=512, filedata=None, iterate=True, show_gaussian=True, gain=None, bits=14):
    '''Plot the histogram of pixel values to identify defect pixels.'''
    if clip_type == 'sigma':
        defect_map, med_img_median, iterated_std, iter = sigma_clip_map(med_img, cutoff_level, iterate, cut_high, cut_low)
    elif clip_type == 'mad':
        defect_map, med_img_median, iterated_std, iter = mad_clip_map(med_img, cutoff_level, iterate, cut_high, cut_low)
    elif clip_type=='absolute':
        defect_map, med_img_median, iterated_std, iter = absolute_clip_map(med_img, cutoff_level, iterate, cut_high, cut_low)
    diff_img = med_img
    num_pix = med_img.size
    max_val = np.max(diff_img)
    min_val = np.min(diff_img)
    (hist, bins, patches) = plt.hist(diff_img.flatten(), bins=num_bins)
    defect_pix_count = np.count_nonzero(np.isnan(defect_map))
    defect_frac = defect_pix_count / num_pix
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    if show_gaussian:
        y = 1 / np.sqrt(2 * np.pi * iterated_std ** 2) * np.exp(-(bin_centers - med_img_median) ** 2 / (2 * iterated_std ** 2))
        y *= num_pix * (max_val - min_val) / num_bins
        plt.plot(bin_centers, y, color='red')
    plt.text(0.0, -0.2, r'Iterated $\sigma_{meas}$: ' + format(iterated_std, '3.2f'), color='red', transform=plt.gca().transAxes)
    # Put vertical black lines at the defect pixel cutoffs
    low_cutoff = med_img_median - cutoff_level * iterated_std
    high_cutoff = med_img_median + cutoff_level * iterated_std
    if clip_type == 'absolute':
        low_cutoff = med_img_median - cutoff_level
        high_cutoff = med_img_median + cutoff_level
        label_str = str(cutoff_level)
    elif clip_type == 'mad':
        label_str = str(cutoff_level) + r'$\sigma_{mad}$'
    else:
        label_str = str(cutoff_level) + r'$\sigma$'
    if cut_high:
        plt.axvline(x=high_cutoff, color='black')
        plt.text(high_cutoff, 1.0, '+' + label_str, color='black', horizontalalignment='left')
    if cut_low:
        plt.axvline(x=low_cutoff, color='black')    
        plt.text(low_cutoff, 1.0, '-' + label_str, color='black', horizontalalignment='right')
    if gain is not None:
        cutoff_dark_current = gain * high_cutoff / 120 # Exposure time is 120 seconds
        plt.text(0.0, -0.25, 'Cutoff Dark Current: ' + format(cutoff_dark_current, '3.2f') + ' e-/s', color='red', transform=plt.gca().transAxes)
    plt.text(0., -0.3, 'Pixel Fraction outside ' + str(cutoff_level) + r'$\sigma_{meas}$: ' + format(defect_frac * 100, '4.3f') + '%', color='red', transform=plt.gca().transAxes)
    plt.ylim(bottom=0.1)
    # plt.xlim(0, 2 ** bits)
    plt.yscale('log')
    plt.xscale('symlog')
    plt.xlabel('Difference from 1s Frame (ADU)')
    plt.ylabel('Number of Pixels')
    plt.title('Defect Pixel Histogram')
    if filedata is not None:
        label_plot(filedata)
    plt.show()
    return defect_map, med_img_median, iterated_std, iter

def sigma_clip_map(data, cutoff_level, iterate=True, cut_high=True, cut_low=True):
    '''Perform sigma-clipping on a dataset.
    
    Parameters
    ----------
    data : numpy array
        The data to be sigma-clipped
    cutoff_level : float
        The number of standard deviations to clip at
    iterate : bool (default=True)
        Whether to iterate the sigma-clipping
    cut_high : bool (default=True)
        Whether to clip high values
    cut_low : bool (default=True)
        Whether to clip low values

    Returns
    -------
    defect_map : numpy array
        An array with NaNs for clipped pixels
    '''
    data_median = np.median(data)
    defect_map = np.ones_like(data)
    new_var = np.nanvar(data)
    old_var = 0
    iter = 0
    while abs(new_var - old_var) > 1e-6:
        old_var = new_var
        if cut_high and cut_low:
            defect_map[abs(data - data_median) > cutoff_level * np.sqrt(old_var)] = np.NaN
        elif cut_high:
            defect_map[(data - data_median) > cutoff_level * np.sqrt(old_var)] = np.NaN
        elif cut_low:
            defect_map[- (data - data_median) > cutoff_level * np.sqrt(old_var)] = np.NaN
        data = data * defect_map
        new_var = np.nanvar(data)
        data_median = np.nanmedian(data)
        if not iterate:
            break
        iter += 1
    return defect_map, data_median, np.sqrt(new_var), iter

def mad_clip_map(data, mad_level, iterate=True, cut_high=True, cut_low=True):
    '''Perform MAD-clipping on a dataset.
    
    Parameters
    ----------
    data : numpy array
        The data to be sigma-clipped
    mad_level : float
        The number of median absolute deviations to clip at
    iterate : bool (default=True)
        Whether to iterate the clipping
    cut_high : bool (default=True)
        Whether to clip high values
    cut_low : bool (default=True)
        Whether to clip low values

    Returns
    -------
    defect_map : numpy array
        An array with NaNs for clipped pixels
    '''
    data_median = np.median(data)
    defect_map = np.ones_like(data)
    new_mad = median_abs_deviation(data, nan_policy='omit', axis=None)
    old_mad = 0
    iter = 0
    while abs(new_mad - old_mad) > 1e-6:
        old_mad = new_mad
        if cut_high and cut_low:
            defect_map[abs(data - data_median) > mad_level * np.sqrt(old_mad)] = np.NaN
        elif cut_high:
            defect_map[(data - data_median) > mad_level * np.sqrt(old_mad)] = np.NaN
        elif cut_low:
            defect_map[- (data - data_median) > mad_level * np.sqrt(old_mad)] = np.NaN
        data = data * defect_map
        new_mad = median_abs_deviation(data, nan_policy='omit', axis=None)
        data_median = np.nanmedian(data)
        if not iterate:
            break
        iter += 1
    return defect_map, data_median, np.sqrt(new_mad), iter

def absolute_clip_map(data, absolute_clip, iterate=True, cut_high=True, cut_low=True):
    '''Perform sigma-clipping on a dataset.
    
    Parameters
    ----------
    data : numpy array
        The data to be sigma-clipped
    absolute_clip : float
        The absolute value to clip at
    iterate : bool (default=True)
        Whether to iterate the sigma-clipping
    cut_high : bool (default=True)
        Whether to clip high values
    cut_low : bool (default=True)
        Whether to clip low values

    Returns
    -------
    defect_map : numpy array
        An array with NaNs for clipped pixels
    '''
    data_median = np.median(data)
    defect_map = np.ones_like(data)
    if cut_high and cut_low:
        defect_map[abs(data - data_median) > absolute_clip] = np.NaN
    elif cut_high:
        defect_map[(data - data_median) > absolute_clip] = np.NaN
    elif cut_low:
        defect_map[- (data - data_median) > absolute_clip] = np.NaN
    data = data * defect_map
    new_var = np.nanvar(data)
    data_median = np.nanmedian(data)
    iter = 0
    return defect_map, data_median, np.sqrt(new_var), iter

# import time
# test_array = np.random.normal(0, 1, 100000)
# t0 = time.time()
# test_array_sigma_clip, med, sigma, iter = sigma_clip_map(test_array, 3)
# t1 = time.time()
# test_array_mad_clip, med, mad, iter2 = mad_clip_map(test_array, 3.5)
# t2 = time.time()
# print(f'Sigma clipping took {t1 - t0} seconds with {iter} iterations')
# print(f'MAD clipping took {t2 - t1} seconds with {iter2} iterations')
# num_nan_sigma_clip = np.count_nonzero(np.isnan(test_array_sigma_clip))
# num_nan_mad_clip = np.count_nonzero(np.isnan(test_array_mad_clip))
# print(f'Number of NaNs from sigma clipping: {num_nan_sigma_clip}')
# print(f'Number of NaNs from MAD clipping: {num_nan_mad_clip}')