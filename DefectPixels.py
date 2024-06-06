import numpy as np
import matplotlib.pyplot as plt

def plot_defect_hist(med_img, sigma_level=5, num_bins=512):
    '''Plot the histogram of pixel values to identify defect pixels.'''
    diff_img = med_img - np.median(med_img)
    defect_map, iterated_std = defect_pix_map(med_img, sigma_level)
    num_pix = med_img.size
    max_val = np.max(diff_img)
    min_val = np.min(diff_img)
    (hist, bins, patches) = plt.hist(diff_img.flatten(), bins=num_bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Plot Gaussian on top of histogram
    y = 1 / np.sqrt(2 * np.pi * iterated_std ** 2) * np.exp(-bin_centers ** 2 / (2 * iterated_std ** 2))
    y *= num_pix * (max_val - min_val) / num_bins
    # Get number of pixels outside of sigma_level * sigma of the mean
    defect_pix_count = np.count_nonzero(np.isnan(defect_map))
    plt.plot(bin_centers, y, color='red')
    plt.text(0.5, 0.95, r'Iterated $\sigma_{meas}$: ' + format(iterated_std, '.2f'), color='red', transform=plt.gca().transAxes)
    plt.text(0.5, 0.9, 'Pixels outside ' + str(sigma_level) + r'$\sigma_{meas}$: ' + format(int(defect_pix_count), '3d'), color='red', transform=plt.gca().transAxes)
    # Put vertical black lines at the defect pixel cutoffs
    plt.axvline(x=sigma_level * iterated_std, color='black')
    plt.axvline(x=-sigma_level * iterated_std, color='black')
    plt.ylim(bottom=0.1)
    plt.yscale('log')
    plt.xlabel('Deviation from Median')
    plt.ylabel('Number of Pixels')
    plt.title('Defect Pixel Histogram')
    plt.show()

def defect_pix_map(med_img, sigma_level):
    '''Return an array with NaNs for pixels that are more than sigma_level from the median'''
    deviation_img = med_img - np.median(med_img)
    defect_map = np.ones_like(deviation_img)
    new_var = np.nanvar(deviation_img)
    old_var = 0
    while abs(new_var - old_var) > 1e-6:
        old_var = new_var
        defect_map[abs(deviation_img) > sigma_level * np.sqrt(old_var)] = np.NaN
        deviation_img = deviation_img * defect_map
        new_var = np.nanvar(deviation_img)    
    return defect_map, np.sqrt(new_var)