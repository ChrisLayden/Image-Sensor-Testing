import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
from astropy.io import fits

# Calculates DSNU and PRNU for a sequence of images; plots spectrograms and histograms

class ImageSequence(object):

    def __init__(self, dir):
        '''Initialize an ImageSequence object

        Parameters
        ----------
        dir : str
            The directory containing the images. All images must be .fits files.
        '''
        
        self.dir = dir
        self.files = os.listdir(dir)
        self.img_files = [file for file in self.files if file.endswith('.fits')]
        # Exclude hidden files that SAOImage generates
        self.img_files = [file for file in self.img_files if not file.startswith('.')]
        self.num_img = len(self.img_files)
        self.first_img = fits.getdata(dir + '/' + self.img_files[0])
        self.img_shape = self.first_img.shape
        self.M = self.img_shape[0]
        self.N = self.img_shape[1]
        self.avg_img = self.get_avg_img()
        self.avg = np.mean(self.avg_img)
        self.var_img = self.get_var_img()
        self.temporal_var = np.mean(self.var_img)
        self.meas_var = np.var(self.avg_img)
        self.spatial_var = self.meas_var - self.temporal_var / self.num_img
        hdu = fits.PrimaryHDU(self.avg_img)
        hdul = fits.HDUList([hdu])
        hdul.writeto('/Users/layden/Downloads/avg_img.fits', overwrite=True)
        
        # self.dsnu = self.get_DSNU()

    def single_pix_values(self, i, j):
        '''Returns the values of the pixel at (i, j) for each image in the sequence'''
        return np.array([fits.getdata(self.dir + '/' + img_file)[i, j] for img_file in self.img_files])

    def plot_image_avgs(self):
        '''Plot the average of each image to identify outliers'''
        avg_list = np.zeros(self.num_img)
        for i, img_file in enumerate(self.img_files):
            img = fits.getdata(self.dir + '/' + img_file)
            avg_list[i] = np.mean(img)
        plt.plot(avg_list)
        plt.show()

    def get_avg_img(self):
        avg_img = np.zeros(self.img_shape)
        for img_file in self.img_files:
            avg_img += fits.getdata(self.dir + '/' + img_file)
        avg_img /= len(self.img_files)
        return avg_img

    def get_var_img(self):
        '''Returns the variance image of the sequence'''
        var_img = np.zeros(self.img_shape)
        for img_file in self.img_files:
            var_img += (fits.getdata(self.dir + '/' + img_file) - self.avg_img) ** 2
        var_img /= (self.num_img - 1)
        return var_img

    def get_temporal_var(self):
        '''Returns the variance of the temporal noise for the images'''
        return np.mean(self.var_img.sum())
    
    def get_DSNU(self):
        '''Returns the dark signal non-uniformity. Only valid for dark images'''
        # ZZZ need to divide by gain
        return np.sqrt(self.spatial_var)

    def get_PRNU(self, dark_seq):
        '''Returns the pixel response non-uniformity, in percent.'''
        return 100 * (np.sqrt(self.spatial_var - dark_seq.spatial_var) / (self.avg - dark_seq.avg))

    def power_spect_horiz(self):
        '''Returns the power spectrum of the image sequence averaged over all rows'''
        diff_img = self.avg_img - self.avg
        fourier_trans = np.fft.fft(diff_img, axis=1) / np.sqrt(self.img_shape[1])
        pow_spect = np.mean(np.absolute(fourier_trans) ** 2, axis=0)
        pow_spect -= self.temporal_var / self.num_img
        # Only go to the Nyguist frequency
        pow_spect = pow_spect[:self.N//2]
        return pow_spect

    def power_spect_vert(self):
        '''Returns the power spectrum of the image sequence averaged over all columns'''
        diff_img = self.avg_img - self.avg
        fourier_trans = np.fft.fft(diff_img, axis=0) / np.sqrt(self.img_shape[0])
        pow_spect = np.mean(np.absolute(fourier_trans) ** 2, axis=1)
        pow_spect -= self.temporal_var / self.num_img
        # Only go to the Nyguist frequency
        pow_spect = pow_spect[:self.M//2]
        return pow_spect

    def plot_spectrograms(self, is_dark=False):
        '''Plot both spectrograms on one figure with two separate axes'''
        pow_spect_horiz = self.power_spect_horiz()
        pow_spect_vert = self.power_spect_vert()
        fig, (ax1, ax2) = plt.subplots(2, 1)
        # If light image, plot as percent
        if is_dark:
            y_axis_horiz = np.sqrt(pow_spect_horiz)
            y_axis_vert = np.sqrt(pow_spect_vert)
            ax1.set_ylabel('Power (DN)')
            ax2.set_ylabel('Power (DN)')
        else:
            y_axis_horiz = np.sqrt(pow_spect_horiz) / self.avg * 100
            y_axis_vert = np.sqrt(pow_spect_vert) / self.avg * 100
            pow_spect_vert /= self.avg ** 2
            ax1.set_ylabel('Power (%)')
            ax2.set_ylabel('Power (%)')            
    
        ax1.plot(np.arange(self.N//2) / self.N, y_axis_horiz)
        ax1.set_xlabel('Cycles (Horizontal; periods/pix)')
        ax1.set_yscale('log')
        ax2.plot(np.arange(self.M//2) / self.M, y_axis_vert)
        ax2.set_xlabel('Cycles (Vertical; periods/pix)')
        ax2.set_yscale('log')
        plt.show()

    def white_noise(self):
        '''Returns the white noise of the image sequence, in percent.'''
        pow_horiz = self.power_spect_horiz()
        return np.sqrt(np.median(pow_horiz)) / self.avg

    def power_spect_dsnu(self):
        pow_spect = self.power_spect_horiz()
        return np.sum(pow_spect) / (self.N - 1)

    def power_spect_prnu(self):
        pow_spect = self.power_spect_horiz()
        return np.sum(pow_spect[8:-8]) / (self.N - 15)

    def high_pass_filter(self, image):
        '''Subtract a 5x5 box filter from the image'''
        box_filter = np.ones([5, 5]) / 25
        image = image - scipy.signal.convolve2d(image, box_filter, mode='same')
        # remove two border rows and columns
        image = image[2:-2, 2:-2]
        return image


    def plot_defect_hist(self, is_dark=False):
        '''Plot the histogram of the image sequence'''
        deviation_img = self.avg_img - self.avg
        if not is_dark:
            deviation_img = self.high_pass_filter(deviation_img)
        max_val = np.max(deviation_img)
        min_val = np.min(deviation_img)
        (hist, bins, patches) = plt.hist(deviation_img.flatten(), bins=256)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Plot Gaussian on top of histogram
        y = 1 / np.sqrt(2 * np.pi * self.meas_var) * np.exp(-bin_centers ** 2 / (2 * self.meas_var))
        # Get bins that are more than 2 sigma above the mean
        outlier_bins = np.where(abs(bin_centers) > 2 * np.sqrt(self.meas_var))
        print("Number of Defect Pixels (>5 sigma difference): ", np.sum(hist[outlier_bins] - y[outlier_bins]))
        plt.plot(bin_centers, y * self.M * self.N * (max_val - min_val) / 256, color='red')
        plt.ylim(bottom=0.1)
        plt.yscale('log')
        plt.xlabel('Deviation from Mean')
        plt.ylabel('Number of Pixels')
        plt.show()

    def plot_acc_defect_hist(self, is_dark=False):
        '''Plot the accumulated defect histogram of the image sequence'''
        deviation_img = self.avg_img - self.avg
        if not is_dark:
            deviation_img = self.high_pass_filter(deviation_img)
        deviation_img = abs(deviation_img)
        (hist, bins) = np.histogram(deviation_img.flatten(), bins=500)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        acc_hist = np.zeros_like(hist)
        for i in range(len(hist)):
            acc_hist[i] = np.sum(hist[i:])
        acc_hist = acc_hist / (self.M * self.N)
        bin_centers = bin_centers / self.avg * 100
        plt.plot(bin_centers, acc_hist)
        plt.xlabel('Deviation from Mean (%)')
        plt.ylabel('Fraction of Pixels With Larger Deviation')
        plt.yscale('log')
        plt.show()

    def get_gain(self, dark_seq):
        '''Returns the gain of the camera'''
        return (self.temporal_var - dark_seq.temporal_var) / (self.avg - dark_seq.avg)

if __name__ == '__main__':

    # prnu_test = ImageSequence('/Users/layden/Documents/TESS/Data-Images/EMVA-Test/PRNU')
    # dsnu_test = ImageSequence('/Users/layden/Documents/TESS/Data-Images/EMVA-Test/DSNU')
    # prnu_test.plot_histogram()
    # dsnu_test.plot_spectrograms()

    # light_dir = '/Volumes/KINGSTON/ASI2600Images/IMX571_Data/Linearity Tests/120 s/Light'
    # dark_dir = '/Volumes/KINGSTON/ASI2600Images/IMX571_Data/Linearity Tests/120 s/Dark'
    # light_dir = '/Volumes/KINGSTON/ASI2600Images/IMX571_Data/QE Tests/No Window/750 nm/Light'
    # dark_dir = '/Volumes/KINGSTON/ASI2600Images/IMX571_Data/QE Tests/No Window/750 nm/Dark'
    # light_seq = ImageSequence(light_dir)
    # dark_seq = ImageSequence(dark_dir)
    # light_seq.plot_image_avgs()


    times_list = ["5 s", "30 s", "60 s", "120 s"]
    times_array = np.array([5, 30, 60, 120])
    gain_list = np.zeros(len(times_list))
    light_avg_list = np.zeros(len(times_list))
    dark_avg_list = np.zeros(len(times_list))
    light_temp_var_list = np.zeros(len(times_list))
    dark_temp_var_list = np.zeros(len(times_list))
    for i, time in enumerate(times_list):
        light_dir = '/Volumes/DATA 1/ASI2600Images/IMX571_Data/Linearity Tests/' + time + '/Light'
        dark_dir = '/Volumes/DATA 1/ASI2600Images/IMX571_Data/Linearity Tests/' + time + '/Dark'
        light_seq = ImageSequence(light_dir)
        dark_seq = ImageSequence(dark_dir)
        gain = (light_seq.temporal_var - dark_seq.temporal_var) / (light_seq.avg - dark_seq.avg)
        print(gain)
        gain_list[i] = gain
        light_avg_list[i] = light_seq.avg
        dark_avg_list[i] = dark_seq.avg
        light_temp_var_list[i] = light_seq.temporal_var
        dark_temp_var_list[i] = dark_seq.temporal_var

    # snr_list = light_avg_list / np.sqrt(light_temp_var_list)
    # snr_max = np.sqrt(light_avg_list)
    # plt.plot(times_array, snr_list, 'o', label='Measured SNR')
    # plt.plot(times_array, snr_max, label='Theoretical Limit')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Exposure Time (s)')
    # plt.ylabel('SNR')
    # plt.legend()

    # # plt.plot(times_array, light_avg_list, 'o')
    # # plt.xlabel('Exposure Time (s)')
    # # plt.ylabel('Mean Signal (ADU)')
    # # x_list = light_avg_list - dark_avg_list
    # # y_list = light_temp_var_list - dark_temp_var_list
    # # plt.plot(x_list, y_list, 'o')
    # # plt.xlabel('Mean Signal (ADU)')
    # # plt.ylabel('Temporal Variance (ADU^2)')
    # plt.show()
    # gain = np.mean(gain_list)