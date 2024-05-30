import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
from astropy.io import fits
import time

# Calculates DSNU and PRNU for a sequence of images; plots spectrograms and histograms

class Nonuniformity(object):

    def __init__(self, gray_dir, dark_dir, dsnu_dir, num_imgs=None):
        '''Initialize a Nonuniformity object

        Parameters
        ----------
        gray_dir : directory path
            Folder containing the sequence of gray images, each a .fits file. Each image
            should be at 50% saturation, and there should be ~100-500 images.
        dark_images : directory path
            Folder containing the sequence of dark images, each a .fits file. Each image
            should be taken with the shutter closed, but everything else the same as the
            gray images.
        dsnu_dir : directory path
            Folder containing the sequence of dark images, each a .fits file. Each image
            should be taken with a light shield in front of the camera and with minimal
            exposure time.
        num_imgs : int (default=None)
            Number of images to use in the sequence. If None, use all images in the directory.
            Useful for testing with a smaller dataset.
        '''
        
        self.gray_dir = gray_dir
        self.dark_dir = dark_dir
        self.dsnu_dir = dsnu_dir
        self.gray_files = [file for file in os.listdir(self.gray_dir) if file.endswith('.fits')]
        self.dark_files = [file for file in os.listdir(self.dark_dir) if file.endswith('.fits')]
        self.dsnu_files = [file for file in os.listdir(self.dsnu_dir) if file.endswith('.fits')]
        # Exclude hidden files that might get generated
        self.gray_files = [file for file in self.gray_files if not file.startswith('.')]
        self.dark_files = [file for file in self.dark_files if not file.startswith('.')]
        self.dsnu_files = [file for file in self.dsnu_files if not file.startswith('.')]
        if num_imgs is not None:
            self.gray_files = self.gray_files[:num_imgs]
            self.dark_files = self.dark_files[:num_imgs]
            self.dsnu_files = self.dsnu_files[:num_imgs]
        self.img_shape = fits.getdata(gray_dir + '/' + self.gray_files[0]).shape
        self.M = self.img_shape[0]
        self.N = self.img_shape[1]
        self.avg_gray_img = self.get_avg_img(img_type='gray')
        self.avg_dark_img = self.get_avg_img(img_type='dark')
        self.avg_dsnu_img = self.get_avg_img(img_type='dsnu')
        self.gray_var_img = self.get_var_img(img_type='gray')
        self.dark_var_img = self.get_var_img(img_type='dark')
        self.dsnu_var_img = self.get_var_img(img_type='dsnu')
        self.gray_temporal_var = np.mean(self.gray_var_img)
        self.dark_temporal_var = np.mean(self.dark_var_img)
        self.dsnu_temporal_var = np.mean(self.dsnu_var_img)
        self.avg_gray_img = self.high_pass_filter(self.avg_gray_img)
        self.avg_dark_img = self.high_pass_filter(self.avg_dark_img)
        self.avg_dsnu_img = self.avg_dsnu_img[9:-9, 9:-9]
        self.M = self.M - 18
        self.N = self.N - 18
        self.gray_meas_var = np.var(self.avg_gray_img)
        self.dark_meas_var = np.var(self.avg_dark_img)
        self.dsnu_meas_var = np.var(self.avg_dsnu_img)
        self.gray_spatial_var = self.gray_meas_var - self.gray_temporal_var / len(self.gray_files)
        self.dark_spatial_var = self.dark_meas_var - self.dark_temporal_var / len(self.dark_files)
        self.dsnu_spatial_var = self.dsnu_meas_var - self.dsnu_temporal_var / len(self.dsnu_files)
        self.prnu = 100 * np.sqrt(self.gray_spatial_var - self.dark_spatial_var) / np.mean(self.avg_gray_img - self.avg_dark_img)
        self.dsnu = np.sqrt(self.dsnu_spatial_var)
        self.gray_pow_spec_horiz = self.power_spect_horiz('gray')
        self.gray_pow_spec_vert = self.power_spect_vert('gray')
        self.dsnu_pow_spec_horiz = self.power_spect_horiz('dsnu')
        self.dsnu_pow_spec_vert = self.power_spect_vert('dsnu')

    def get_avg_img(self, img_type):
        avg_img = np.zeros(self.img_shape)
        if img_type == 'gray':
            dir = self.gray_dir
            img_files = self.gray_files
        elif img_type == 'dark':
            dir = self.dark_dir
            img_files = self.dark_files
        elif img_type == 'dsnu':
            dir = self.dsnu_dir
            img_files = self.dsnu_files
        for img_file in img_files:
            avg_img += fits.getdata(dir + '/' + img_file)
        avg_img /= len(img_files)
        return avg_img

    def get_var_img(self, img_type):
        '''Returns the variance image of the sequence'''
        if img_type == 'gray':
            dir = self.gray_dir
            img_files = self.gray_files
            avg_img = self.avg_gray_img
        elif img_type == 'dark':
            dir = self.dark_dir
            img_files = self.dark_files
            avg_img = self.avg_dark_img
        elif img_type == 'dsnu':
            dir = self.dsnu_dir
            img_files = self.dsnu_files
            avg_img = self.avg_dsnu_img
        var_img = np.zeros(np.shape(avg_img))
        for img_file in img_files:
            var_img += (fits.getdata(dir + '/' + img_file) - avg_img) ** 2
        num_img = len(img_files)
        var_img /= (num_img - 1)
        return var_img

    def power_spect_horiz(self, img_type):
        '''Returns the power spectrum of the image sequence averaged over all rows'''
        if img_type == 'gray':
            diff_img = (self.avg_gray_img - self.avg_dark_img) - (self.avg_gray_img - self.avg_dark_img).mean()
            temporal_var = self.gray_temporal_var
            num_img = len(self.gray_files)
        elif img_type == 'dsnu':
            diff_img = self.avg_dsnu_img - self.avg_dsnu_img.mean()
            temporal_var = self.dsnu_temporal_var
            num_img = len(self.dsnu_files)
        fourier_trans = np.fft.fft(diff_img, axis=1) / np.sqrt(self.img_shape[1])
        pow_spec = np.mean(np.absolute(fourier_trans) ** 2, axis=0)
        pow_spec -= temporal_var / num_img
        return pow_spec
    
    def power_spect_vert(self, img_type):
        '''Returns the power spectrum of the image sequence averaged over all rows'''
        if img_type == 'gray':
            diff_img = (self.avg_gray_img - self.avg_dark_img) - (self.avg_gray_img - self.avg_dark_img).mean()
            temporal_var = self.gray_temporal_var
            num_img = len(self.gray_files)
        elif img_type == 'dsnu':
            diff_img = self.avg_dsnu_img - self.avg_dsnu_img.mean()
            temporal_var = self.dsnu_temporal_var
            num_img = len(self.dsnu_files)
        fourier_trans = np.fft.fft(diff_img, axis=0) / np.sqrt(self.img_shape[0])
        pow_spec = np.mean(np.absolute(fourier_trans) ** 2, axis=1)
        pow_spec -= temporal_var / num_img
        return pow_spec

    def plot_gray_spectrograms(self, logx=False, logy=True):
        '''Plot both spectrograms on one figure with two separate axes'''
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout(pad=2)
        avg = np.mean(self.avg_gray_img - self.avg_dark_img)
        y_axis_horiz = np.sqrt(self.gray_pow_spec_horiz) / avg * 100
        prnu_horiz = np.sqrt(np.sum(self.gray_pow_spec_horiz[7:-7]) / self.N - 15) / avg * 100
        y_axis_vert = np.sqrt(self.gray_pow_spec_vert) / avg * 100
        prnu_vert = np.sqrt(np.sum(self.gray_pow_spec_vert[7:-7]) / self.M - 15) / avg * 100
        white_noise_horiz = np.median(y_axis_horiz)
        white_noise_vert = np.median(y_axis_vert)
        nonwhite_factor_horiz = (prnu_horiz / white_noise_horiz) ** 2
        nonwhite_factor_vert = (prnu_vert / white_noise_vert) ** 2
        ax1.set_ylabel('Power (%)')
        ax2.set_ylabel('Power (%)')
        sigma_y = np.sqrt(self.gray_temporal_var - self.dark_temporal_var) / avg * 100
        ax1.axhline(sigma_y, color='black', linestyle='dashed')
        ax1.text(0.45, sigma_y + 0.5, r'$\sigma_y$: ' + format(sigma_y, '.2f') + '%')
        ax2.axhline(sigma_y, color='black', linestyle='dashed')
        ax2.text(0.45, sigma_y + 0.5, r'$\sigma_y$: ' + format(sigma_y, '.2f') + '%')
        ax1.text(0.4, 0.6, 'PRNU: ' + format(prnu_horiz, '.2f') + '%',
                 color='red', transform=ax1.transAxes)
        ax1.text(0.4, 0.5, r'$s_w$: ' + format(white_noise_horiz, '.2f') + '%',
                 color='red', transform=ax1.transAxes)
        ax1.text(0.4, 0.4, 'F: ' + format(nonwhite_factor_horiz, '.2f'),
                 color='red', transform=ax1.transAxes)
        ax2.text(0.4, 0.6, 'PRNU: ' + format(prnu_vert, '.2f') + '%',
                 color='red', transform=ax2.transAxes)
        ax2.text(0.4, 0.5, r'$s_w$: '  + format(white_noise_vert, '.2f') + '%',
                 color='red', transform=ax2.transAxes)
        ax2.text(0.4, 0.4, 'F: ' + format(nonwhite_factor_vert, '.2f'),
                 color='red', transform=ax2.transAxes)
        ax1.plot(np.arange(self.N//2) / self.N, y_axis_horiz[:self.N//2])
        ax1.set_xlabel('Cycles (Horizontal; periods/pix)')
        ax2.plot(np.arange(self.M//2) / self.M, y_axis_vert[:self.M//2])
        ax2.set_xlabel('Cycles (Vertical; periods/pix)')
        if logx:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        if logy:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
        plt.show()
        
    def plot_dsnu_spectrograms(self, gain, logx=False, logy=True):
        '''Plot both spectrograms on one figure with two separate axes'''
        # Gain in units ADU/e-
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout(pad=2)
        y_axis_horiz = np.sqrt(self.dsnu_pow_spec_horiz)
        dsnu_horiz = np.sqrt(np.sum(self.dsnu_pow_spec_horiz) / self.N - 1)
        dsnu_horiz /= gain
        y_axis_vert = np.sqrt(self.dsnu_pow_spec_vert)
        dsnu_vert = np.sqrt(np.sum(self.dsnu_pow_spec_vert) / self.M - 1)
        dsnu_vert /= gain
        white_noise_horiz = np.median(y_axis_horiz)
        white_noise_horiz /= gain
        white_noise_vert = np.median(y_axis_vert)
        white_noise_vert /= gain
        nonwhite_factor_horiz = (dsnu_horiz / white_noise_horiz) ** 2
        nonwhite_factor_vert = (dsnu_vert / white_noise_vert) ** 2
        ax1.set_ylabel('Power (ADU)')
        ax2.set_ylabel('Power (ADU)')
        sigma_y = self.dsnu_temporal_var
        ax1.axhline(sigma_y, color='black', linestyle='dashed')
        ax1.text(0.45, sigma_y + 0.5, r'$\sigma_y$: ' + format(sigma_y, '.2f'))
        ax2.axhline(sigma_y, color='black', linestyle='dashed')
        ax2.text(0.45, sigma_y + 0.5, r'$\sigma_y$: ' + format(sigma_y, '.2f'))
        ax1.text(0.4, 0.6, 'DSNU: ' + format(dsnu_horiz, '.2f') + ' e-',
                 color='red', transform=ax1.transAxes)
        ax1.text(0.4, 0.5, r'$s_w$: ' + format(white_noise_horiz, '.2f') + ' e-',
                 color='red', transform=ax1.transAxes)
        ax1.text(0.4, 0.4, 'F: ' + format(nonwhite_factor_horiz, '.2f'),
                 color='red', transform=ax1.transAxes)
        ax2.text(0.4, 0.6, 'DSNU: ' + format(dsnu_vert, '.2f') + ' e-',
                 color='red', transform=ax2.transAxes)
        ax2.text(0.4, 0.5, r'$s_w$: '  + format(white_noise_vert, '.2f') + ' e-',
                 color='red', transform=ax2.transAxes)
        ax2.text(0.4, 0.4, 'F: ' + format(nonwhite_factor_vert, '.2f'),
                 color='red', transform=ax2.transAxes)
        ax1.plot(np.arange(self.N//2) / self.N, y_axis_horiz[:self.N//2])
        ax1.set_xlabel('Cycles (Horizontal; periods/pix)')
        ax2.plot(np.arange(self.M//2) / self.M, y_axis_vert[:self.M//2])
        ax2.set_xlabel('Cycles (Vertical; periods/pix)')
        if logx:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
        if logy:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
        plt.show()

    def radial_periodogram(self, center=None):
        '''Plot a radial spectrogram of the gray images.'''
        import time
        diff_img = (self.avg_gray_img - self.avg_dark_img)
        if center is None:
            center = (self.M // 2, self.N // 2)
        # Find the radial distance from the center for each pixel in the average image
        t0 = time.time()
        x = np.arange(self.N) - center[1]
        y = np.arange(self.M) - center[0]
        X, Y = np.meshgrid(x, y)
        # Round to nearest pixel
        R = np.rint(np.sqrt(X ** 2 + Y ** 2)).astype(int)
        # Flatten the image and radial distance arrays
        diff_img_flat = diff_img.flatten()
        R_flat = R.flatten()
        t1 = time.time()
        print("Time to set up: ", t1 - t0)
        # Sort the radial distance array
        sort_indices = np.argsort(R_flat)
        R_flat = R_flat[sort_indices]
        diff_img_flat = diff_img_flat[sort_indices]
        t2 = time.time()
        print("Time to sort: ", t2 - t1)
        max_r = np.max(R_flat)
        r_vals = np.arange(max_r + 1)
        diff_img_radial = np.zeros_like(r_vals)
        r = 0
        count = 0
        for i, val in enumerate(diff_img_flat):
            if r == R_flat[i]:
                count += 1
                diff_img_radial[r] += val
            else:
                diff_img_radial[r] /= count
                r = R_flat[i]
                count = 1
                diff_img_radial[r] += val
        t3 = time.time()
        print("Time to average: ", t3 - t2)
        diff_img_radial -= np.mean(diff_img_radial).astype(int)
        fourier_trans = np.fft.fft(diff_img_radial)
        pow_spec = np.absolute(fourier_trans) ** 2
        # Just plot the first half
        r_plot_vals = r_vals[:len(r_vals) // 2] / max_r
        pow_spec = pow_spec[:len(pow_spec) // 2]
        plt.plot(r_plot_vals, pow_spec)
        plt.xlabel('Cycles (Radial; periods/pix)')
        plt.ylabel('Power')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
    def power_spect_dsnu(self):
        dsnu_horiz = np.sqrt(np.sum(self.dsnu_pow_spec_horiz) / (self.N - 1))
        dsnu_vert = np.sqrt(np.sum(self.dsnu_pow_spec_vert) / (self.M - 1))
        return dsnu_horiz, dsnu_vert

    def power_spect_prnu(self):
        # Returns PRNU estimated from spectrogram in percent
        prnu_horiz = np.sqrt(np.sum(self.gray_pow_spec_horiz) / (self.N - 1))
        prnu_vert = np.sqrt(np.sum(self.gray_pow_spec_vert) / (self.M - 1))
        prnu_horiz = prnu_horiz / np.mean(self.avg_gray_img - self.avg_dark_img) * 100
        prnu_vert = prnu_vert / np.mean(self.avg_gray_img - self.avg_dark_img) * 100
        return prnu_horiz, prnu_vert

    def high_pass_filter(self, image):
        '''Subtract a 5x5 box filter from the image'''
        box_filter = np.ones([5, 5]) / 25
        avg_signal = image.mean()
        image = avg_signal + image[2:-2, 2:-2] - scipy.signal.convolve2d(image, box_filter, mode='valid')
        return image
    
    def high_pass_filter_2020(self, image):
        '''Use the more complicated high pass filter from EMVA 4.0 Linear'''
        box_filter_7 = np.ones([7, 7]) / 49
        box_filter_11 = np.ones([11, 11]) / 121
        binomial_filter_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        new_image = scipy.signal.convolve2d(image, box_filter_7, mode='valid')
        new_image = scipy.signal.convolve2d(new_image, box_filter_11, mode='valid')
        new_image = scipy.signal.convolve2d(new_image, binomial_filter_3x3, mode='valid')
        new_image = np.mean(image) + image[9:-9, 9:-9] - new_image
        return new_image

    def plot_defect_hist(self, img_type):
        '''Plot the histogram of the image sequence'''
        if img_type == 'gray':
            avg_img = self.avg_gray_img
            avg = np.mean(avg_img)
            meas_var = self.gray_meas_var
            plt.title('PRNU Defect Histogram')
        elif img_type == 'dsnu':
            avg_img = self.avg_dsnu_img
            avg = np.mean(avg_img)
            meas_var = self.dsnu_meas_var
            plt.title('DSNU Defect Histogram')
        deviation_img = avg_img - avg
        max_val = np.max(deviation_img)
        min_val = np.min(deviation_img)
        num_bins =256
        (hist, bins, patches) = plt.hist(deviation_img.flatten(), bins=num_bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Plot Gaussian on top of histogram
        y = 1 / np.sqrt(2 * np.pi * meas_var) * np.exp(-bin_centers ** 2 / (2 * meas_var))
        y *= self.M * self.N * (max_val - min_val) / num_bins
        # Get bins that are more than 2 sigma outside of the mean
        outlier_bins = np.where(abs(bin_centers) > 5 * np.sqrt(meas_var))
        expected_5sigma_pixels = np.sum(hist) * 5.73 * 10 ** (-7)
        defect_pixels = np.sum(hist[outlier_bins])  - expected_5sigma_pixels
        print("Number of Defect Pixels (Excess pixels with >5 sigma difference): ", defect_pixels)
        plt.plot(bin_centers, y, color='red')
        plt.text(0.05, 0.95, r'$\sigma_{meas}$: ' + format(np.sqrt(meas_var), '.2f'), color='red', transform=plt.gca().transAxes)
        plt.text(0.05, 0.9, r'Pixels outside $5\sigma_{meas}$: ' + format(int(defect_pixels), '3d'), color='red', transform=plt.gca().transAxes)
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

if __name__ == '__main__':

    # dirCOSMOS = '/Users/layden/Library/CloudStorage/Box-Box/Scientific CMOS - MKI ONLY (might contain EAR; ITAR)/Teledyne_COSMOS/MKI Lab data/04-22-2024/Data Set 1 - 9am'
    # gray_images_name = dirCOSMOS + '/HighSHighG_RS_raw_1us 2024 April 22 09_25_04.fits'
    # new_dir = '/Users/layden/Documents/COSMOS/Nonuniformity/HS_HG_RS_DSNUImages'
    # # This fits file contains many images. Separate each one into its own fits file.
    # img_data = fits.getdata(gray_images_name)
    # for i in range(50):
    #     new_filename = new_dir + '/dsnu_image_' + str(i) + '.fits'
    #     hdu = fits.PrimaryHDU(img_data[i])
    #     hdul = fits.HDUList([hdu])
    #     hdul.writeto(new_filename, overwrite=True)
    home_dir = '/Users/layden/Documents/COSMOS/Nonuniformity'
    gray_dir = home_dir + '/HS_HG_RS_F5A_GrayImages'
    dark_dir = home_dir + '/HS_HG_RS_F5A_DarkImages'
    dsnu_dir = home_dir + '/HS_HG_RS_DSNUImages'
    nonuniformity = Nonuniformity(gray_dir, dark_dir, dsnu_dir, num_imgs=5)
    print("PRNU: ", nonuniformity.prnu)
    print("DSNU: ", nonuniformity.dsnu)
    nonuniformity.plot_defect_hist('dsnu')
    nonuniformity.plot_defect_hist('gray')
    nonuniformity.plot_dsnu_spectrograms(gain=0.971)
    nonuniformity.plot_gray_spectrograms()

