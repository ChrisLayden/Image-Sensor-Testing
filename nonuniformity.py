import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
from astropy.io import fits
import time

# Calculates DSNU and PRNU for a sequence of images; plots spectrograms and histograms

class NonuniformityCalc(object):

    def __init__(self, gray_dir, dark_dir, dsnu_dir, num_imgs=None, do_filtering=False):
        '''Initialize a NonuniformityCalc object

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
        do_filtering : bool (default=False)
            If True, apply a high pass filter to the images before calculating nonuniformity.
            This is recommended for EMVA 4.0 Linear compliance testing.
        '''
        
        self.gray_dir = gray_dir
        self.dark_dir = dark_dir
        self.dsnu_dir = dsnu_dir
        self.do_filtering = do_filtering
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
        self.avg_gray_img = self.get_avg_img(img_type='gray')
        self.avg_dark_img = self.get_avg_img(img_type='dark')
        self.avg_dsnu_img = self.get_avg_img(img_type='dsnu')
        self.M_gray = self.avg_gray_img.shape[0]
        self.N_gray = self.avg_gray_img.shape[1]
        self.M_dsnu = self.avg_dsnu_img.shape[0]
        self.N_dsnu = self.avg_dsnu_img.shape[1]
        # Calculate temporal variances (variance of each pixel over the sequence of images)
        # before high-pass filtering. This avoids having to high-pass filter each individual
        # image and gives pretty much the same result.
        self.gray_temporal_var = np.mean(self.get_var_img(img_type='gray'))
        self.dark_temporal_var = np.mean(self.get_var_img(img_type='dark'))
        self.dsnu_temporal_var = np.mean(self.get_var_img(img_type='dsnu'))
        if self.do_filtering:
            self.avg_gray_img = self.high_pass_filter_2020(self.avg_gray_img)
            self.avg_dark_img = self.high_pass_filter_2020(self.avg_dark_img)
            # Filtered images have slightly fewer pixels due to cutting off edges
            self.M_gray = self.avg_gray_img.shape[0]
            self.N_gray = self.avg_gray_img.shape[1]
        self.gray_meas_var = np.var(self.avg_gray_img)
        self.dark_meas_var = np.var(self.avg_dark_img)
        self.dsnu_meas_var = np.var(self.avg_dsnu_img)
        self.defect_pix = self.defect_pix_map('gray') * self.defect_pix_map('dsnu')
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
        img_shape = fits.getdata(self.gray_dir + '/' + self.gray_files[0]).shape
        avg_img = np.zeros(img_shape)
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
            num_rows = self.M_gray
            num_img = len(self.gray_files)
        elif img_type == 'dsnu':
            diff_img = self.avg_dsnu_img - self.avg_dsnu_img.mean()
            temporal_var = self.dsnu_temporal_var
            num_rows = self.M_dsnu
            num_img = len(self.dsnu_files)
        fourier_trans = np.fft.fft(diff_img, axis=1) / np.sqrt(num_rows)
        pow_spec = np.mean(np.absolute(fourier_trans) ** 2, axis=0)
        pow_spec -= temporal_var / num_img
        return pow_spec
    
    def power_spect_vert(self, img_type):
        '''Returns the power spectrum of the image sequence averaged over all rows'''
        if img_type == 'gray':
            diff_img = (self.avg_gray_img - self.avg_dark_img) - (self.avg_gray_img - self.avg_dark_img).mean()
            temporal_var = self.gray_temporal_var
            num_cols = self.N_gray
            num_img = len(self.gray_files)
        elif img_type == 'dsnu':
            diff_img = self.avg_dsnu_img - self.avg_dsnu_img.mean()
            temporal_var = self.dsnu_temporal_var
            num_cols = self.N_dsnu
            num_img = len(self.dsnu_files)
        fourier_trans = np.fft.fft(diff_img, axis=0) / np.sqrt(num_cols)
        pow_spec = np.mean(np.absolute(fourier_trans) ** 2, axis=1)
        pow_spec -= temporal_var / num_img
        return pow_spec

    def plot_gray_spectrograms(self, logx=False, logy=True):
        '''Plot both spectrograms on one figure with two separate axes'''
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout(pad=2)
        avg = np.mean(self.avg_gray_img - self.avg_dark_img)
        y_axis_horiz = np.sqrt(self.gray_pow_spec_horiz) / avg * 100
        prnu_horiz = np.sqrt(np.sum(self.gray_pow_spec_horiz[7:-7]) / self.N_gray - 15) / avg * 100
        y_axis_vert = np.sqrt(self.gray_pow_spec_vert) / avg * 100
        prnu_vert = np.sqrt(np.sum(self.gray_pow_spec_vert[7:-7]) / self.M_gray - 15) / avg * 100
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
        ax1.plot(np.arange(self.N_gray//2) / self.N_gray, y_axis_horiz[:self.N_gray//2])
        ax1.set_xlabel('Cycles (Horizontal; periods/pix)')
        ax2.plot(np.arange(self.M_gray//2) / self.M_gray, y_axis_vert[:self.M_gray//2])
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
        dsnu_horiz = np.sqrt(np.sum(self.dsnu_pow_spec_horiz) / self.N_dsnu - 1)
        dsnu_horiz /= gain
        y_axis_vert = np.sqrt(self.dsnu_pow_spec_vert)
        dsnu_vert = np.sqrt(np.sum(self.dsnu_pow_spec_vert) / self.M_dsnu - 1)
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
        ax1.plot(np.arange(self.N_dsnu//2) / self.N_dsnu, y_axis_horiz[:self.N_dsnu//2])
        ax1.set_xlabel('Cycles (Horizontal; periods/pix)')
        ax2.plot(np.arange(self.M_dsnu//2) / self.M_dsnu, y_axis_vert[:self.M_dsnu//2])
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
            center = (self.M_gray // 2, self.N_gray // 2)
        # Find the radial distance from the center for each pixel in the average image
        t0 = time.time()
        x = np.arange(self.N_gray) - center[1]
        y = np.arange(self.M_gray) - center[0]
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
        dsnu_horiz = np.sqrt(np.sum(self.dsnu_pow_spec_horiz) / (self.N_dsnu - 1))
        dsnu_vert = np.sqrt(np.sum(self.dsnu_pow_spec_vert) / (self.M_dsnu - 1))
        return dsnu_horiz, dsnu_vert

    def power_spect_prnu(self):
        # Returns PRNU estimated from spectrogram in percent
        prnu_horiz = np.sqrt(np.sum(self.gray_pow_spec_horiz) / (self.N_gray - 1))
        prnu_vert = np.sqrt(np.sum(self.gray_pow_spec_vert) / (self.M_gray - 1))
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
        # Note that this filter maintains a larger fraction of the high-frequency signal, maintaining a total
        # of 99.4% of white noise (in standard deviation, not variance). So you don't really need to divide
        # by anything to correct for noise lost due to the filter.
        box_filter_7 = np.ones([7, 7]) / 49
        box_filter_11 = np.ones([11, 11]) / 121
        binomial_filter_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        new_image = scipy.signal.convolve2d(image, box_filter_7, mode='valid')
        new_image = scipy.signal.convolve2d(new_image, box_filter_11, mode='valid')
        new_image = scipy.signal.convolve2d(new_image, binomial_filter_3x3, mode='valid')
        new_image = np.mean(image) + image[9:-9, 9:-9] - new_image
        return new_image

    def plot_defect_hist(self, img_type):
        '''Plot the histogram of pixel values to identify defect pixels.'''
        if img_type == 'gray':
            avg_img = self.avg_gray_img
            avg = np.mean(avg_img)
            meas_var = self.gray_meas_var
            num_pix = self.M_gray * self.N_gray
            plt.title('PRNU Defect Histogram')
        elif img_type == 'dsnu':
            avg_img = self.avg_dsnu_img
            avg = np.mean(avg_img)
            meas_var = self.dsnu_meas_var
            num_pix = self.M_dsnu * self.N_dsnu
            plt.title('DSNU Defect Histogram')
        deviation_img = avg_img - avg
        max_val = np.max(deviation_img)
        min_val = np.min(deviation_img)
        num_bins = 256
        (hist, bins, patches) = plt.hist(deviation_img.flatten(), bins=num_bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Plot Gaussian on top of histogram
        y = 1 / np.sqrt(2 * np.pi * meas_var) * np.exp(-bin_centers ** 2 / (2 * meas_var))
        y *= num_pix * (max_val - min_val) / num_bins
        # Get bins that are more than 2 sigma outside of the mean
        outlier_bins = np.where(abs(bin_centers) > 5 * np.sqrt(meas_var))
        expected_5sigma_pixels = np.sum(hist) * 5.73 * 10 ** (-7)
        defect_pixels = np.sum(hist[outlier_bins])  - expected_5sigma_pixels
        plt.plot(bin_centers, y, color='red')
        plt.text(0.05, 0.95, r'$\sigma_{meas}$: ' + format(np.sqrt(meas_var), '.2f'), color='red', transform=plt.gca().transAxes)
        plt.text(0.05, 0.9, r'Pixels outside $5\sigma_{meas}$: ' + format(int(defect_pixels), '3d'), color='red', transform=plt.gca().transAxes)
        plt.ylim(bottom=0.1)
        plt.yscale('log')
        plt.xlabel('Deviation from Mean')
        plt.ylabel('Number of Pixels')
        plt.show()

    def defect_pix_map(self, img_type):
        '''Return an array with 1s for pixels that are more than 5 sigma from the mean'''
        if img_type == 'gray':
            avg_img = self.avg_gray_img
            avg = np.mean(avg_img)
            meas_var = self.gray_meas_var
        elif img_type == 'dsnu':
            avg_img = self.avg_dsnu_img
            avg = np.mean(avg_img)
            meas_var = self.dsnu_meas_var
        deviation_img = avg_img - avg
        defect_map = np.ones_like(avg_img)
        defect_map[abs(deviation_img) > 5 * np.sqrt(meas_var)] = 0
        # If we're doing filtering, pad the defect map with zeros to get back to the
        # original size. This means edge pixels will not be noticed as defect pixels.
        if img_type == 'gray' and self.do_filtering:
            defect_map = np.pad(defect_map, ((9, 9), (9, 9)), 'constant', constant_values=1)
        return defect_map

    def plot_profiles(self, img_type):
        '''Plot middle, minimum, maximum, and average profiles for columns and rows'''
        if img_type == 'gray':
            avg_img = self.avg_gray_img - self.avg_dark_img
            horiz_middle = avg_img[self.M_gray // 2, :]
            vert_middle = avg_img[:, self.N_gray // 2]
        elif img_type == 'dsnu':
            avg_img = self.avg_dsnu_img
            horiz_middle = avg_img[self.M_dsnu // 2, :]
            vert_middle = avg_img[:, self.N_dsnu // 2]
        avg = np.mean(avg_img)
        horiz_mean = np.mean(avg_img, axis=0)
        vert_mean = np.mean(avg_img, axis=1)
        horiz_min = np.min(avg_img, axis=0)
        vert_min = np.min(avg_img, axis=1)
        horiz_max = np.max(avg_img, axis=0)
        vert_max = np.max(avg_img, axis=1)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout(pad=2)
        ax1.plot(horiz_min, label='Min')
        ax1.plot(horiz_max, label='Max')
        ax1.plot(horiz_middle, label='Middle')
        ax1.plot(horiz_mean, label='Mean')
        ax1.set_xlabel('Horizontal')
        ax1.set_ylabel('Deviation from Mean')
        ax1.set_ylim(0.9 * avg, 1.1 * avg)
        ax1.legend()
        ax2.plot(vert_min, label='Min')
        ax2.plot(vert_max, label='Max')
        ax2.plot(vert_middle, label='Middle')
        ax2.plot(vert_mean, label='Mean')
        ax2.set_xlabel('Vertical')
        ax2.set_ylabel('Deviation from Mean')
        ax2.set_ylim(0.9 * avg, 1.1 * avg)
        ax2.legend()
        plt.show()

    def get_read_noise(self, num_images=5, gain=1):
        '''Calculate the read noise of the camera'''
        avg_img = self.avg_dsnu_img
        sq_diff_img = np.zeros_like(avg_img)
        for i in range(num_images):
            img = fits.getdata(self.dsnu_dir + '/' + self.dsnu_files[i])
            sq_diff_img += (img - avg_img) ** 2 / num_images
        read_noise = np.sqrt(np.mean(sq_diff_img)) * gain
        return read_noise

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
    nonuniformity = NonuniformityCalc(gray_dir, dark_dir, dsnu_dir, num_imgs=5, do_filtering=False)
    print("PRNU: ", nonuniformity.prnu)
    print("DSNU: ", nonuniformity.dsnu)
    print("Read Noise: ", nonuniformity.get_read_noise(gain=0.971))
    # nonuniformity.plot_profiles('dsnu')
    nonuniformity.plot_defect_hist('dsnu')
    nonuniformity.plot_defect_hist('gray')
    # nonuniformity.plot_dsnu_spectrograms(gain=0.971)
    # nonuniformity.plot_gray_spectrograms()
    dark_defect_map = nonuniformity.defect_pix_map('dsnu')
    gray_defect_map = nonuniformity.defect_pix_map('gray')
    full_defect_map = dark_defect_map * gray_defect_map
    plt.imshow(gray_defect_map, cmap='gray')
    plt.show()
    print('Number of DSNU Defect Pixels: ', np.sum(dark_defect_map == 0))
    print('Number of Gray Defect Pixels: ', np.sum(gray_defect_map == 0))
    print('Number of Defect Pixels: ', np.sum(full_defect_map == 0))

