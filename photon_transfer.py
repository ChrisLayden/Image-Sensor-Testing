import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.optimize

# dir = '/Users/layden/Desktop/IMX487 Photon Transfer'
# dir = os.getcwd() + '\\..\\Saved Images\\Photon Transfer'
dir = '/Users/layden/Library/CloudStorage/Box-Box/clayden7/TESS-GEO Sensor Testing/IMX487/Photon Transfer'
# Gain: 25 dB
# Setup: filter 7b (500 nm), photodetector control signal set to 0.5, exposure time ranging from 1 ms to 8 s

class PhotonTransfer(object):
    '''A class for creating photon transfer curves.'''

    def __init__(self, dir, bits=16):
        '''Initialize a photon transfer object.
        
        Parameters
        ----------
        dir : str
            The directory containing two subdirectories: one titled Dark Images
            and one titled Gray Images. Within these subdirectories, the name of
            each image should start with the exposure time in milliseconds,
            e.g. '0.1ms_1.fits'. Image files may be in either .fits or .png
            format. For each exposure time, there should be at least two dark
            images and two gray images. By default, it uses only the first two
            images for each exposure time.
        bits : int
            The number of bits used to represent each pixel. Default is 16.
        '''

        self.dir = dir
        self.bits = bits
        self.dark_dir = dir + '/Dark Images'
        self.gray_dir = dir + '/Gray Images'
        self.dark_files = os.listdir(self.dark_dir)
        self.gray_files = os.listdir(self.gray_dir)
        self.exposure_times = self.get_exposure_times()
        self.gray_mean_list = np.zeros(len(self.exposure_times))
        self.gray_temp_var_list = np.zeros(len(self.exposure_times))
        self.dark_mean_list = np.zeros(len(self.exposure_times))
        self.dark_temp_var_list = np.zeros(len(self.exposure_times))
        for i, time in enumerate(self.exposure_times):
            results = self.mean_and_var(time)
            self.gray_mean_list[i] = results[0]
            self.gray_temp_var_list[i] = results[1]
            self.dark_mean_list[i] = results[2]
            self.dark_temp_var_list[i] = results[3]
        (self.gain, self.gain_err) = self.get_gain()

    def get_img_array(self, file):
        '''Returns a numpy array of the image data.'''
        if file.endswith('.fits'):
            return fits.getdata(file).astype(int)

        else:
            raise ValueError('File must be in .fits or .png format')

    def get_exposure_times(self):
        '''Returns a list of the exposure times in milliseconds.'''
        dark_times = []
        gray_times = []
        for file in self.gray_files:
            if file.endswith('.fits') or file.endswith('.png'):
                gray_times.append(file.split('ms')[0])
        for file in self.dark_files:
            if file.endswith('.fits') or file.endswith('.png'):
                dark_times.append(file.split('ms')[0])
        # Only keep times for which there are at least two dark images
        # and two gray images
        exposure_times = [time for time in dark_times if
                          (dark_times.count(time) >= 2 and
                           gray_times.count(time) >= 2)]
        # Drop duplicates
        exposure_times = list(set(exposure_times))
        # Sort the strings according to their numerical value
        return(sorted(exposure_times, key=lambda x: float(x)))

    def mean_and_var(self, exposure_time):
        dark_files = [file for file in self.dark_files if
                      file.startswith(exposure_time + 'ms') and file.endswith('.fits')]
        gray_files = [file for file in self.gray_files if
                      file.startswith(exposure_time + 'ms') and file.endswith('.fits')]
        gray_image_0 = self.get_img_array(self.gray_dir + '/' + gray_files[0])
        gray_image_1 = self.get_img_array(self.gray_dir + '/' + gray_files[1])
        dark_image_0 = self.get_img_array(self.dark_dir + '/' + dark_files[0])
        dark_image_1 = self.get_img_array(self.dark_dir + '/' + dark_files[1])
        gray_mean = np.mean(gray_image_0 + gray_image_1) / 2
        dark_mean = np.mean(dark_image_0 + dark_image_1) / 2
        gray_temporal_var = np.mean((gray_image_0 - gray_image_1) ** 2) / 2
        dark_temporal_var = np.mean((dark_image_0 - dark_image_1) ** 2) / 2
        return [gray_mean, gray_temporal_var, dark_mean, dark_temporal_var]
    
    def get_gain(self):
        '''Returns the best fit gain, and std. error of this gain, in ADU/e-.'''
        x = self.gray_mean_list - self.dark_mean_list
        y = self.gray_temp_var_list - self.dark_temp_var_list
        max_x = 2 ** self.bits
        # Fit up to 70% of max ADU to a line
        popt, pcov = scipy.optimize.curve_fit(lambda x, a: a * x,
                                              x[x < 0.7 * max_x],
                                              y[x < 0.7 * max_x])
        gain = popt[0]
        gain_err = np.sqrt(pcov[0][0])
        return gain, gain_err
    
    def plot_phot_transfer(self):
        '''Plots the photon transfer curve.'''
        x = self.gray_mean_list - self.dark_mean_list
        y = self.gray_temp_var_list - self.dark_temp_var_list
        max_x = 2 ** self.bits
        plt.plot(x, y, 'o', label='Temporal Variance')
        plt.plot(x[x < 0.7 * max_x], self.gain * x[x < 0.7 * max_x], label='Fit')
        # Display the gain on the plot
        plt.text(0.05 * max_x, 0.95 * max(y), 'Gain: ' + format(self.gain, '.2f')
                 + r'$\pm$' + format(self.gain_err, '.2f') + ' ADU/e-')
        plt.xlabel(r'$\mu_{gray} - \mu_{dark}$ (ADU)')
        plt.ylabel(r'$\sigma_{gray}^2 - \sigma_{dark}^2$ (ADU$^2$)')
        plt.title('Photon Transfer Curve')
        plt.show()


photTransfer = PhotonTransfer(dir, bits=12)
photTransfer.plot_phot_transfer()


# class LinearityMeasurement(object):

#     def __init__(self, dir, reg=[0, int(1e6), 0, int(1e6)]):
#         '''Initialize a linearity measurement object
        
#         Parameters
#         ----------
#         dir : str
#             The directory containing subdirectories of images taken at different exposure times
#         '''
#         self.dir = dir
#         self.sub_dirs = os.listdir(dir)
#         self.exposure_times = [float(sub_dir.split(' ')[0]) for sub_dir in self.sub_dirs]
#         self.num_times = len(self.exposure_times)
#         one_image = self.get_one_img(self.sub_dirs[0])
#         (self.M, self.N) = one_image.shape
#         reg[1] = min(reg[1], self.M)
#         reg[3] = min(reg[3], self.N)
#         self.reg = reg
#         self.mu_list = self.get_mu_list()
#         self.mu_list_dark = self.get_mu_list(dark=True)
#         self.var_list = self.get_var_list()
#         self.var_list_dark = self.get_var_list(dark=True)
        

#     def gray_value_mean(self, image_1, image_2):
#         '''Returns the mean gray value of the two images'''
#         return (np.mean(image_1) + np.mean(image_2)) / 2

#     def gray_value_temp_var(self, image_1, image_2):
#         '''Returns the temporal variance of the gray value of the two images'''
#         return np.mean((image_1 - image_2) ** 2) / 2

#     def get_one_img(self, sub_dir, dark=False, img_num=0):
#         '''Returns one image from a subdirectory'''
#         if dark:
#             dir_str = "Dark"
#         else:
#             dir_str = "Light"
#         img_files = os.listdir(self.dir + '/' + sub_dir + '/' + dir_str)
#         img_files = [file for file in img_files if file.endswith('.fits')]
#         img = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[img_num])
#         return img

#     def get_mu_list(self, dark=False):
#         '''Returns a list of the mean gray values of the images in each subdirectory'''
#         if dark:
#             dir_str = "Dark"
#         else:
#             dir_str = "Light"
#         mu_list = np.zeros(self.num_times)
#         for i, sub_dir in enumerate(self.sub_dirs):
#             img_files = os.listdir(self.dir + '/' + sub_dir + '/' + dir_str)
#             img_files = [file for file in img_files if file.endswith('.fits')]
#             img_1 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[0])
#             img_2 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[1])
#             img_1 = img_1[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
#             img_2 = img_2[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
#             mu_list[i] = self.gray_value_mean(img_1, img_2)
#         return np.array(mu_list)

#     def get_var_list(self, dark=False):
#         '''Returns a list of the temporal variance of gray values of the images in each subdirectory'''
#         if dark:
#             dir_str = "Dark"
#         else:
#             dir_str = "Light"
#         var_list = np.zeros(self.num_times)
#         for i, sub_dir in enumerate(self.sub_dirs):
#             print(sub_dir)
#             img_files = os.listdir(self.dir + '/' + sub_dir + '/' + dir_str)
#             img_files = [file for file in img_files if file.endswith('.fits')]
#             img_1 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[0])
#             img_2 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[1])
#             img_1 = img_1[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
#             img_2 = img_2[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
#             var_list[i] = self.gray_value_temp_var(img_1, img_2)
#         print(var_list)
#         return np.array(var_list)