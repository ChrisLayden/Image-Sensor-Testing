import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

# Assesses the linearity of an optical sensor

class LinearityMeasurement(object):

    def __init__(self, dir, reg=[0, int(1e6), 0, int(1e6)]):
        '''Initialize a linearity measurement object
        
        Parameters
        ----------
        dir : str
            The directory containing subdirectories of images taken at different exposure times
        '''
        self.dir = dir
        self.sub_dirs = os.listdir(dir)
        self.exposure_times = [float(sub_dir.split(' ')[0]) for sub_dir in self.sub_dirs]
        self.num_times = len(self.exposure_times)
        one_image = self.get_one_img(self.sub_dirs[0])
        (self.M, self.N) = one_image.shape
        reg[1] = min(reg[1], self.M)
        reg[3] = min(reg[3], self.N)
        self.reg = reg
        self.mu_list = self.get_mu_list()
        self.mu_list_dark = self.get_mu_list(dark=True)
        self.var_list = self.get_var_list()
        self.var_list_dark = self.get_var_list(dark=True)
        

    def gray_value_mean(self, image_1, image_2):
        '''Returns the mean gray value of the two images'''
        return (np.mean(image_1) + np.mean(image_2)) / 2

    def gray_value_temp_var(self, image_1, image_2):
        '''Returns the temporal variance of the gray value of the two images'''
        return np.mean((image_1 - image_2) ** 2) / 2

    def get_one_img(self, sub_dir, dark=False, img_num=0):
        '''Returns one image from a subdirectory'''
        if dark:
            dir_str = "Dark"
        else:
            dir_str = "Light"
        img_files = os.listdir(self.dir + '/' + sub_dir + '/' + dir_str)
        img_files = [file for file in img_files if file.endswith('.fits')]
        img = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[img_num])
        return img

    def get_mu_list(self, dark=False):
        '''Returns a list of the mean gray values of the images in each subdirectory'''
        if dark:
            dir_str = "Dark"
        else:
            dir_str = "Light"
        mu_list = np.zeros(self.num_times)
        for i, sub_dir in enumerate(self.sub_dirs):
            img_files = os.listdir(self.dir + '/' + sub_dir + '/' + dir_str)
            img_files = [file for file in img_files if file.endswith('.fits')]
            img_1 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[0])
            img_2 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[1])
            img_1 = img_1[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
            img_2 = img_2[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
            mu_list[i] = self.gray_value_mean(img_1, img_2)
        return np.array(mu_list)

    def get_var_list(self, dark=False):
        '''Returns a list of the temporal variance of gray values of the images in each subdirectory'''
        if dark:
            dir_str = "Dark"
        else:
            dir_str = "Light"
        var_list = np.zeros(self.num_times)
        for i, sub_dir in enumerate(self.sub_dirs):
            print(sub_dir)
            img_files = os.listdir(self.dir + '/' + sub_dir + '/' + dir_str)
            img_files = [file for file in img_files if file.endswith('.fits')]
            img_1 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[0])
            img_2 = fits.getdata(self.dir + '/' + sub_dir + '/' + dir_str + '/' + img_files[1])
            img_1 = img_1[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
            img_2 = img_2[self.reg[0]:self.reg[1], self.reg[2]:self.reg[3]]
            var_list[i] = self.gray_value_temp_var(img_1, img_2)
        print(var_list)
        return np.array(var_list)


# imx571_dir = '/Volumes/KINGSTON/ASI2600Images/2023-07-24'
# imx571 = LinearityMeasurement(imx571_dir, reg=[3000, 4000, 0, 500])
# x = imx571.mu_list - imx571.mu_list_dark
# y = imx571.var_list - imx571.var_list_dark
# plt.scatter(x,y)
# plt.show()
# print(y/x)

imx571_dir = '/Volumes/KINGSTON/ASI2600Images/IMX571_Data/'
imx571 = LinearityMeasurement(imx571_dir)
x = imx571.mu_list - imx571.mu_list_dark
y = imx571.var_list - imx571.var_list_dark
plt.scatter(x,y)
plt.show()
print(y/x)

# gauss_rand = np.random.normal(loc=10, scale=5, size=(1000,1000))
# test_img_1 = np.random.poisson(lam=100, size=(1000, 1000)) + gauss_rand
# test_img_2 = np.random.poisson(lam=100, size=(1000, 1000)) + gauss_rand
# print(np.var(test_img_1))
# print(np.mean((test_img_1 - test_img_2) ** 2) / 2)

