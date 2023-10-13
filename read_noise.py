

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# def read_noise(file_1, file_2, gain):
#     '''Calculate the read noise of a camera from two zero-exposure images.
    
#     Parameters
#     ----------
#     file_1 : str
#     gain : float
#         The gain of the camera, in ADU/e-.
        
#     Returns
#     -------
#     read_noise : float
#         The read noise of the camera, in e-/pix.
#     '''

#     diff_img = np.abs(img_2 - img_1)
#     read_noise = np.mean(diff_img) / np.sqrt(2) * gain
#     print(read_noise)

# dir = '/Users/layden/Desktop/11_41_48'
# dir = '/Users/layden/Desktop/13_07_55'
dir = '/Users/layden/Desktop/read_noise'
files = os.listdir(dir)
img_files = [file for file in files if file.endswith('.fits')]
# Exclude hidden files that SAOImage generates
img_files = [file for file in img_files if not file.startswith('.')]
num_img = len(img_files)
first_img = fits.getdata(dir + '/' + img_files[0]).astype('int')
second_img = fits.getdata(dir + '/' + img_files[1]).astype('int')
for i in range(2):
    img = fits.getdata(dir + '/' + img_files[i]).astype('int') / 2
    if i == 0:
        avg_img = img
    else:
        avg_img += img

for i in range(2):
    if i == 0:
        std_img = (fits.getdata(dir + '/' + img_files[i]).astype('int') - avg_img) ** 2
    else:
        std_img += (fits.getdata(dir + '/' + img_files[i]).astype('int') - avg_img) ** 2
print(avg_img)
# std_img = np.sqrt(std_img / 2)
# print(std_img.mean() * 0.75)

diff_img = np.abs(second_img - first_img)
print(diff_img ** 2)
# gain = 0.75
# read_noise = np.mean(diff_img) / np.sqrt(2) * gain
# print(read_noise)
# print(read_noise)
# # read_noise = np.mean(diff_img) * 0.75
# # plt.hist(diff_img.flatten(), bins=100)
# # plt.yscale('log')
# # plt.show()
# print(read_noise)
# # print(np.std(first_img) * 0.75)