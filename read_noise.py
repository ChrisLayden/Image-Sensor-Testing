

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# dir = '/Users/layden/Desktop/11_41_48'
# dir = '/Users/layden/Desktop/13_07_55'
dir = '/Users/layden/Desktop/13_18_13'
files = os.listdir(dir)
img_files = [file for file in files if file.endswith('.fits')]
# Exclude hidden files that SAOImage generates
img_files = [file for file in img_files if not file.startswith('.')]
num_img = len(img_files)
first_img = fits.getdata(dir + '/' + img_files[0]).astype('int')
second_img = fits.getdata(dir + '/' + img_files[1]).astype('int')
diff_img = np.abs(second_img - first_img)
gain = 0.75
read_noise = np.std(diff_img) / np.sqrt(2) * gain
# read_noise = np.mean(diff_img) * 0.75
# plt.hist(diff_img.flatten(), bins=100)
# plt.yscale('log')
# plt.show()
print(read_noise)
print(np.std(first_img) * 0.75)