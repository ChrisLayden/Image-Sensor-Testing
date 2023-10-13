import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import datetime
import pandas as pd

image_dir = '/Users/layden/Documents/TESS/QE'
files = os.listdir(image_dir)
img_files = [file for file in files if file.endswith('.fits')]
img_files = [file for file in files if file.startswith('Filter2')]
# Exclude hidden files that SAOImage generates
img_files = [file for file in img_files if not file.startswith('.')]
num_img = len(img_files)
first_img = fits.getdata(image_dir + '/' + img_files[0]).astype('int')
second_img = fits.getdata(image_dir + '/' + img_files[1]).astype('int')
print(np.mean(first_img), np.mean(second_img))

year = 1900
month = 1
day = 1
start_time = datetime.datetime(year=year, month=month, day=day, hour=12, minute=36, second=0, microsecond=1)
exposure_time = 60
end_time = start_time + datetime.timedelta(seconds=exposure_time)
# Load csv using pandas. First column is a time.
df = pd.read_csv('/Users/layden/Documents/TESS/QE/qe_visible.csv')
df.columns = ['time', 'ThorLabs PD', 'Photodetector']
df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S.%f'))

# Plot PD and photodetector reading vs time to check timing of light/dark images
fig, ax1 = plt.subplots()
ax1.set_xlabel('Time')
ax1.set_ylabel('ThorLabs PD (W)')
ax1.scatter(df['time'], df['ThorLabs PD'], s=1)
ax2 = ax1.twinx()
ax2.set_ylabel('Photodetector')
ax2.scatter(df['time'], df['Photodetector'], s=1, color='red')
fig.tight_layout()
plt.show()

# Get entries that are greater than start_time and less than end_time
df = df[(df['time'] > start_time) & (df['time'] < end_time)]
print(df['ThorLabs PD'].mean())