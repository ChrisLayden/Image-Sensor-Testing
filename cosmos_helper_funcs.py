# Chris Layden and Jill Juneau
# Functions to read in COSMOS image data, which come in fits files.

import os
import warnings
import re
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

def find_keywords(keywords, compare_word):
    # Find the indices of the keywords in a FITS header that contain the compare_word
    indices = []
    for i, keyword in enumerate(keywords):
        if re.search(compare_word, keyword, re.IGNORECASE):
            indices.append(i)
    return indices

# Keywords to find for COSMOS data
cosmos_keywords = ['DATE', 'DATE-OBS', 'BITPIX', 'SENSOR INFORMATION SENSORNAME', 'ACTIVEAREA HEIGHT', 'ACTIVEAREA WIDTH', 'ACTIVEAREA BOTTOMMARGIN',
                'ACTIVEAREA LEFTMARGIN', 'ACTIVEAREA RIGHTMARGIN', 'PIXEL WIDTH', 'ADC BITDEPTH', 'ANALOGGAIN',
                'ADC QUALITY', 'CORRECTPIXEL', 'FRAMERATE', 'FRAMESTOSTORE', 'BASEFILENAME', 'TEMPERATURE READING',
                'READOUTCONTROL MODE', 'SHUTTERTIMING EXPOSURETIME']
cosmos_fieldnames = ['date', 'time_stamp_acq', 'bit_per_pix', 'sensor_info', 'active_height', 'active_width', 'bottom_margin', 'leftmargin', 'rightmargin',
            'pixel_size_um', 'adc_bit', 'analog_gain', 'adc_speed', 'corrected', 'fps', 'frame_num', 'filename',
            'cam_temp_Cel', 'shutter', 'exposure_ms']

def get_stacks(user_dir, keywords=cosmos_keywords, fieldnames=cosmos_fieldnames,
               get_mean_img=False, get_var_img=False, num_imgs=None):
    '''Read in a folder of FITS files. Extract data and relevant header keywords.
    
    Parameters
    ----------
    dir : str
        The directory containing the FITS files
    keywords : list (default=[])
        A list of keywords to search for in the FITS headers
    fieldnames : list (default=[])
        A list of fieldnames to store the extracted keywords in each stack dictionary
    get_mean_img : bool (default=False)
        Whether to extract the mean image from each stack
    get_var_img : bool (default=False)
        Whether to extract the variance image from each stack. This image gives the variance in the value
        read by each pixel across the stack of images.
    num_images : int (default=None)
        The number of images to return in the imagestack, for memory purposes. If None, return all images.
    
    Returns
    -------
    stacks : list
        A list of dictionaries containing the images and relevant information from the FITS headers.
        Each dictionary is a stack of images from a single FITS file.
    '''

    # Check to make sure that folder actually exists. Warn user if it doesn't.
    if not os.path.isdir(user_dir):
        error_message = f'Error: The following folder does not exist:\n{user_dir}'
        warnings.warn(error_message)
        raise FileNotFoundError(error_message)

    # Initialize the list of stacks
    stacks = []
    the_files = [file for file in os.listdir(user_dir) if file.endswith('.fits')]

    for file in the_files:
        base_file_name = os.path.basename(file)
        full_file_name = os.path.join(user_dir, base_file_name)
        print(f'Now reading {full_file_name}')
        if not full_file_name.endswith('.fits'):
            print(f'File {full_file_name} is not a FITS file. Skipping.')
            continue
        
        # Read FITS file
        hdul = fits.open(full_file_name)
        info = hdul.info(False)
        primary_header = hdul[0].header
        imagestack = hdul[0].data.astype(int)
        if get_mean_img:
            mean_img = np.mean(imagestack, axis=0)
        else:
            mean_img = None
        if get_var_img:
            var_img = np.var(imagestack, axis=0, ddof=1)
        else:
            var_img = None
        if num_imgs is not None:
            if num_imgs == 0:
                imagestack = None
            else:
                imagestack = imagestack[:num_imgs]
        

        file_data = {'baseFileName': base_file_name, 'fullFileName': full_file_name,
                     'info': info, 'imagestack': imagestack, 'mean_img': mean_img, 'var_img': var_img}        
        for keyword, fieldname in zip(keywords, fieldnames):
            primary_keywords = list(primary_header.keys())
            primary_indices = find_keywords(primary_keywords, keyword)
            if primary_indices:
                index = primary_indices[0]
                extracted_text = primary_header[index]
                file_data[fieldname] = extracted_text
            else:
                file_data[fieldname] = None
        stacks.append(file_data)
        hdul.close()

    return stacks

def label_plot(filedata):
    label_string = 'MODE: ' + filedata['adc_speed'] + ' '
    if filedata['analog_gain'] is not None:
        label_string += filedata['analog_gain'] + 'Gain '
    label_string += filedata['shutter']
    if filedata['corrected'] == 'True':
        label_string += ' CorrectedPixels '
    else:
        label_string += ' RawPixels '
    label_string += 'EXP: ' + filedata['exposure_ms'] + ' ms' + '\n'
    label_string += 'TIMESTAMP: ' + filedata['time_stamp_acq'] + ' CAMERA TEMP: ' + filedata['cam_temp_Cel'] + ' C'
    plt.suptitle(label_string, y=1.05)

def get_mean_images(stacks):
    mean_img_dict = {}
    for stack in stacks:
        # Label the readout mode
        if stack['adc_speed'] == 'HighSpeed':
            mode = 'HS_'
        elif stack['adc_speed'] == 'LowNoise':
            mode = 'LS_'
        elif stack['adc_speed'] == 'HighDynamicRange':
            mode = 'HDR_'
        # Label the gain mode
        if stack['analog_gain'] == 'High':
            mode += 'HG_'
        elif stack['analog_gain'] == 'Low':
            mode += 'LG_'
        # Label the shutter mode
        if stack['shutter'] == 'RollingShutter':
            mode += 'RS_'
        elif stack['shutter'] == 'GlobalShutter':
            mode += 'GS_'
        exp_time = float(stack['exposure_ms'])
        exp_time_str = str(int(1000 * exp_time)) + 'us'
        label = mode + exp_time_str
        # Check if the label is already in the dictionary. Skip if so.
        if label in mean_img_dict.keys():
            continue
        # Take the mean image, skipping the first image (some weirdness with all first images
        # appearing brighter)
        mean_img = np.mean(stack['imagestack'][1:-1], axis=0)
        mean_img_dict[label] = mean_img
    print('Mean images extracted with labels ' + str(mean_img_dict.keys()))
    return mean_img_dict

# Conversion gain values from the datasheet and X-ray data. This is in e-/ADU
cosmos_gain_dict = {
    'DSgain_HSHGRS': 0.95,
    'DSgain_HSHGGS': 0.95,
    'DSgain_HSLGRS': 8.34,
    'DSgain_HSLGGS': 8.34,
    'DSgain_LSLGRS': 2.66,
    'DSgain_LSLGGS': 2.66,
    'DSgain_HDRRS': 0.4,
    'XRgain_HSHGRS': 1.024,
    'XRgain_HSHGGS': 1.001,
    'XRgain_HSLGRS': 7.847,
    'XRgain_HSLGGS': 7.819,
    'XRgain_LSLGRS': 2.095,
    'XRgain_LSLGGS': 1.958,
}

if __name__ == '__main__':
    User_myFolder = '/Users/layden/Library/CloudStorage/Box-Box/Scientific CMOS - MKI ONLY (might contain EAR; ITAR)/Teledyne_COSMOS/Analysis Images/Defect Pixels/LS_LG_RS'
    stacks = get_stacks(User_myFolder)
    mean_img_dict = get_mean_images(stacks)

