# Chris Layden
# 31 May 2024
# Functions to calculate the flux from an integrating sphere at a detector

import numpy as np
import scipy

def flux_per_port_area(y, x, d, r):
    '''Integrand for the flux from an integrating sphere at a detector
    
    Parameters
    ----------
    x : float
        The x position of a point on the integrating sphere
    y : float
        The y position of a point on the integrating sphere
    d : float
        Double the radial distance from the center of the detector to the point
    r : float
        The distance from the center of the integrating sphere port to the detector
    
    Returns
    -------
    flux_per_area : float
        The flux per unit area at the detector from the integrating sphere
    '''

    # Calculate the distance from the point on the integrating sphere to the point on the detector
    dist_squared = (r) ** 2 + x ** 2 + (y - d / 2) ** 2
    # Flux needs to be scaled by cosine of the angle between the normal to the sphere and
    # the line to the detector
    cos_theta = r / np.sqrt(r ** 2 + (y - d / 2) ** 2)
    flux_per_area = cos_theta / dist_squared
    return flux_per_area

def sphere_flux(D, d, r):
    '''Calculate the flux of the integrating sphere at a detector point.
    
    Parameters
    ----------
    D : float
        The diameter of the integrating sphere port
    d : float
        Double the radial distance from the center of the detector to the point
        of interest
    r : float
        The distance from the center of the integrating sphere port to the detector
    
    Returns
    -------
    flux : float
        The flux at the detector point. This is not normalized to anything,
        so it's only useful for comparing the flux at different points
    '''
    # Integrate the flux per area over the entire port area
    int_result_0 = scipy.integrate.dblquad(flux_per_port_area, -D/2, D/2,
                                           lambda y: -np.sqrt((D / 2) ** 2 - y ** 2),
                                           lambda y: np.sqrt((D / 2) ** 2 - y ** 2),
                                           args=(d, r))
    flux = int_result_0[0]
    return flux

def max_sphere_nonuniformity(D, d, r):
    '''Calculate the maximum nonuniformity across a detector.
    
    Parameters
    ----------
    D : float
        The diameter of the integrating sphere port
    d : float
        The maximum length of the detector.
        d/2 is the projected radial distance from the center of the integrating
        sphere port to the point of interest.
    r : float
        The axial distance from the integrating sphere port to the detector
    
    Returns
    -------
    rel_flux : float
        The ratio between the flux at the point of interest and the flux at 
        the center of the detector
    '''
    outer_flux = sphere_flux(D, d, r)
    center_flux = sphere_flux(D, 0, r)
    rel_flux = outer_flux / center_flux
    return rel_flux

def sphere_flux_correction(D, rPD, dPD, rSensor, dSensor=0, sensorSize=0):
    '''Find the scaling factor between the flux at the photodiode and at the sensor

    Parameters
    ----------
    D : float
        The diameter of the integrating sphere port
    dPD : float
        The projected radial distance from the center of the integrating sphere
        port to the photodiode
    rPD : float
        The axial distance from the integrating sphere port to the photodiode
    rSensor : float
        The axial distance from the integrating sphere port to the sensor
    dSensor : float
        The projected radial distance from the center of the integrating sphere
        port to the center of the sensor
    sensorSize : float
        The width/height of the sensor (assume sensor is square). We'll average
        the flux over the sensor area if this is nonzero.
    
    Returns
    -------
    correction : float
        The scaling factor between the flux at the photodiode and the flux at the sensor
    '''

    # Find the flux at the photodiode
    pd_flux = sphere_flux(D, dPD * 2, rPD)
    # Find the flux averaged across the sensor
    if sensorSize == 0:
        avg_sensor_flux = sphere_flux(D, dSensor * 2, rSensor)
    else:
        avg_sensor_flux = 0
        for i in range(0, 11):
            for j in range(0, 11):
                subsensor_x = dSensor -sensorSize / 2 + i * sensorSize / 10
                subsensor_y = -sensorSize / 2 + j * sensorSize / 10
                subsensor_d = np.sqrt(subsensor_x ** 2 + subsensor_y ** 2)
                sensor_flux= sphere_flux(D, subsensor_d * 2, rSensor)
                avg_sensor_flux += sensor_flux / 121
    correction = avg_sensor_flux / pd_flux
    return correction

def sphere_prnu(D, xSensor, ySensor, r, sensorSize, numPoints=11):
    '''Calculate the apparent prnu across a sensor due to nonuniform illumination.

    Parameters
    ----------
    D : float
        The diameter of the integrating sphere port
    xSensor : float
        The x position of the sensor center
    ySensor : float
        The y position of the sensor center
    r : float
        The axial distance from the integrating sphere port to the sensor
    sensorSize : float
        The width/height of the sensor (assume sensor is square)

    Returns
    -------
    prnu : float
        The pixel response nonuniformity across the sensor due solely to
        nonuniform illumination, as a fraction (not a percent). Assumes an
        ideal integrating sphere.
    '''
    # Find the flux at different points on the sensor
    flux_vals = np.zeros((numPoints, numPoints))
    for i in range(0, numPoints):
        for j in range(0, numPoints):
            subsensor_x = xSensor - sensorSize / 2 + i * sensorSize / (numPoints - 1)
            subsensor_y = ySensor - sensorSize / 2 + j * sensorSize / (numPoints - 1)
            subsensor_d = np.sqrt(subsensor_x ** 2 + subsensor_y ** 2)
            sensor_flux = sphere_flux(D, subsensor_d * 2, r)
            flux_vals[i, j] = sensor_flux
    prnu = np.std(flux_vals.flatten()) / np.mean(flux_vals)
    return prnu
