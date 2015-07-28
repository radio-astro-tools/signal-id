
'''
Utility functions used in signal_id
'''

# Try to pull in bottleneck (a faster implementation of some numpy
# functions) and default to scipy if this fails
try:
    from bottleneck import nanmedian, nanstd
except ImportError:
    from scipy.stats import nanmedian, nanstd

import numpy as np
import scipy.stats as ss


def mad(data, sigma=True, axis=None,
        force=False, medval=0.0):
    """
    Return the median absolute deviation (or the absolute deviation
    about a fixed value - default zero - if force is set to True). By
    default returns the equivalent sigma. Axis functionality adapted
    from https://github.com/keflavich/agpy/blob/master/agpy/mad.py
    Flips nans to True (default false) to use with nans.

    Parameters
    ----------
    data : np.ndarray
        Data set.
    sigma : bool, optional
        Enables std estimation from MAD.
    axis : {int, None}, optional
        Axis to evaluate MAD along.
    force : bool, optional
        Force the median to be a given value.
    medval : float, optional
        Forced median value.

    Returns
    -------
    mad : float
        MAD estimation. If sigma is True, MAD*1.4826 is returned.
    """
    # Check for nans in the data
    nans = False
    if np.isnan(data).any():
        nans = True

    if axis > 0:
        if force:
            med = medval
        else:
            if nans:
                med = nanmedian(data.swapaxes(0, axis), axis=0)
            else:
                med = np.median(data.swapaxes(0, axis), axis=0)
        if nans:
            mad = nanmedian(np.abs(data.swapaxes(0, axis) - med), axis=0)
        else:
            mad = np.median(np.abs(data.swapaxes(0, axis) - med), axis=0)
    else:
        if force:
            med = medval
        else:
            if nans:
                med = nanmedian(data, axis=axis)
            else:
                med = np.median(data, axis=axis)
        if nans:
            mad = nanmedian(np.abs(data - med), axis=axis)
        else:
            mad = np.median(np.abs(data - med), axis=axis)

    if not sigma:
        return mad
    else:
        return mad*1.4826


def std(data, axis=None):
    if axis > 0:
        return nanstd(data.swapaxes(0, axis), axis=0)
    else:
        return nanstd(data, axis=axis)

# COMMENTARY: Deprecate or incorporate (axis functionality would be
# ideal)


def sigma_rob(data, iterations=1, thresh=3.0, function=mad,
              function_kwargs={}):
    """
    Iterative m.a.d. based sigma with positive outlier rejection.
    """
    noise = function(data, **function_kwargs)
    for i in range(iterations):
        ind = (data <= thresh*noise).nonzero()
        noise = function(data[ind], **function_kwargs)
    return noise


def sig_n_outliers(n_data, n_out=1.0, pos_only=True):
    """
    Return the sigma needed to expect n (default 1) outliers given
    n_data points.
    """
    perc = float(n_out)/float(n_data)
    if pos_only is False:
        perc *= 2.0
    return abs(ss.norm.ppf(perc))


def get_pixel_scales(mywcs):
    """Extract a pixel scale (this assumes square pixels) from a wcs.
    """
    # borrowed from @keflavich who borrowed from aplpy

    # THIS NEEDS HELP!

    pix00 = mywcs.wcs_pix2world(0, 0, 0, 1)
    pix10 = mywcs.wcs_pix2world(1, 0, 0, 1)
    pix01 = mywcs.wcs_pix2world(0, 1, 0, 1)

    scale1 = ((pix00[0] - pix01[0])**2*np.cos(np.pi/180.*pix00[1])**2 +
             (pix00[1] - pix01[1])**2)**0.5
    scale2 = ((pix00[0] - pix10[0])**2*np.cos(np.pi/180.*pix00[1])**2 +
             (pix00[1] - pix10[1])**2)**0.5

    if abs((scale1 - scale2)/scale1) > 0.1:
        print "Pixels may not be square!"

    return scale1

    #mywcs = mywcs.sub([astropy.wcs.WCSSUB_CELESTIAL])
    #cdelt = np.array(mywcs.wcs.get_cdelt())
    #pc = np.array(mywcs.wcs.get_pc())
    # I too like to live dangerously:
    #scale = np.array([cdelt[0] * (pc[0,0]**2 + pc[1,0]**2)**0.5,
    # cdelt[1] * (pc[0,1]**2 + pc[1,1]**2)**0.5])
    #return abs(scale[0])


def get_celestial_axes(wcs):
    '''
    Return which axis or axes are celestial.
    '''

    is_celestial = []
    axis_types = wcs.get_axis_types()

    for ax in range(wcs.naxis):
        if axis_types[ax]['coordinate_type'] == "celestial":
            is_celestial.append(ax)

    return is_celestial