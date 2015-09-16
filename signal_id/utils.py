
'''
Utility functions used in signal_id
'''

# Try to pull in bottleneck (a faster implementation of some numpy
# functions) and default to scipy if this fails
try:
    from bottleneck import nanmedian, nanstd
except ImportError:
    from scipy.stats import nanmedian, nanstd

import warnings
import numpy as np
import scipy.stats as ss
import scipy.ndimage as nd

import astropy.convolution as conv


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


def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False):
    """

    Remove connected components smaller than the specified size.
    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest. If the array
        type is int, it is assumed that it contains already-labeled objects.
        The ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable connected component size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    in_place : bool, optional (default: False)
        If `True`, remove the connected components in the input array itself.
        Otherwise, make a copy.
    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.
    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.
    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]], dtype=bool)
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]], dtype=bool)
    >>> d = morphology.remove_small_objects(a, 6, in_place=True)
    >>> d is a
    True


    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    """
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)

    if in_place:
        out = ar
    else:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = nd.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        nd.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2:
        warnings.warn("Only one label was provided to `remove_small_objects`. "
                      "Did you mean to use a boolean array?")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
