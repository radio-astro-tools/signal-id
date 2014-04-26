def mad(data, sigma=True, axis=None):
    """
    Return the median absolute deviation.  Axis functionality adapted
    from https://github.com/keflavich/agpy/blob/master/agpy/mad.py
    """
    if axis>0:
        med = np.median(data.swapaxes(0,axis),axis=0)
        mad = np.median(np.abs(data.swapaxes(0,axis) - med),axis=0)
    else:
        med = np.median(data,axis=axis)
        mad = np.median(np.abs(data - med),axis=axis)
    if sigma==False:
        return mad
    else:
        return mad*1.4826

def nanmad(data, sigma=True, axis=None):
    """
    Return the median absolute deviation.  Axis functionality adapted
    from https://github.com/keflavich/agpy/blob/master/agpy/mad.py
    """
    if axis>0:
        med = nanmedian(data.swapaxes(0,axis),axis=0)
        mad = nanmedian(np.abs(data.swapaxes(0,axis) - med),axis=0)
    else:
        med = nanmedian(data,axis=axis)
        mad = nanmedian(np.abs(data - med),axis=axis)
    if not sigma:
        return mad
    else:
        return mad*1.4826

def sigma_rob(data, iterations=1, thresh=3.0):
    """
    Iterative m.a.d. based sigma with positive outlier rejection.
    """
    noise = mad(data)
    for i in range(iterations):
        ind = (data <= thresh*noise).nonzero()
        noise = mad(data[ind])
    return noise

def sig_n_outliers(n_data, n_out=1.0, pos_only=True):
    """
    Return the sigma needed to expect n (default 1) outliers given
    n_data points.
    """
    perc = float(n_out)/float(n_data)
    if pos_only == False:
        perc *= 2.0
    return abs(ss.norm.ppf(perc))

def get_pixel_scales(mywcs):
    # borrowed from @keflavich who borrowed from aplpy
    mywcs = mywcs.sub([astropy.wcs.WCSSUB_CELESTIAL])
    cdelt = np.array(mywcs.wcs.get_cdelt())
    pc = np.array(mywcs.wcs.get_pc())
    # I too like to live dangerously:
    scale = np.array([cdelt[0] * (pc[0,0]**2 + pc[1,0]**2)**0.5,
     cdelt[1] * (pc[0,1]**2 + pc[1,1]**2)**0.5])
    return abs(scale[0])
