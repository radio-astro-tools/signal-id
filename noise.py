# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Imports
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.signal as ssig


# Try to pull in bottleneck and fail over to scipy
try:
    from bottleneck import nanmedian, nanstd
except ImportError:
    from scipy.stats import nanmedian, nanstd

from spectral_cube.spectral_cube import SpectralCubeMask,SpectralCube


# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# The Noise Object
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

class Noise:
    """ 
    ...
    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Contents
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # This is a global "representative" noise distribution width as a scalar
    scale = None   

    # This is a spatial map of the width parameter
    spatial_norm = None

    # This is the 1D spectral normalization constructed so
    # that spectral_norm[channel] * map[x,y] = noise width at
    # this point in cube

    spectral_norm = None

    # Noise distribution comaptible with scipy
    distribution = None
    distribution_shape = None

    # Map of arbitrary shapes for the distribution. 
    distribution_shape_map = None
    cube = None

    # Holds the signal to noise ratio.
    # SNR[x,y,z] = Data[x,y,z] / (scale*spectral_norm[z]*spatial_norm[x,y])
    # snr = None

    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Initialization
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(
        self,
        cube,
        scale=None,
        spatial_norm = None,
        spectral_norm = None):
        """Construct a new Noise object."""

        self.cube = cube
        self.distribution = ss.norm
        if scale is None:
            self.calculate_fit()
            self.scale=self.distribution_shape[1] # [1] is the std. of a Gaussian

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Generate noise estimate methods
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    
        
    def calculate_fit(self):
        """
        Fits the distribution of the data using the scipy.fit
        functionality.  This uses the distribution that characterizes
        the noise values.
        """
        self.distribution_shape = self.distribution.fit(\
            self.cube.flattened())
        return


    def calculate_naive(self):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the standard deviation is a rigorous method.
        """

        swapa = self.cube.get_filled_data().astype('=f')
        
        self.scale = nanstd(swapa)
        self.spatial_norm = nanstd(swapa,axis=0)/self.scale
        self.spectral_norm = nanstd(swapa.reshape((swapa.shape[0],
                                                   swapa.shape[1]*
                                                   swapa.shape[2])),
                                    axis=1)/self.scale
        return

    def calculate_mad(self,niter=1,spatial_smooth=None,spectral_smooth=None):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the median absolute deviation is a rigorous method.
        """

        data = self.cube.get_filled_data().astype('=f')
        self.scale = nanmad(data)
        if self.spatial_norm is None:
            self.spatial_norm = np.ones((data.shape[1],data.shape[2]))
            self.spectral_norm = np.ones((data.shape[0]))
        for count in range(niter):
            scale = self.get_scale_cube()
            snr = data/scale
            self.spatial_norm = nanmad(snr,axis=0)
            if spatial_smooth is not None:
                self.spatial_norm = ssig.medfilt2d(self.spatial_norm,kernel_size=spatial_smooth)

            snr = data/self.get_scale_cube()
            self.spectral_norm = nanmad(snr.reshape((snr.shape[0],
                                                     snr.shape[1]*
                                                     snr.shape[2])),
                                        axis=1)
            if spectral_smooth is not None:
                self.spectral_norm = ssig.medfilt(self.spectral_norm,kernel_size=spectral_smooth)

        self.distribution_shape=(0,self.scale)    
        return


    def calculate_std(self,niter=1,spatial_smooth=None,spectral_smooth=None):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the median absolute deviation is a rigorous method.
        """

        data = self.cube.get_filled_data().astype('=f')
        self.scale = nanstd(data)
        if self.spatial_norm is None:
            self.spatial_norm = np.ones((data.shape[1],data.shape[2]))
            self.spectral_norm = np.ones((data.shape[0]))
        for count in range(niter):
            scale = self.get_scale_cube()
            snr = data/scale
            self.spatial_norm = nanstd(snr,axis=0)
            if spatial_smooth is not None:
                self.spatial_norm = ssig.medfilt2d(self.spatial_norm,kernel_size=spatial_smooth)

            snr = data/self.get_scale_cube()
            self.spectral_norm = nanstd(snr.reshape((snr.shape[0],
                                                     snr.shape[1]*
                                                     snr.shape[2])),
                                        axis=1)
            if spectral_smooth is not None:
                self.spectral_norm = ssig.medfilt(self.spectral_norm,kernel_size=spectral_smooth)

        self.distribution_shape=(0,self.scale)    
        return

    def rolling_shape_fit(self,
                            boxsize=5):

        shape_map = np.zeros(self.cube.shape+(len(\
            self.distribution_shape),))
        data = self.cube.get_filled_data()
        iterator = np.nditer(data,flags=['multi_index'])
        xoff,yoff,zoff = np.meshgrid(np.arange(-boxsize,boxsize),
                                     np.arange(-boxsize,boxsize),
                                     np.arange(-boxsize,boxsize),
                                     indexing='ij')
        while not iterator.finished:
            position = iterator.multi_index
            xmatch = xoff+position[0]
            ymatch = yoff+position[1]
            zmatch = zoff+position[2]
            inarray = (xmatch>=0)&(xmatch<data.shape[0])&\
                      (ymatch>=0)&(ymatch<data.shape[1])&\
                      (zmatch>=0)&(zmatch<data.shape[2])
                        
            shape_map[position[0],
                      position[1],
                      position[2],:] = self.distribution.fit(\
                                    data[xmatch[inarray].ravel(),
                                         ymatch[inarray].ravel(),
                                         zmatch[inarray].ravel()])
            iterator.iternext()

        self.distribution_shape_map = shape_map

    def rolling_shape_mad(self,
                            boxsize=5):

        shape_map = np.zeros(self.cube.shape+(len(\
            self.distribution_shape),))
        data = self.cube.get_filled_data()
        iterator = np.nditer(data,flags=['multi_index'])
        xoff,yoff,zoff = np.meshgrid(np.arange(-boxsize,boxsize),
                                     np.arange(-boxsize,boxsize),
                                     np.arange(-boxsize,boxsize),
                                     indexing='ij')

        while not iterator.finished:
            position = iterator.multi_index
            xmatch = xoff+position[0]
            ymatch = yoff+position[1]
            zmatch = zoff+position[2]
            inarray = (xmatch>=0)&(xmatch<data.shape[0])&\
                      (ymatch>=0)&(ymatch<data.shape[1])&\
                      (zmatch>=0)&(zmatch<data.shape[2])

            shape_map[position[0],
                      position[1],
                      position[2],
                      1] = nanmad(data[xmatch[inarray].ravel(),
                                       ymatch[inarray].ravel(),
                                       zmatch[inarray].ravel()])
            iterator.iternext()

        self.distribution_shape_map = shape_map

    def mask_out_signal(self,niter=1):
        for count in range(niter):
            if self.spatial_norm is not None:
                noise = self.get_scale_cube()
                snr = self.cube.get_filled_data()/noise
            else:
                snr = self.cube.get_filled_data()/self.scale
            newmask = SpectralCubeMask(np.abs(snr)<sig_n_outliers(self.cube.size),self.cube.wcs)
            self.cube = self.cube.apply_mask(newmask)

    def get_scale_cube(self):
        ax0 = np.reshape(self.spectral_norm,(self.spectral_norm.shape+tuple((1,1))))
        ax12 = np.reshape(self.spatial_norm,tuple((1,))+self.spatial_norm.shape)
        noise = (ax12*ax0)*self.scale
        return noise

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Visualization and analysis methods
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    


    def cube_of_noise(self):
        """
        Generates a matched data set of pure noise with properties matching
        the stated distribution.
        """
        return self.distribution.rvs(*self.distribution_shape,
                                     size=self.cube.shape)

    def plot_noise(self,normalize=True):
        """
        Makes a plot of the data distribution and the estimated
        parameters of the PDF.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if normalize:
            xmin = self.distribution.ppf(1./self.cube.size,0,1)
            xmax = self.distribution.ppf(1-1./self.cube.size,0,1)
            xsamples = np.linspace(xmin,xmax,100)
            Nbins = np.min([int(np.sqrt(self.cube.size)),100])
            binwidth = (xmax-xmin)/Nbins
            plt.xlim(xmin,xmax)
            data = self.cube.get_filled_data().astype('=f')
            scale = self.get_scale_cube()
            snr = data/scale
            plotdata = snr[np.isfinite(snr)].ravel()
            plt.hist(plotdata,bins=Nbins,log=True)
            plt.plot(xsamples,binwidth*self.cube.size*
                     self.distribution.pdf(xsamples,0,1),
                     linewidth=4,alpha=0.75)
            plt.xlabel('Normalized Signal-to-Noise Ratio')
            plt.ylabel('Number of Points')


        else:
            xmin = self.distribution.ppf(1./self.cube.size,*
                                         self.distribution_shape)
            xmax = self.distribution.ppf(1-1./self.cube.size,*
                                         self.distribution_shape)
            xsamples = np.linspace(xmin,xmax,100)
            Nbins = np.min([int(np.sqrt(self.cube.size)),100])
            binwidth = (xmax-xmin)/Nbins
            plt.xlim(xmin,xmax)
            plotdata = self.cube.flattened()            

            plt.hist(plotdata,bins=Nbins,log=True)
            plt.plot(xsamples,binwidth*self.cube.size*
                     self.distribution.pdf(xsamples,*self.distribution_shape),
                     linewidth=4,alpha=0.75)
            plt.xlabel('Data Value')
            plt.ylabel('Number of Points')

        
    def calc_1d(
        self, 
        method="ROBUST",
        timer=False,
        verbose=False,
        showlog=True):
        """
        Calculate a single noise estimate for a data set.
        """

        # .............................................................
        # Time the operation if requested.
        # .............................................................

        if timer:
            start=time.time()

        # .............................................................
        # Fit the 1-d noise distribution
        # .............................................................

        # Identify which values to fit - the data are valid and there
        # is no signal associated with them.

        use = self.data.valid
        if self.signal != None:
            use *= (self.signal.data == False)
        if self.data.signal != None:
            use *= (self.data.signal.data == False)
        
        # Call the external noise fitter
        self.scale = est_noise_1d(
            self.data.data[use],
            method=method)

        if timer:
            stop=time.time()
            print "Fitting the noise (1d) took ", stop-start


# ------------------------------------------------------------
# STASTICS HELPER PROCEDURES
# ------------------------------------------------------------
# These may be of general use and so are not part of the noise
# class. Instead they can be called piecemeal.

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
    if sigma==False:
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

# ------------------------------------------------------------
# Commentary
# ------------------------------------------------------------

# In theory the masked array class inside of numpy should expedite
# handling of blanked data (similarly the scipy.stats.nanmedian or
# nanstd functions). However, the masked array median operator seems
# to be either broken or infeasibly slow. This forces us into loops,
# which (shockingly) work out to be the fastest of the ways I have
# tried, but are still far from good.
    
