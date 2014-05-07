# GENERAL COMMENTARY: I think we want to go in the direction of better
# modularity in the functionality and then a very few, very high-level
# user-facing functions with a lot of flags to direct work flow.

# GENERAL COMMENTARY: I worry about astropy depdendencies and
# ease-of-use for our target audience. This fundamentally needs to be
# usable by people trying to do clean masking.

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Imports
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# Standard imports should always work
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.signal as ssig
import warnings

# Try to pull in bottleneck (a faster implementation of some numpy
# functions) and default to scipy if this fails
try:
    from bottleneck import nanmedian, nanstd
except ImportError:
    from scipy.stats import nanmedian, nanstd

# Astropy - used for convolution and WCS
import astropy.wcs
from astropy.convolution import convolve_fft,convolve

# Spectral cube object from radio-astro-tools
from spectral_cube import SpectralCube,BooleanArrayMask

# Radio beam object from radio-astro-tools
try:
    from radio_beam import Beam
except ImportError:
    warnings.warn("No radio_beam.Beam instances found. Convolution\
     with radio beams will not work")

# Utility functions from this package
from utils import *

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# The Noise Object
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

class Noise:
    """Class used to track the parameters of the noise in a radio data cube.  

    Limitations
    -----------
    Many of the routines in the package rely on the assumption of the 
    data distribution begin centered around zero.

    Right now, a normal distribution is assumed for a large amount of
    the functionality. Extensions to and arbitrary distribution are
    outlined but not implemented.
    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Contents
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # ........................
    # Normal noise description
    # ........................

    # This is a global "representative" noise distribution width
    # (e.g., sigma for the normal distribution) as a scalar
    scale = None   

    # This is a 2d (spatial) map of the width parameter, normalized to
    # the overall noise scaling.
    spatial_norm = None

    # This is a 1d (spectral) array of the width parameter, normalized
    # to the overall noise scaling.
    spectral_norm = None

    # Then the noise a a position (x,y,z) is:
    # scale * spectral_norm[z] * spatial_norm[y,x]

    # This is a two-d mask indicating where we have data to measure the spatial noise
    spatial_footprint = None
    
    # This is a one-d mask indicating where we have data to measure the spectral noise
    spectral_footprint = None

    # .............................
    # General noise description
    # .............................

    # This is a noise distribution function compatible with the
    # scipy.stats package - it will default to a normal distribution
    distribution = None

    # This is the global value of the shape parameters
    distribution_shape = None

    # This is a three dimensional distribution of shape parameters
    distribution_shape_cube = None

    # In this case, the shape parameters can be fed to the linked
    # scipy stats function to achieve matched functionality for any of
    # their implemented probability distributions. This is a work in
    # progress.

    # .............................
    # Associated data cube and beam
    # ............................

    cube = None
    beam = None
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Initialization
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(
        self,
        cube,
        scale=None,
        spatial_norm = None,
        spectral_norm = None,
        beam = None):
        """Construct a new Noise object."""

        if isinstance(cube,SpectralCube):
            self.cube = cube
            self.spatial_footprint = np.any(cube.get_mask_array(),axis=0)
        else:
            warnings.warn("Noise currently requires a SpectralCube instance.")

        if isinstance(beam,Beam):
            self.beam = beam

        # Default to a normal distribution
        self.distribution = ss.norm

        # Fit the data
        if scale is None:
            self.calculate_fit()
            self.scalar_noise() # [1] is the std. of a Gaussian

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Expose the noise in various ways
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def get_scale_cube(self):
        """
        Return a cube of noise width estimates.
        """
        ax0 = np.reshape(self.spectral_norm, 
                         (self.spectral_norm.shape+tuple((1,1))))
        ax12 = np.reshape(self.spatial_norm,
                          tuple((1,))+self.spatial_norm.shape)
        noise = (ax12*ax0)*self.scale
        return noise

    def cube_of_noise(self):
        """
        Generates a data cube of randomly generated noise with properties matching
        the measured distribution.
        """        

        # ISSUE - this currently only uses the beam if there is a
        # spatial noise map. That's not right. You always want to use
        # the beam if you have it.

        if self.spatial_norm is None:
            return self.distribution.rvs(
                *self.distribution_shape,
                size=self.cube.shape)
        else:
            noise = np.random.randn(*self.cube.shape)
            if self.beam is not None:
                self.beam.as_kernel(get_pixel_scales(self.cube.wcs))
                # Iterate convolution over plane (ugh)
                for plane in np.arange(self.cube.shape[0]):
                    noise[plane,:,:] = convolve_fft(noise[plane,:,:],
                        self.beam.as_kernel(get_pixel_scales(self.cube.wcs)),
                        normalize_kernel=True)
            return noise * self.get_scale_cube()

    def snr(self,
            as_prob=False):
        """
        Return a signal-to-noise cube.
        """
        raise NotImplementedError()

        # Holds the signal to noise ratio.
        # SNR[x,y,z] = Data[x,y,z] / (scale*spectral_norm[z]*spatial_norm[x,y])
        # snr = None

    def to_file(self,
                outfile=None):
        """
        Write noise fit results to file.
        """
        raise NotImplementedError()
                
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Generate noise estimate methods
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    def scalar_noise(self):
        """
        Derive the scale of the noise distribution using only the 
        negative values in the array and the MAD()
        """

        # POSSIBLE IMPROVEMENT: Why not have this be either (a) a
        # general thing with multiple methods exposed or (b) just
        # something that gets done as a subset of the 

        negs = self.cube.flattened().value.astype('=f')
        negs = negs[negs<0]
        self.scale = nanmad(np.hstack([negs,-1*negs]), median_value=0.0)
        return

    def calculate_std(
            self,
            scalar=False):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the standard deviation is a rigorous method.
        
        Parameters: 
        scalar : boolean (default False)
           Fit only a single number. Otherwise fit spectral and spatial variations.
        """

        # POSSIBLE IMPROVEMENT - add iterative outlier rejection
        # here.
        
        # Extract the data from the spectral cube object
        data = self.cube.filled_data[:].astype('=f')
        
        # Calculate the overall scale
        self.scale = nanstd(data)
        
        # Return if fed an image and not a cube
        if self.data.ndim == 2 or scalar == True:
            return

        # Calculate the spatial variations after removing the
        # overall scaling
        self.spatial_norm = nanstd(data,axis=0)/self.scale
        
        # Calculate the spectral variations after removing both
        # the overall and spatial variations. Do this by
        # flattening into a two-d array with the two image
        # dimensions stacked together.
        self.spectral_norm = nanstd(
            (data/self.spatial_norm/self.scale).reshape((data.shape[0],
                                                         data.shape[1]*
                                                         data.shape[2])), axis=1)

        return

    def calculate_mad(
            self,
            niter=1,
            spatial_smooth=None,
            spectral_smooth=None,
            spatial_flat=False,
            spectral_flat=False):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the median absolute deviation is a rigorous method.

        Parameters
        ----------
        niter : number
            Number of iterations used in estimating the separate 
            components of the spatial and spectral noise variations.
            Default=1
        spatial_smooth : number
            Number of pixels used in spatially smoothing the normalization for the 
            nosie map (via a 2D median filter)
        spectral_smooth : number
            Number of pixels used in spectral smoothing the normalization for the 
            nosie map (via a 1D median filter)
        spatial_flat : boolean
            Assumes there is no spatial variation in the noise pattern.  
            Default = False
        spectral_flat : boolean
            Assumes no spectral variation in the noise pattern.
            Default=false

        Returns
        -------
        None : updates Noise.spatial_norm, Noise.spectral_norm and Noise.scale 
        in place.
        """

        # POSSIBLE IMPROVEMENT: Separate out the smoothing functions
        # as generic operators on the estimate. They only need to be
        # in here if they are going to be used in the actual estimation
        
        # Access the data in the spectral cube
        data = self.cube._get_filled_data(check_endian=True)

        # Estimate the noise
        self.scalar_noise()

        # Initialize the spatial and spectral variations.
        if self.spatial_norm is None:
            self.spatial_norm = np.ones((data.shape[1],data.shape[2]))
            self.spectral_norm = np.ones((data.shape[0]))

        # ISSUE: Not obvious what this wants to do right now - why
        # iterate if you lose memory of the previous iterations each
        # time? Presumably the idea is some cumulative filtering or
        # outlier rejection? If it's just smoothing, move it out to
        # separate functions to operate on the noise. I'm going to
        # edit this.

        for count in range(niter):
            if not spatial_flat:
                snr = data/self.get_scale_cube()
                self.spatial_norm = nanmad(snr,axis=0)*self.spatial_norm
                if spatial_smooth is not None:
                    self.spatial_norm = ssig.medfilt2d(self.spatial_norm,
                        kernel_size=spatial_smooth)
                    if self.beam is not None:
                        self.spatial_norm = convolve_fft(self.spatial_norm, 
                                                         self.beam.as_kernel(get_pixel_scales(self.cube.wcs)),
                                                         interpolate_nan=True,normalize_kernel=True)
            else:
                self.spatial_norm = np.ones([data.shape[1],data.shape[2]])
            if not spectral_flat:
                snr = data/self.get_scale_cube()
                self.spectral_norm = nanmad(snr.reshape((snr.shape[0],
                                                         snr.shape[1]*
                                                         snr.shape[2])),
                                            axis=1)*self.spectral_norm
                if spectral_smooth is not None:
                    self.spectral_norm = ssig.medfilt(self.spectral_norm,
                        kernel_size=spectral_smooth)
            else:
                self.spectral_norm = np.ones((data.shape[0]))

        # Cleanup - replace the variations with "1" (i.e., revert to
        # the average noise) wherever they end up not defined and
        # blank the noise estimates outside the footprint defined by
        # the valid data.

        self.spectral_norm[np.isnan(self.spectral_norm) | (self.spectral_norm==0)]=1.
        self.spatial_norm[np.isnan(self.spatial_norm) | (self.spatial_norm==0)]=1.
        self.spatial_norm[~self.spatial_footprint]=np.nan
        self.distribution_shape=(0,self.scale)

        return

    def calculate_std(self,niter=1,spatial_smooth=None,spectral_smooth=None):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the median absolute deviation is a rigorous method.
        """

        data = self.cube.filled_data[:].astype('=f')
        self.scale = nanstd(data)
        if self.spatial_norm is None:
            self.spatial_norm = np.ones((data.shape[1],data.shape[2]))
            self.spectral_norm = np.ones((data.shape[0]))
        for count in range(niter):
            scale = self.get_scale_cube()
            snr = data/scale
            self.spatial_norm = nanstd(snr,axis=0)*self.spatial_norm
            if self.beam is not None:
                self.spatial_norm = convolve_fft(self.spatial_norm, 
                    self.beam.as_kernel(get_pixel_scales(self.cube.wcs)),
                    interpolate_nan=True,normalize_kernel=True)
            if spatial_smooth is not None:
                self.spatial_norm = ssig.medfilt2d(self.spatial_norm,
                    kernel_size=spatial_smooth)

            snr = data/self.get_scale_cube()
            self.spectral_norm = nanstd(snr.reshape((snr.shape[0],
                                                     snr.shape[1]*
                                                     snr.shape[2])),
                                        axis=1)*self.spectral_norm
            if spectral_smooth is not None:
                self.spectral_norm = ssig.medfilt(self.spectral_norm,
                    kernel_size=spectral_smooth)
        self.spectral_norm[np.isnan(self.spectral_norm) | (self.spectral_norm==0)]=1.
        self.spatial_norm[np.isnan(self.spatial_norm) | (self.spatial_norm==0)]=1.
        self.spatial_norm[~self.spatial_footprint]=np.nan
        self.distribution_shape=(0,self.scale)    
        return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Parallel generic scipy.stats approach
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def calculate_fit(self):
        """
        Fits the distribution of the data using the scipy.fit
        functionality. This uses the distribution that characterizes
        the noise values.
        """
        self.distribution_shape = self.distribution.fit(\
            self.cube.flattened().value)
        return

    def rolling_shape_fit(
            self,
            boxsize=5,
            use_mad=False):
        """
        Calculates the shape of the distribution of the data in the cube by rolling
        a structuring element(currently a box) over the data and fitting the 
        distribution within the box.  The fit uses the currently defined scipy.stats 
        distribution function, which defaults to a Normal distribution.  

        This updates the Noise.distribution_shape() vector in place

        Parameters
        ----------
        boxsize : integer
            Half-width of the box (total side length is 2*boxsize+1)
        use_mad : boolean (default False)
            Sets the method to estimate the noise using the m.a.d.
            (else it fits the distribution using the scipy method)

        """

        # POSSIBLE IMPROVEMENT: specify number of cells into which to
        # divide the cube (i.e., top-down spec rather than bottom up)

        # POSSIBLE IMPROVEMENT: box size should be beam-aware if beam
        # is present.

        # POSSIBLE IMPROVEMENT: box size should have the possibility
        # of being assymetric in spectral and spatial dimensions.

        # Initialize a new shape map
        shape_map = np.zeros(self.cube.shape+(len(\
            self.distribution_shape),))

        # Get the data from the spectral cube object
        data = self.cube.filled_data[:]

        # Initialize an iterator over the cube
        iterator = np.nditer(data,flags=['multi_index'])
        xoff,yoff,zoff = np.meshgrid(np.arange(-boxsize,boxsize),
                                     np.arange(-boxsize,boxsize),
                                     np.arange(-boxsize,boxsize),
                                     indexing='ij')

        # Iterate over the grid
        while not iterator.finished:
            position = iterator.multi_index
            xmatch = xoff+position[0]
            ymatch = yoff+position[1]
            zmatch = zoff+position[2]
            inarray = (xmatch>=0)&(xmatch<data.shape[0])&\
                      (ymatch>=0)&(ymatch<data.shape[1])&\
                      (zmatch>=0)&(zmatch<data.shape[2])
                        
            if use_mad:                
                shape_map[position[0],
                          position[1],
                          position[2],
                          1] = nanmad(data[xmatch[inarray].ravel(),
                                           ymatch[inarray].ravel(),
                                           zmatch[inarray].ravel()])
            else:
                shape_map[
                    position[0], 
                    position[1], 
                    position[2],:] = self.distribution.fit(\
                                                           data[xmatch[inarray].ravel(),
                                                                ymatch[inarray].ravel(),
                                                                zmatch[inarray].ravel()])
            iterator.iternext()

        # ISSUE: what to do when the grid cell is largely empty?
        # Empties/interpolation an option.

        # POSSIBLE IMPROVEMENT: Some smoothing to get to a continuous
        # variation. Cubic or linear interpolation? Linear smoothing?

        # Place the fit distribution into memory
        self.distribution_shape_map = shape_map

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Manipulate the noise estimates
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def smooth_noise_map(
            self):
        raise NotImplementedError()

    def smooth_noise_spectra(
            self):
        raise NotImplementedError()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Interface with signal
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def mask_out_signal(self,niter=1):
        """
        Sets the mask property of the SpectralCube associated with the noise object
        to exclude the noise through a (iterative) application of the Chauvenet 
        rejection criterion (i.e., mask out all points outside of +/- N sigma of zero).

        Parameters
        ----------
        niter : number
            Number of iterations used in estimating the separate 
            components of the spatial and spectral noise variations.
            Default=1
        """
        for count in range(niter):
            if self.spatial_norm is not None:
                noise = self.get_scale_cube()
                snr = self.cube.filled_data[:]/noise
            else:
                snr = self.cube.filled_data[:]/self.scale
            # Include negatives in the signal mask or not?
            newmask = BooleanArrayMask(np.abs(snr)<sig_n_outliers(self.cube.size),self.cube.wcs)
            self.cube = self.cube.with_mask(newmask)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Visualization methods
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    def plot_histogram(self,normalize=True):
        """
        Makes a plot of the data distribution and the estimated
        parameters of the PDF.

        Parameters
        ----------
        normalize : Normalize the plot by the spatially and 
        spectrally varying noise scale to cast the plot in terms of 
        signal-to-noise ratio. Default: True
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
            data = self.cube.filled_data[:].astype('=f')
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
            plotdata = self.cube.flattened().value            

            plt.hist(plotdata,bins=Nbins,log=True)
            plt.plot(xsamples,binwidth*self.cube.size*
                     self.distribution.pdf(xsamples,*self.distribution_shape),
                     linewidth=4,alpha=0.75)
            plt.xlabel('Data Value')
            plt.ylabel('Number of Points')

    def plot_spectrum(self):
        """
        Makes a plot of the spectral variation of the noise.
        """

        if self.spectral_norm is None:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        channel = np.arange(len(self.spectral_norm))
        plt.plot(channel, self.spectral_norm)

    def plot_map(self):
        """
        Makes a plot of the spatial variation of the noise.
        """

        if self.spatial_norm is None:
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        plt.imshow(channel, self.spatial_norm, origin="lower")

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# EXTERNAL FUNCTIONS STILL INTEGRAL TO NOISE
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

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

def nanmad(data, sigma=True, axis=None, median_value=None):
    """
    Return the median absolute deviation.  Axis functionality adapted
    from https://github.com/keflavich/agpy/blob/master/agpy/mad.py
    """
    if axis>0:
        if median_value is None:
            med = nanmedian(data.swapaxes(0,axis),axis=0)
        else:
            med = median_value
        mad = nanmedian(np.abs(data.swapaxes(0,axis) - med),axis=0)
    else:
        if median_value is None:
            med = nanmedian(data,axis=axis)
        else:
            med = median_value
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
