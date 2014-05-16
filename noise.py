# GENERAL COMMENTARY: I worry about astropy depdendencies and
# ease-of-use for our target audience. This fundamentally needs to be
# usable by people trying to do clean masking.

# GENERAL COMMENTARY: Next high nail seems like signal masking.

# ADD TIME AND COMMENTARY/VERBOSE

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
from spectral_cube import BooleanArrayMask,SpectralCube

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

    # This is a two-d mask indicating where we have data to measure
    # the spatial noise
    spatial_footprint = None
    
    # This is a one-d mask indicating where we have data to measure
    # the spectral noise
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
        
        # SUGGESTION: calculate on initialization?

        # Fit the data
        if scale is None:
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
        Generates a data cube of randomly generated noise with properties
        matching the measured distribution.
        """        

        noise = np.random.randn(*self.cube.shape)
        if self.beam is not None:
            beam = self.beam.as_kernel(get_pixel_scales(self.cube.wcs))
            # Iterate convolution over plane (ugh)
            for plane in np.arange(self.cube.shape[0]):
                noise[plane,:,:] = convolve_fft(noise[plane,:,:],
                                                beam,
                                                normalize_kernel=True)
        
        return noise * self.get_scale_cube()
            
        # Eventually, we want to use self.distribution.rvs for arbitrary distributions

    def snr(self,
            as_prob=False):
        """
        Return a signal-to-noise cube.
        """

        return self.data/self.get_scale_cube()

        if as_prob:
            raise NotImplementedError()

    def from_file(self,
                infile=None):
        """
        Read noise fit results from file.
        """

        # Parse a cube as a scale cube, rading twod image and spectra

        raise NotImplementedError()

    def to_file(self,
                outfile=None):
        """
        Write noise fit results to file.
        """

        # Dump the scale cube?

        raise NotImplementedError()
                
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Generate noise estimates
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    def calculate_scale(
            self,
            method="MAD"):
        """
        Derive the scale of the noise distribution using only the 
        negative values in the array and the MAD()
        """
        if method == "MAD":
            negs = self.cube.flattened().value.astype('=f')
            negs = negs[negs<0]
            self.scale = mad(
                negs, 
                force=True, 
                medval=0.0,
                nans=True)
        if method == "STD":
            data = self.cube.filled_data[:].astype('=f')        
            self.scale = nanstd(data)

        self.distribution_shape=(0,self.scale)
        return

    def calculate_spatial(
            self,
            method="MAD",
            cumul=False):
        """
        Calculate spatial variations.
        """

        # This approach is not defined for images, though other
        # approaches could work
        if self.data.ndim == 2:
            return

        # There are two approaches: iterate on top of the current
        # noise estimate or generate a new estimate
        if cumul:
            snr = data/self.get_scale_cube()
        else:
            snr = data/self.scale

        # Switch estimate on methodology
        if method == "MAD":
            estimate = mad(snr,axis=0,nans=True)

        if method == "STD":
            estimate = nanstd(snr,axis=0)

        # If we are doing a cumulative calculation then append the estimate
        if cumul:
            self.spatial_norm = estimate*self.spatial_norm
        else:
            self.spatial_norm = estimate

        # Enforce the footprint of viable data and set lingering not-a-numbers to 1
        self.spatial_norm[np.isnan(self.spatial_norm) | (self.spatial_norm==0)]=1.
        self.spatial_norm[~self.spatial_footprint]=np.nan
        return
        
    def calculate_spectral(
            self,
            method="MAD",
            cumul=False):
        """
        Calculate spectral variations.
        """

        # Not defined for images
        if self.data.ndim == 2:
            return

        # There are two approaches: iterate on top of the current
        # noise estimate or generate a new estimate
        if cumul:
            snr = data/self.get_scale_cube()
        else:
            snr = data/self.spatial/self.scale

        # Reshape the working data into a two-d array with the spatial
        # dimensions collapsed together
        snr.reshape((data.shape[0], data.shape[1]*data.shape[2]))

        # Switch on calculation methodology
        if method == "MAD":
            estimate = mad(snr,axis=0,nans=True)
        if method == "STD":
            estimate = nanstd(snr, axis=1)

        # If we are doing a cumulative calculation then append the estimate
        if cumul:
            self.spectral_norm = estimate*self.spectral_norm
        else:
            self.spectral_norm = estimate

        # Set channels where the estimate has failed to 1 (the scalar noise)
        self.spectral_norm[np.isnan(self.spectral_norm) | (self.spectral_norm==0)]=1.
        return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Manipulate noise estimates
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    def spatial_smooth(
            self,
            kernel=None,
            convbeam=True,
            ):
        """
        Smooth the noise estimate in the spatial dimension. Two
        components: median smoothing and convolving with the beam.
        """
        # Manually median filter (square box)
        if kernel is not None:
            self.spatial_norm = ssig.medfilt2d(self.spatial_norm,
                                               kernel_size=kernel)

        # Convolve with the beam (if it's known)
        if convbeam and self.beam is not None:
            self.spatial_norm = convolve_fft(self.spatial_norm, 
                                             self.beam.as_kernel(get_pixel_scales(self.cube.wcs)),
                                             interpolate_nan=True,normalize_kernel=True)
        return

    def spectral_smooth(
            self,
            kernel=None
            ):
        """
        Median smooth the noise estimate along the spectral dimension.
        """
        # Manually median filter
        self.spectral_norm = ssig.medfilt(self.spectral_norm,
                                          kernel_size=kernel)
        return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # High level exposure for full estimation
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def estimate_noise(
            self,
            method="MAD",
            iter=1,
            spatial_flat=False,
            spatial_smooth=None,
            spectral_flat=False,
            spectral_smooth=None):
        """
        High level noise estimation procedure.
        """

        # Calculate the overall scale
        calculate_scale(method=method)
        self.spatial_norm = np.ones([data.shape[1],data.shape[2]])
        self.spectral_norm = np.ones((data.shape[0]))

        # Iterate over spatial and spectral variations
        for count in range(niter):
            if not spatial_flat:
                self.calculate_spatial(method=method, cumul=True)
            if spatial_smooth is not None or self.beam is not None:
                self.spatial_smooth(kernel=spatial_smooth,convbeam=True)
            if not spectral_flat:
                self.calculate_spectral(method=method, cumul=True)
            if spectral_smooth is not None:
                self.spectral_smooth(kernel=spectral_smooth)

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

        # This needs reworking but is close.

        for count in range(niter):
            if self.spatial_norm is not None:
                noise = self.get_scale_cube()
                snr = self.cube.filled_data[:]/noise
            else:
                snr = self.cube.filled_data[:]/self.scale
            # Include negatives in the signal mask or not?
            newmask = BooleanArrayMask(np.abs(snr)<
                sig_n_outliers(self.cube.size),self.cube.wcs)
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

def mad(
        data, 
        sigma=True, 
        axis=None,
        force=False,
        medval=0.0,
        nans=False):
    """Return the median absolute deviation (or the absolute deviation
    about a fixed value - default zero - if force is set to True). By
    default returns the equivalent sigma. Axis functionality adapted
    from https://github.com/keflavich/agpy/blob/master/agpy/mad.py
    Flips nans to True (default false) to use with nans.
    """
    if axis>0:
        if force:
            med = medval
        else:
            if nans:
                med = nanmedian(data.swapaxes(0,axis),axis=0)
            else:
                med = np.median(data.swapaxes(0,axis),axis=0)
        if nans:
            mad = nanmedian(np.abs(data.swapaxes(0,axis) - med),axis=0)
        else:
            mad = np.median(np.abs(data.swapaxes(0,axis) - med),axis=0)
    else:
        if force:
            med = medval
        else:
            med = np.median(data,axis=axis)
        mad = np.median(np.abs(data - med),axis=axis)
    if sigma==False:
        return mad
    else:
        return mad*1.4826

# COMMENTARY: Deprecate or incorporate (axis functionality would be
# ideal)

def sigma_rob(
        data, 
        iterations=1, 
        thresh=3.0):
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

# COMMENTARY: The above two are good here, this should either go to a
# utils directory or move to the spectral_cube object (the latter
# preferrably).

def get_pixel_scales(mywcs):
    """Extract a pixel scale (this assumes square pixels) from a wcs.
    """
    # borrowed from @keflavich who borrowed from aplpy
    
    mywcs = mywcs.sub([astropy.wcs.WCSSUB_CELESTIAL])
    cdelt = np.array(mywcs.wcs.get_cdelt())
    pc = np.array(mywcs.wcs.get_pc())
    # I too like to live dangerously:
    scale = np.array([cdelt[0] * (pc[0,0]**2 + pc[1,0]**2)**0.5,
     cdelt[1] * (pc[0,1]**2 + pc[1,1]**2)**0.5])
    return abs(scale[0])


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Parallel generic scipy.stats approach
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Right now this is more of a roadmap than a viable
    # implementation.

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
