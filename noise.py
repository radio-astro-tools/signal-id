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
from spectral_cube.spectral_cube import SpectralCubeMask,SpectralCube

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
    """ 
    Class used to track the parameters of the noise in a radio data cube.  

    Limitations
    -----------
    Many of the routines in the package rely on the assumption of the 
    data distribution begin centered around zero.
    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Contents
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # This is a global "representative" noise distribution width as a scalar
    scale = None   

    # This is a spatial map of the width parameter
    spatial_norm = None
    spatial_footprint = None

    # This is the 1D spectral normalization constructed so
    # that spectral_norm[channel] * map[x,y] = noise width at
    # this point in cube

    spectral_norm = None
    spectral_footprint = None

    # Noise distribution comaptible with scipy
    distribution = None
    distribution_shape = None

    # Map of arbitrary shapes for the distribution. 
    distribution_shape_map = None

    # Associated data cube and beam
    cube = None
    beam = None

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
        spectral_norm = None,
        beam = None):
        """Construct a new Noise object."""

        if isinstance(cube,SpectralCube):
            self.cube = cube
            self.spatial_footprint = np.any(cube.get_mask_array(),axis=0)

        if isinstance(beam,Beam):
            self.beam = beam

        self.distribution = ss.norm

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
        Generates a matched data set of pure noise with properties matching
        the stated distribution.
        """
        if self.spatial_norm is None:
            return self.distribution.rvs(*self.distribution_shape,
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

    def snr(self):
        """
        Return a signal-to-noise cube.
        """

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

    def scalar_noise(self):
        """
        Derive the scale of the noise distribution using only the 
        negative values in the array and the MAD()
        """
        negs = self.cube.flattened().astype('=f')
        negs = negs[negs<0]
        self.scale = nanmad(negs)
        return

    def calculate_naive(self):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the standard deviation is a rigorous method.

        Parameters: None
        """

        swapa = self.cube.get_filled_data().astype('=f')
        
        self.scale = nanstd(swapa)
        self.spatial_norm = nanstd(swapa,axis=0)/self.scale
        self.spectral_norm = nanstd(swapa.reshape((swapa.shape[0],
                                                   swapa.shape[1]*
                                                   swapa.shape[2])),
                                    axis=1)/self.scale
        return

    def calculate_mad(self,niter=1,spatial_smooth=None,
        spectral_smooth=None,spatial_flat=False,spectral_flat=False):
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

        data = self.cube.get_filled_data().astype('=f')
        self.scalar_noise()
        if self.spatial_norm is None:
            self.spatial_norm = np.ones((data.shape[1],data.shape[2]))
            self.spectral_norm = np.ones((data.shape[0]))
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

        data = self.cube.get_filled_data().astype('=f')
        self.scale = nanstd(data)
        if self.spatial_norm is None:
            self.spatial_norm = np.ones((data.shape[1],data.shape[2]))
            self.spectral_norm = np.ones((data.shape[0]))
        for count in range(niter):
            scale = self.get_scale_cube()
            snr = data/scale
            self.spatial_norm = nanstd(snr,axis=0)*self.spatial_norm
            if beam is not None:
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

    def rolling_shape_fit(self,
                            boxsize=5):
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

        """
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
                snr = self.cube.get_filled_data()/noise
            else:
                snr = self.cube.get_filled_data()/self.scale
            # Include negatives in the signal mask or not?
            newmask = SpectralCubeMask(np.abs(snr)<
                sig_n_outliers(self.cube.size),self.cube.wcs)
            self.cube = self.cube.apply_mask(newmask)

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
