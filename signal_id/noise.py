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

# Astropy - used for convolution and WCS
import astropy.wcs
from astropy.convolution import convolve_fft, convolve, Kernel2D

# Spectral cube object from radio-astro-tools
from spectral_cube import SpectralCube, BooleanArrayMask

# Radio beam object from radio-astro-tools
try:
    from radio_beam import Beam
except ImportError:
    warnings.warn("No radio_beam.Beam instances found. Convolution\
     with radio beams will not work")
    Beam = type(None)
# Utility functions from this package
from utils import *

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# The Noise Object
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


class Noise(object):
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

    def __init__(self, cube, scale=None, spatial_norm=None,
                 spectral_norm=None, beam=None, method="MAD"):
        """
        Construct a new Noise object.

        Parameters
        ----------

        method : {'MAD','STD'}
            Chooses method for estimating noise variance either 'MAD'
            for median absolute deviation and 'STD' for standard
            deviation.  Default: 'MAD'

        """
        if isinstance(cube,SpectralCube):
            self.cube = cube
        elif isinstance(cube, str):
            self.cube = SpectralCube.read(cube)
        else:
            warnings.warn("Noise currently requires a SpectralCube instance.")

        self.spatial_footprint = np.any(self.cube.get_mask_array(), axis=0)

        if beam is None:
            try:
                self.beam = cube.beam
            except AttributeError:
                warnings.warn("cube object has no associated beam. All beam "
                              "operations are disabled.")
                self.beam = None
            self.astropy_beam_flag = False
        else:
            if isinstance(beam, Beam):
                self.astropy_beam_flag = False
            elif isinstance(beam, Kernel2D):
                self.astropy_beam_flag = True
            else:
                warnings.warn("beam must be a radio_beam Beam object or an "
                              "astropy Kernel2D object. All beam operations "
                              "are disabled.")
            self.beam = beam

        # Default to a normal distribution
        self.distribution = ss.norm

        # SUGGESTION: calculate on initialization?

        # Fit the data
        if scale is None:
            self.calculate_scale(method=method)  # [1] is the std. of a Gaussian
            self.spatial_norm = np.ones((self.cube.shape[1], self.cube.shape[2]))
            self.spectral_norm = np.ones((self.cube.shape[0]))

        # Compute the scale_cube
        self.get_scale_cube()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Expose the noise in various ways
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def get_scale_cube(self):
        """
        Return a cube of noise width estimates.
        """

        # Make spectral_norm and spatial_norm 3D arrays
        ax0 = self.spectral_norm[:, np.newaxis, np.newaxis]
        ax12 = self.spatial_norm[np.newaxis, :]

        self._scale_cube = (ax12 * ax0) * self.scale
        return self

    @property
    def scale_cube(self):
        return self._scale_cube

    def get_noise_cube(self):
        """
        Generates a data cube of randomly generated noise with properties
        matching the measured distribution.
        """

        noise = np.random.randn(*self.cube.shape)
        if self.beam is not None:
            if self.astropy_beam_flag:
                beam = self.beam
            else:
                beam = self.beam.as_kernel(get_pixel_scales(self.cube.wcs))
            # Iterate convolution over plane (ugh)
            for plane in np.arange(self.cube.shape[0]):
                noise[plane, :, :] = convolve_fft(noise[plane, :, :],
                                                  beam,
                                                  normalize_kernel=True)

            self._noise_cube = noise * self.scale_cube
        return self

    @property
    def noise_cube(self):
        return self._noise_cube

        # Eventually, we want to use self.distribution.rvs for arbitrary distributions

    @property
    def snr(self,
            as_prob=False):
        """
        Return a signal-to-noise cube.
        """

        return self.cube.filled_data[:].value/self.scale_cube

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

    def calculate_scale(self, method="MAD", iterations=1, thresh=3.0):
        """
        Derive the scale of the noise distribution using only the
        negative values in the array and the MAD, or using STD.
        Both methods employ positive outlier rejection. The rejection
        criterion is set by ```thresh```.
        """

        data = self.cube.flattened().value.astype('=f')

        if method == "MAD":
            if (data < 0).any():
                negs = data[data < 0]
                self.scale = sigma_rob(negs, iterations=iterations,
                                       thresh=thresh, function=mad,
                                       function_kwargs={"force": True, "medval": 0.0})
            else:
                warnings.warn("No negative values in the cube. \
                               Using STD to get scale.")
                method = "STD"
        if method == "STD":
            self.scale = sigma_rob(data, iterations=iterations, thresh=thresh,
                                   function=std, function_kwargs={'axis': 0})

        ### MEAN DEFAULTS TO 0 ALWAYS!!!
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
        if self.cube.ndim == 2:
            return

        # There are two approaches: iterate on top of the current
        # noise estimate or generate a new estimate
        if cumul:
            snr = self.snr
        else:
            snr = self.cube.filled_data[:].value/\
                self.spectral_norm[:, np.newaxis, np.newaxis]/self.scale

        # Switch estimate on methodology
        if method == "MAD":
            estimate = mad(snr, axis=0)

        if method == "STD":
            estimate = nanstd(snr, axis=0)

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
        if self.cube.ndim == 2:
            return

        # There are two approaches: iterate on top of the current
        # noise estimate or generate a new estimate
        if cumul:
            snr = self.snr
        else:
            snr = self.cube.filled_data[:].value/\
                  self.spatial_norm[np.newaxis, :]/self.scale

        # Reshape the working data into a two-d array with the spatial
        # dimensions collapsed together
        new_shape = (self.cube.shape[0], self.cube.shape[1]*self.cube.shape[2])
        snr = snr.reshape(new_shape)

        # Switch on calculation methodology
        if method == "MAD":
            estimate = mad(snr, axis=1)
        if method == "STD":
            estimate = nanstd(snr, axis=1)

        # Enforce that the geometric mean of the norm is unity.
        # This stabilizes the iterative estimates.
        estimate = estimate/np.exp(np.nanmean(np.log(estimate)))

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
            spatial_smooth=None,
            spectral_smooth=None,
            niter=1
            ):
        """
        Smooth the noise estimate in the spatial dimension. Two
        components: median smoothing and convolving with the beam.
        """

        # Manually median filter (square box)
        if kernel is not None:
            print "Median filtering"
            self.spatial_norm = ssig.medfilt2d(self.spatial_norm,
                                               kernel_size=kernel)

        data = self.cube.filled_data[:].astype('=f')

        if self.spatial_norm is None:
            self.spatial_norm = np.ones(data.shape[-2:])
            self.spectral_norm = np.ones((data.shape[0]))
        for count in range(niter):
            scale = self.scale_cube
            snr = data/scale
            self.spatial_norm = nanstd(snr,axis=0)*self.spatial_norm
            if self.beam is not None:
                if self.astropy_beam_flag:
                    beam = self.beam
                else:
                    beam = self.beam.as_kernel(get_pixel_scales(self.cube.wcs))

                self.spatial_norm = convolve_fft(self.spatial_norm,
                                                 beam,
                                                 interpolate_nan=True,
                                                 normalize_kernel=True)
            if spatial_smooth is not None:
                self.spatial_norm = ssig.medfilt2d(self.spatial_norm,
                    kernel_size=spatial_smooth)

            snr = data/self.scale_cube
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
        ### THIS IS ALREADY SET IN calculate_scale
        # self.distribution_shape=(0,self.scale)
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
            niter=1,
            spatial_flat=False,
            spatial_smooth=None,
            spectral_flat=False,
            spectral_smooth=None,
            signal_mask=False,
            verbose=False):
        """
        High level noise estimation procedure.  Fills in spatial and
        spectral norm estimates for the noise objects on calling.

        Parameters
        ----------
        method : {'MAD','STD'}
            Chooses method for estimating noise variance either 'MAD'
            for median absolute deviation and 'STD' for standard
            deviation.  Default: 'MAD'
        niter : int
            Number of iterations used in refining spatial vs. spectral
            estimates.
        spatial_flat : bool
            If True asserts that there is no spatial variation in the
            noise scale. Default: False
        spectral_flat : bool
            If True asserts that there is no spectail variation in the
            noise scale.  Default: False
        spatial_smooth : arraylike
            Smooth spatial norm estimate by this kernel.
        spectral_smooth : arraylike
            Smooth spectral norm estimate by this kernel.
        signal_mask : bool
            Mask out signal at end of each iteration.  Default: False
        verbose:
            Increase verbosity to maximum!  Default: False
        """

        # Calculate the overall scale
        if verbose:
            print "Calculating overall scale."
        self.calculate_scale(method=method)
        self.spatial_norm = np.ones((self.cube.shape[1], self.cube.shape[2]))
        self.spectral_norm = np.ones((self.cube.shape[0]))

        # Iterate over spatial and spectral variations
        if verbose:
            print "Iterating to find spatial and spectral variation."
        for count in range(niter):
            if not spatial_flat:
                if verbose:
                    print "Iteration {0}: Calculating spatial variation".format(count)
                self.calculate_spatial(method=method, cumul=True)
            if spatial_smooth is not None or self.beam is not None:
                if verbose:
                    print "Iteration {0}: Smoothing spatial variations by beam".format(count)
                self.spatial_smooth(kernel=spatial_smooth, convbeam=True,
                                    niter=niter, spatial_smooth=spatial_smooth,
                                    spectral_smooth=spectral_smooth)
            if not spectral_flat:
                if verbose:
                    print "Iteration {0}: Calculating spectral variations".format(count)
                self.calculate_spectral(method=method, cumul=True)
            if spectral_smooth is not None:
                if verbose:
                    print "Iteration {0}: Smoothing spectral variations".format(count)
                self.spectral_smooth(kernel=spectral_smooth)
            if signal_mask:
                if verbose:
                    print "Iteration {0}: Masking out signal".format(count)
                self.mask_out_signal()

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
                noise = self.scale_cube
                snr = self.cube.filled_data[:].value/noise
            else:
                snr = self.cube.filled_data[:].value/self.scale
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
            data = self.cube.filled_data[:].value.astype('=f')
            scale = self.scale_cube
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
            print Nbins
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
        plt.clf()
        plt.plot(channel, self.spectral_norm)
        plt.xlabel('Channel')
        plt.ylabel('Spectral norm of noise estimate')

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

        plt.clf()
        plt.imshow(self.spatial_norm, origin="lower")
        plt.colorbar()

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
        data = self.cube.filled_data[:].value

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
                          1] = mad(data[xmatch[inarray].ravel(),
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
