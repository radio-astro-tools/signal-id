# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Importx
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import time

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
    distribution_map = None

    data = None
    spec_axis = None

    # Holds the signal to noise ratio.
    # SNR[x,y,z] = Data[x,y,z] / (scale*spectral_norm[z]*spatial_norm[x,y])
    snr = None

    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Initialize and infrastructure
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(
        self,
        data,
        scale=None,
        spatial_norm = None,
        spectral_norm = None):
        """
        Construct a new Noise object.
        """
        self.data = data        
# Hardwire spec_axis for now.  Wait for spectral cube.
        self.spec_axis = 0
        self.distribution = ss.norm
        self.calculate_fit()
        
    def calculate_fit(self):
        """
        Fits the distribution of the data using the scipy.fit
        functionality.  This uses the distribution that characterizes
        the noise values.
        """
        self.distribution_shape = self.distribution.fit(self.data)
        return


    def calculate_naive(self):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the standard deviation is a rigorous method.
        """
        if self.spec_axis !=0:
            swapa = self.data.swapaxes(0,self.spec_axis)
        else:
            swapa = self.data
        self.scale = swapa.std()
        self.spatial_norm = swapa.std(axis=0)/self.scale
        self.spectral_norm = swapa.reshape((swapa.shape[0],
                                                swapa.shape[1]*
                                                swapa.shape[2])).\
                                                std(axis=1)/self.scale
        return

    def calculate_mad(self):
        """
        Calculates the naive values for the scale and norms under the
        assumption that the standard deviation is a rigorous method.
        """
        if self.spec_axis !=0:
            swapa = self.data.swapaxes(0,self.spec_axis)
        else:
            swapa = self.data
        self.scale = mad(swapa)
        self.spatial_norm = mad(swapa,axis=0)/self.scale
        
        self.spectral_norm = mad(swapa.reshape((swapa.shape[0],
                                                swapa.shape[1]*
                                                swapa.shape[2])),axis=1)/\
                                                self.scale
        return

    
    def cube_of_noise(self):
        """
        Generates a matched data set of pure noise with properties matching
        the stated distribution.
        """
        return self.distribution.rvs(*self.distribution_shape,
                                     size=self.data.shape)

    def plot_noise(self):
        """
        Makes a plot of the data distribution and the estimated
        parameters of the PDF.
        """
        try:
            import matplotlib.pyplot as pl
        except ImportError:
            return

        xmin = self.distribution.ppf(1./self.data.size,*self.distribution_shape)
        xmax = self.distribution.ppf(1-1./self.data.size,*self.distribution_shape)
        print(xmin,xmax)
        xsamples = np.linspace(xmin,xmax,100)
        Nbins = np.min([int(np.sqrt(self.data.size)),100])
        binwidth = (xmax-xmin)/Nbins
        pl.xlim(xmin,xmax)
        pl.hist(self.data.ravel(),bins = Nbins,log=True)
        pl.plot(xsamples,binwidth*self.data.size*
                self.distribution.pdf(xsamples,*self.distribution_shape),
                linewidth=4,alpha=0.75)
        pl.xlabel('Data Value')
        pl.ylabel('Number of Points')
                
        

#     def set_spectral_axis(
#         self,
#         val=None
#         ):
#         """
#         Set the spectral axis.
#         """
#         if val is not None:
#             self.spec_axis = val

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Generate a noise estimate
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

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

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Noise Routines
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# These may be of general use and so are not part of the noise
# class. Instead they can be called piecemeal.


# ------------------------------------------------------------
# STASTICS HELPER PROCEDURES
# ------------------------------------------------------------

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
    
