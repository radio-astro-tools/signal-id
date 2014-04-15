# Mask class, intended to hold masks (as opposed to data or
# assignment). Extends the cube class.

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# IMPORTS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

import time
import copy
import numpy as np

from scipy.ndimage import histogram
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.ndimage import label, find_objects

import matplotlib.pyplot as plt

from pyprops import cube, noise
from struct import *

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# MASK OBJECT
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

class Mask(cube.Cube):
    """
    ...
    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Attributes (in addition to those in Cube)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    backup = None
    linked_data = None
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Initialize
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(
        self,
        *args,
        **kwargs
        ):
        """
        Construct a new mask object.
        """
        thresh = kwargs.pop("thresh", 0.5)
        cube.Cube.__init__(self, *args, **kwargs)        
        self.data = (self.data > thresh)
        self.data[self.valid == False] = False
        self.valid = None

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Copy from another cube
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Modify the lower level call to link to the data

    def init_from_cube(
        self, 
        prev):
        """
        Initialize a new cube from another cube. Copy the data.
        """
        cube.Cube.init_from_cube(self, prev)
        self.linked_data = prev

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Links to data cube
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def set_linked_data(
        self,
        val=None
        ):
        """
        Link the mask object to a data cube object.
        """
        if val != None:
            self.linked_data = val

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Backup/undo
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def step_back(
        self):
        """
        Restore the backup mask, setting it to be the new mask.
        """
        if self.backup != None:
            self.data = self.backup

    def save_backup(
        self):
        """
        Restore the backup mask, setting it to be the new mask.
        """
        self.backup = self.data

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Read/write
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def from_casa_image(
        self,
        *args,
        **kwargs
        ):
        """
        Read a mask from a CASA image.
        """
        
        # Pull out two keyowrds ...
        append = kwargs.pop("append", None)
        thresh = kwargs.pop("thresh", 0.5)

        if append == True:
            self.backup = self.data
        
        cube.Cube.from_casa_image(
            self,
            *args,
            **kwargs)

        self.data = (self.data > thresh)
        self.valid = None
        if append:
            self.data = self.data*self.backup        

    def to_casa_image(
        self,
        *args,
        **kwargs):
        """
        Write a mask to a CASA image file.
        """

        # Recast as float
        if kwargs.has_key("data") == False:
            kwargs["data"] = self.data*1.0

        cube.Cube.to_casa_image(
            self,
            *args,
            **kwargs)

    def from_fits_file(
        self,
        *args,
        **kwargs):
        """
        Read a mask from a FITS file.
        """

        # Pull out two keyowrds ...
        append = kwargs.pop("append", None)
        thresh = kwargs.pop("thresh", 0.5)

        if append == True:
            self.backup = self.data
        
        cube.Cube.from_fits_file(
            self,
            *args,
            **kwargs)

        self.data = (self.data > thresh)
        self.valid = None
        if append:
            self.data = self.data*self.backup

    def to_fits_file(
        self, 
        *args, 
        **kwargs):
        """
        Write the cube to a FITS file.
        """

        # Recast as float
        if kwargs.has_key("data") == False:
            kwargs["data"] = self.data*1.0

        cube.Cube.to_fits_file(
            self,
            *args,
            **kwargs)
        
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Expose the mask
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def twod(
        self,
        axis=None):
        """
        Return a two-dimensional version of the mask.
        """
        if axis==None:
            if self.spec_axis==None:
                if self.linked_data.spec_axis!=None:
                    axis=self.linked_data.spec_axis
            else:
                axis=self.spec_axis

        if self.data.ndim == 2:
            return self.data

        if axis == None:
            return None
        
        return (np.sum(self.data, axis=axis) >= 1)
        

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Manipulate the mask
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def pare_on_volume(
        self,
        thresh=None,
        corners=False,
        timer=False,
        backup=False
        ):
        """
        Remove discrete regions that do not meet a pixel-volume
        threshold. Requires a pixel threshold be set.
        """

        # .............................................................
        # Error checking
        # .............................................................

        if thresh==None:
            print "Need a threshold."
            return

        # .............................................................
        # Back up the mask first if requested
        # .............................................................

        if backup==True:
            self.backup=self.data

        # .............................................................
        # Label the mask
        # .............................................................
        
        structure = (Struct(
                "simple", 
                ndim=self.data.ndim,                
                corners=corners)).struct

        labels, nlabels = label(self.data,
                                structure=structure)

        # .............................................................
        # Histogram the labels
        # .............................................................

        hist = histogram(
            labels, 0.5, nlabels+0.5, nlabels)
        
        # .............................................................
        # Identify the low-volume regions
        # .............................................................
        
        if np.sum(hist < thresh) == 0:
            return
        
        loc = find_objects(labels)

        for reg in np.arange(1,nlabels):
            if hist[reg-1] > thresh:
                continue
            self.data[loc[reg-1]] *= (labels[loc[reg-1]] != reg)
            
    def erode_small_regions(
        self,        
        major=3,
        depth=2,
        timer=False,
        backup=False
        ):
        """
        Use 'morphological opening' (erosion followed by dilation) to
        remove small regions from the mask.
        """

        # .............................................................
        # Back up the mask first if requested
        # .............................................................

        if backup==True:
            self.backup=self.data

        # .............................................................
        # Time the operation if requested.
        # .............................................................

        if timer:
            start=time.time()
            full_start=time.time()

        # .............................................................
        # Construction of structuring element
        # .............................................................        

        structure = Struct(
            "rectangle", 
            major=major, 
            zaxis=self.spec_axis, 
            depth=depth)
        
        # .............................................................
        # Erosion
        # .............................................................

        self.data = binary_erosion(
            self.data, 
            structure=structure.struct,
            iterations=1
            )

        # .............................................................
        # Dilation
        # .............................................................

        self.data = binary_erosion(
            self.data, 
            structure=structure.struct,
            iterations=1
            )

        # .............................................................
        # Finish timing
        # .............................................................

        if timer:
            full_stop=time.time()
            print "Small region suppression took ", full_stop-full_start

        return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Grow the mask
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def grow(
        self,
        iters=-1,
        xy_only=False,
        z_only=False,
        corners=False,
        constraint=None,
        timer=False,
        verbose=False,
        backup=False
        ):
        """
        Manipulate an existing mask. Mostly wraps binary dilation operator
        in scipy with easy flags to create structuring elements.
        """

        # .............................................................
        # Back up the mask first if requested
        # .............................................................

        if backup==True:
            self.backup=self.data

        # .............................................................
        # Time the operation if requested.
        # .............................................................

        if timer:
            start=time.time()
            full_start=time.time()

        # .............................................................
        # Construct the dilation structure (calls "connectivity" in the
        # blobutils).
        # .............................................................
   
        skip_axes = []

        # ... if 2d-only then blank the spectral axis (if there is
        # one) in the connectivity definition.

        if xy_only == True:
            if self.spec_axis != None:
                skip_axes.append(self.spec_axis)

        # ... if 1d-only then blank the position axes (if they exist)
        # in the connectivity definition.

        if z_only == True:
            axes = range(self.data.ndim)
            for axis in axes:
                if axis != self.spec_axis:
                    skip_axes.append(axis)

        # ... build the sturcturing element

        structure = Struct(
            "simple", 
            ndim=self.data.ndim,                
            corners=corners)
        for skip in skip_axes:
            structure.suppress_axis(skip)

        # .............................................................
        # Apply the binary dilation with the constructed parameters
        # .............................................................

        self.data = binary_dilation(
            self.data, 
            structure=structure.struct,
            iterations=iters,
            mask=constraint,
            )

        # .............................................................
        # Finish timing
        # .............................................................

        if timer:
            full_stop=time.time()
            print "Mask expansion took ", full_stop-full_start

        return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Generate a new mask
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def threshold(
        self,
        usesnr=True,
        scale=1.0,
        thresh=4.0,
        nchan=2,
        outof=None,
        validonly=True,
        append=False,
        backup=False,
        timer=False,
        verbose=True
        ):
        """
        Masking using joint velocity channel conditions.
        
        usesnr (default True) : use a signal-to-noise cube if one can be 
        derived from the data (requires associated data and noise).

        scale (default 1.0) : a factor used to scale the threshold. Set by
        default to 1.0, appropriate for a threshold in S/N units being
        applied to mask a signal-to-noise cube. Set it to

        threshold (default 4.0) : the threshold (times the scale) that
        must be exceeded for a pixel to be included in the mask.
                
        nchan (default 2) : number of channels that must be above the
        specified threshold in order for a region to be included in the
        mask.
        
        out_of (default to nchan) : relaxes the requirement for the number
        of channels searched at once for emission. This can be used to
        allow for noisy / spiky spectra. E.g., require 3 channels out of 5
        to match the threshold with out_of=5, nchan=3. Note that then all
        five channels will included in the final mask.
                
        Defaults to requiring two out of two channels (nchan=2,
        out_of=None) above 4.0 (thresh=4.0), assuming that it has been fed
        a signal-to-noise mask (scale=1.0).
        """

        # .............................................................
        # Back up the mask first if requested
        # .............................................................

        if backup==True:
            self.backup=self.data

        # .............................................................
        # Time the operation if requested.
        # .............................................................

        if timer:
            start=time.time()
            full_start=time.time()

        # .............................................................
        # Set defaults and catch errors
        # .............................................................

        nchan = int(nchan)

        # default out_of to nchan
        if outof==None:
            outof = nchan

        # catch error case
        if outof < nchan:
            outof = nchan 

        if verbose:
            print "Thresholding in "+str(nchan)+ \
                " out of "+str(outof)+" channels."

        # .............................................................
        # Get the data that we will work with
        # .............................................................

        if usesnr:
            working_data = self.linked_data.snr()
        else:
            working_data = self.linked_data.data

        # .............................................................
        # Build the mask
        # .............................................................

        # initial mask set by threshold
        base_mask = (working_data >= thresh*scale)
        if verbose:
            print " ... total after initial threshold: ", np.sum(base_mask)

        # If we have a spectral axis apply the joint conditions
        if self.spec_axis != None and self.linked_data.data.ndim > 2:

            # roll the cube "out_of" times along the spectral axis and keep a
            # running tally of the number of points above the threshold by
            # summing mask.
            rolled = np.int_(base_mask)
            for i in (np.arange(1,outof,1)):
                rolled += np.roll(base_mask,i,axis=self.spec_axis)
    
            # keep only points in the mask which meet the "nchan" criteria
            base_mask = (rolled >= nchan)
            if verbose:
                print " ... total after roll: ", np.sum(base_mask)

            # roll the mask in the other direction to ensure that all points
            # that contributed to the valid point are included in the final
            # mask
            rolled = np.int_(base_mask)
            for i in (np.arange(1,outof,1)):
                rolled += np.roll(base_mask,-1*i,axis=self.spec_axis)

            # calculate the final mask, adding a finite check, now a bool
            base_mask = (rolled >= 1)

            if verbose:
                print " ... total after roll back: ", np.sum(base_mask)

        # .............................................................
        # Append or replace
        # .............................................................

        if append:
            self.data *= base_mask
        else:
            self.data = base_mask

        if validonly:
            self.data *= self.linked_data.valid

        # .............................................................
        # Finish timing
        # .............................................................

        if timer:
            full_stop=time.time()
            print "Joint thresholding took ", full_stop-full_start

        # return
        return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # The CPROPS Recipe
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def cprops_mask(
        self,
        hithresh=4.0,
        lothresh=2.0,
        nchan=2,
        usesnr=True,
        scale=1.0,
        corners=False,
        append=False,
        backup=False,
        timer=False        
        ):
        pass
        
        # .............................................................
        # Back up the mask first if requested
        # .............................................................

        if backup==True:
            self.backup=self.data

        # .............................................................
        # Time the operation if requested.
        # .............................................................

        if timer:
            start=time.time()
            full_start=time.time()

        # .............................................................
        # Build the mask
        # .............................................................

        inner_mask = Mask(self.linked_data)
        outer_mask = Mask(self.linked_data)
        inner_mask.threshold(            
            usesnr=usesnr,
            scale=scale,
            thresh=hithresh,
            nchan=nchan,
            append=append,
            timer=timer
            )

        inner_mask.erode_small_regions(
            major=3,
            depth=2,
            timer=timer)
            
        outer_mask.threshold(            
            usesnr=usesnr,
            scale=scale,
            thresh=lothresh,
            nchan=nchan,
            append=append,
            timer=timer
            )

        inner_mask.grow(
            corners=corners,
            constraint=outer_mask.data,
            timer=timer
            )
            
        # .............................................................
        # Append or replace
        # .............................................................

        if append:
            self.data *= inner_mask.data
        else:
            self.data = inner_mask.data

        # .............................................................
        # Finish timing
        # .............................................................

        if timer:
            full_stop=time.time()
            print "CPROPS-style masking took ", full_stop-full_start

        # return
        return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Visualize mask
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def contour_on_peak(
        self,
        scale=10
        ):

        plt.figure()

        map = self.linked_data.peak_map()

        vmax = np.max(map[np.isfinite(map)])
        vmin = 0.0
        if scale != None:
            if self.linked_data.noise != None:
                if self.linked_data.noise.scale != None:
                    vmax = scale*self.linked_data.noise.scale
                    vmin = 0.0

        plt.imshow(
            map,
            vmin=vmin,
            vmax=vmax,
            origin='lower')
        plt.contour(
            self.twod(),
            linewidths=0.5,
            colors='white'
            )
        plt.show()
        
