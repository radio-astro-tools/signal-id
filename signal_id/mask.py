# Generate and manipulate three-d Boolean arrays for purposes of
# signal identification.

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# IMPORTS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# python
import time
import copy
import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# radio tools
from spectral_cube import SpectralCube, BooleanArrayMask
from spectral_cube.masks import is_broadcastable_and_smaller

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# BASE CLASS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%


class RadioMask(object):
    """
    Holds a binary array with associated metadata.
    """

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Attributes and Properties
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Current version of the array
    _value = None

    # Previous version of the array
    _backup = None

    # Toggles
    _implicit_backup = True

    # The associated cube
    _linked_data = None

    # TBD: noise / SNR cube

    # TBD: beam

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Construction
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def __init__(self, data, thresh=None, backup=True, *args):
        '''

        Parameters
        ----------
        data : {SpectralCube, str, np.ndarray}
            Datacube to create the mask from.
        thresh : float, optional
            Threshold level used to create the mask.
        '''

        if isinstance(data, SpectralCube):
                self.from_spec_cube(data, thresh=thresh, *args)

        elif isinstance(data, str):
            self.from_file(data, *args)

        elif isinstance(data, np.ndarray):
            self.from_array(data, *args)

        else:
            raise TypeError("Input of type %s is not accepted." % (type(data)))

        # Apply specified threshold to create an initial mask
        if thresh is not None:
            self._mask *= self.linked_data > thresh

        # Start log of method calls
        logging.basicConfig()
        self._log = logging.getLogger("method_calls")
        self._log.info("Object created on : ",
                       time.asctime(time.localtime(time.time())))

        # Switch on backup
        if backup:
            self.enable_backup()
        else:
            self.disable_backup()

    def from_file(self, fname, thresh=None, format='fits'):
        cube = SpectralCube.read(fname, format=format)
        self.from_spec_cube(cube, thresh=None)

    def from_spec_cube(self, cube, thresh=None):
        self._linked_data = cube
        self._mask = cube._mask.include()
        self._wcs = cube.wcs

    def from_array(self, array, thresh=None, wcs=None):
        self._linked_data = array
        self._mask = np.isfinite(array)
        self._wcs = wcs

    @property
    def linked_data(self):
        return self._linked_data

    @property
    def mask(self):
        return self._mask

    @property
    def wcs(self):
        return self._wcs

    @property
    def shape(self):
        return self._mask.shape

    @property
    def log(self):
        return self._log

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Output
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def copy(self):
        """
        Return a copy.
        """
        return copy.deepcopy(self)

    def as_spec_cube(self, scale=True):
        """
        Return a spectral cube. Use scale to change type.
        """
        if isinstance(self._linked_data, SpectralCube):
            return SpectralCube(self._mask*scale,
                                wcs=self._wcs)
        return SpectralCube(self._mask*scale,
                            wcs=self._wcs)

    def to_mask(self):
        return BooleanArrayMask(self._mask, self._wcs)

    def to_invertedmask(self):
        copy_mask = self.copy()
        copy_mask.invert()
        return BooleanArrayMask(self._mask, self._wcs)

    def write(self, fname, scale=1):
        """
        Write to a file. Default to using ints.
        """
        # So wasteful...
        cube = self.as_spec_cube(self, scale=scale)
        cube.write(fname)

    def attach_to_cube(self, cube=None, empty=np.NaN):
        """
        Attach the mask to a cube.
        """
        if cube is None:
            cube = self.linked_data

        if isinstance(cube, SpectralCube):
            mask = BooleanArrayMask(self._mask, self._wcs)
            return cube.with_mask(mask)

        if isinstance(cube, np.ndarray):
            if cube.shape == self._mask.shape:
                # Replace False with NaNs
                cube = np.where(self._mask, cube, empty)
            else:
                raise ValueError("Mask is not the same shape as the cube.")

        else:
            raise TypeError("Cube insufficiently specified.")

        return cube

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Expose the mask in various ways
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def as_indices(self):
        """
        As a tuple of indices where True, useful for indexing.
        """
        return np.where(self._mask)

    def as_index_array(self, coordaxis=0):
        """
        As a numpy array of True indices. Optionally specify the axis for
        the coordinates (0 or 1)
        """
        if coordaxis == 0:
            return np.vstack(np.where(self._mask))
        else:
            return np.vstack(np.where(self._mask)).transpose()

    def oned(self, axis=0, sum=False):
        raise NotImplementedError()

    def twod(self, axis=0, sum_axis=False):
        """
        Return a two-dimensional version of the mask.
        """
        if self._mask.ndim == 2:
            return self._mask
        if sum_axis:
            return np.sum(self._mask, axis=axis)
        else:
            return np.max(self._mask, axis=axis)

    def independent_channels(self, struct=None):
        raise NotImplementedError()

    def independent_areas(self, struct=None):
        raise NotImplementedError()

    def independent_cubes(self, struct=None):
        raise NotImplementedError()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Undo/Redo
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def enable_backup(self):
        self._implict_backup = True

    def disable_backup(self):
        self._implict_backup = False

    def log_and_backup(self, func):
        '''
        Back-up and log method calls.
        '''
        self._log.info(func.__name__)
        if self.is_backup_enabled:
            self._backup = self._mask.copy()

    def undo(self):
        self._log.info("UNDO")
        temp = self._backup.copy()
        self._backup = self._mask.copy()
        self._mask = temp

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Operators
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Union
    def union(self, other):
        self.log_and_backup(self.union)
        if isinstance(other, RadioMask):
            other = other.mask
        # Check if arrays are broadcastable
        if not is_broadcastable_and_smaller(self.shape, other.shape):
            raise ValueError("Mask shapes are not broadcastable.")

        self._mask = np.logical_or(self._mask, other)

    # Intersection
    def intersection(self, other):
        self.log_and_backup(self.intersection)
        if isinstance(other, RadioMask):
            other = other.mask
        # Check if arrays are broadcastable
        if not is_broadcastable_and_smaller(self.shape, other.shape):
            raise ValueError("Mask shapes are not broadcastable.")

        self._mask = np.logical_and(self._mask, other)

    # Exclusive or
    def xor(self, other):
        self.log_and_backup(self.xor)
        if isinstance(other, RadioMask):
            other = other.mask
        # Check if arrays are broadcastable
        if not is_broadcastable_and_smaller(self.shape, other.shape):
            raise ValueError("Mask shapes are not broadcastable.")

        self._mask = np.logical_xor(self._mask, other)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Manipulation
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Inversion
    def invert(self, struct=None):
        self.log_and_backup(self.invert)
        self._mask = np.logical_not(self._mask)

    # Dilation
    def dilate(self, struct=None, iterations=1):
        self.log_and_backup(self.dilate)
        self._mask = nd.binary_dilation(self._mask, structure=struct,
                                        iterations=iterations)

    # Erosion
    def erode(self, struct=None, iterations=1):
        self.log_and_backup(self.erode)
        self._mask = nd.binary_erosion(self._mask, structure=struct,
                                       iterations=iterations)

    # Opening
    def open(self, struct=None, iterations=1):
        self.log_and_backup(self.open)
        self._mask = nd.binary_opening(self._mask, structure=struct,
                                       iterations=iterations)

    # Closing
    def close(self, struct=None, iterations=1):
        self.log_and_backup(self.close)
        self._mask = nd.binary_closing(self._mask, structure=struct,
                                       iterations=iterations)

    # Reject on property
    def reject_region(self, func=None, thresh=None, struct=None):
        # self.log_and_backup(self.reject_region)
        raise NotImplementedError()

    # Reject on volume (special case)
    def reject_on_volume(self, thresh=None, struct=None):
        # self.log_and_backup(self.reject_volume)
        raise NotImplementedError()

    def apply_custom_func(self, func, *args):
        self.log_and_backup(self.apply_custom_func)
        self._mask = func(self._mask, *args)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # Mask generation
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Structured thresholding
    def threshold(self, thresh=None, struct=None):
        raise NotImplementedError()

    # Projected 2d prior ("drop down" a twod mask)

    # Projected 3d prior ("inflate" a velocity field)

    # Define line-free channels

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # RECIPES
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Convolve-then-threshold (2+1d)

    # High-reject-grow-low

    # Autotune thresholding

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# STRUCTURING ELEMENTS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# VISUALIZATION AND DIAGNOSTICS
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

