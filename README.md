signal-id
=========

Signal identification tools (masking and noise) for spectral line data.


The Noise Spec
==============


* noise is a object linked to a SpectralCube characterizing its
3D noise properties  

* Noise is characterized by a scalar noise estimate scale, which
typifies the noise for a cube.  The scalar is broadcast to all points
in the image or the cube using a 2D spatial rescaling and a 1D
spectral rescaling parameter so that the noise scale is 

	 noise[x,y,z] = scale * spatial_norm[x,y] * spectral_norm[z]

* The distribution defaults to a normal distribution with a zero
center.  Use scipy.stats function handlers to test and sample from the
variation.

* Instantiates from the representations (the width of a normal
distribution), or (usually) from data (generates noise width
estimates).

* Generates SNR cube when combined with SpectralCube structure.

The Mask Spec
=============

* rich object that knows basic ppv cube functionality

* does not explicitly contain a linked data object, but accepts them
  as necessary in method calls (?noise linkage is to data?)

* basic operations (combined with other): suggested - multiplication =
  intersection, addition = union, subtraction = xor, divison =
  madness. These need to be exposed, don't have to be operators.

* masking methodology for basic user:

  * thresholding
  * thresholding + structuring element (extends "joint thresholding")
  * velocity field + width

* basic mask manipulation methods
  
  * dilation
  * erosion
  * opening
  * closing

  * reject_island_on_condition - label then evaluate each island,
    reject on some condition. Possible link to some property
    calculator but also maybe to implement as a function pass.
    Potentially involves some .

    (e.g., area, extent, perimeter)

* to/from disk in sensible way

* interface with cube:
  * to produce vectorized data
  * to allow cube to be split into a set of smaller subcubes that span
    all contiguous assignment
  
