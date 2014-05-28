import signal_id.noise as noise
import scipy.stats as ss
import numpy as np
from spectral_cube import SpectralCube,BooleanArrayMask
from astropy.wcs import wcs
from astropy.io import fits
HEADER_STR = "SIMPLE  =                    T / Written by IDL:  Fri Feb 20 13:46:36 2009      BITPIX  =                  -32  /                                               NAXIS   =                    3  /                                               NAXIS1  =                 1884  /                                               NAXIS2  =                 2606  /                                               NAXIS3  =                  200 //                                               EXTEND  =                    T  /                                               BSCALE  =    1.00000000000E+00  /                                               BZERO   =    0.00000000000E+00  /                                               BLANK   =                   -1  /                                               TELESCOP= 'VLA     '  /                                                         CDELT1  =   -5.55555561268E-04  /                                               CRPIX1  =    1.37300000000E+03  /                                               CRVAL1  =    2.31837500515E+01  /                                               CTYPE1  = 'RA---SIN'  /                                                         CDELT2  =    5.55555561268E-04  /                                               CRPIX2  =    1.15200000000E+03  /                                               CRVAL2  =    3.05765277962E+01  /                                               CTYPE2  = 'DEC--SIN'  /                                                         CDELT3  =    1.28821496879E+03  /                                               CRPIX3  =    1.00000000000E+00  /                                               CRVAL3  =   -3.21214698632E+05  /                                               CTYPE3  = 'VELO-HEL'  /                                                         DATE-OBS= '1998-06-18T16:30:25.4'  /                                            RESTFREQ=    1.42040571841E+09  /                                               CELLSCAL= 'CONSTANT'  /                                                         BUNIT   = 'JY/BEAM '  /                                                         EPOCH   =    2.00000000000E+03  /                                               OBJECT  = 'M33     '           /                                                OBSERVER= 'AT206   '  /                                                         VOBS    =   -2.57256763070E+01  /                                               LTYPE   = 'channel '  /                                                         LSTART  =    2.15000000000E+02  /                                               LWIDTH  =    1.00000000000E+00  /                                               LSTEP   =    1.00000000000E+00  /                                               BTYPE   = 'intensity'  /                                                        DATAMIN =   -6.57081836835E-03  /                                               DATAMAX =    1.52362231165E-02  /     "

def test_mad():
    assert noise.mad(np.array([1,2,3])) == 1.4826

def test_nanmad():
    assert noise.mad(np.array([1,2,3,np.nan,np.nan,np.nan]),nans=True) == 1.4826

def test_scalegen():
    np.random.seed(8675309)
    data = np.random.randn(30,40,50)
    mask = np.ones((30,40,50),dtype='bool')
    h = fits.header.Header.fromstring(HEADER_STR)
    h['NAXIS1'] = 50
    h['NAXIS2'] = 40
    h['NAXIS3'] = 30
    h['NAXIS4'] = 1
    cube = SpectralCube(data,wcs.WCS(h),mask=BooleanArrayMask(mask,wcs=wcs.WCS(h)))
    noiseobj = noise.Noise(cube)
    assert np.isclose(noiseobj.scale,0.9912745238184928)
