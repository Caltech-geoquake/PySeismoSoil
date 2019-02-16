# Author: Jian Shi

import numpy as np
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_site_response as sr

#%%----------------------------------------------------------------------------
def tau_MKZ(gamma, *, gamma_ref, beta, s, Gmax):
    '''
    Calculate the MKZ shear stress. The MKZ model is proposed in Matasovic and
    Vucetic (1993), and has the following form:

                              Gmax * gamma
        T(gamma) = ---------------------------------------
                      1 + beta * (gamma / gamma_ref)^s

    where T         = shear stress
          gamma     = shear strain
          Gmax      = initial shear modulus
          beta      = a shape parameter of the MKZ model
          gamma_ref = reference strain, another shape parameter of the MKZ model
          s         = another shape parameter of the MKZ model

    Parameters
    ----------
    gamma : numpy.ndarray
        The shear strain array. Must be a 1D array. Its unit should be '1',
        rather than '%'.
    gamma_ref : float
        Reference shear strain, a shape parameter of the MKZ model
    beta : float
        A shape parameter of the MKZ model
    s : float
        A shape parameter of the MKZ model
    Gmax : float
        Initial shear modulus. Its unit can be arbitrary, but we recommend Pa.

    Returns
    -------
    T_MKZ : numpy.ndarray
        The shear stress determined by the formula above. Same shape as `x`,
        and same unit as `Gmax`.
    '''
    hlp.assert_1D_numpy_array(gamma, name='`gamma`')
    T_MKZ = Gmax * gamma / ( 1 + beta * (np.abs(gamma) / gamma_ref)**s )

    return T_MKZ


