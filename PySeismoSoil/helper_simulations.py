# Author: Jian Shi

import numpy as np

from . import helper_generic as hlp
from . import helper_site_response as sr

#%%----------------------------------------------------------------------------
def check_layer_count(vs_profile, *, GGmax_and_damping_curves=None,
                      G_param=None, xi_param=None):
    '''
    Check that ``G_param`` and ``xi_param`` have enough sets of parameters for
    ``vs_profile``, or ``GGmax_curves`` and ``xi_curves`` have enough sets of
    curves for ``vs_profile``.

    Parameters
    ----------
    vs_profile : class_Vs_profile.Vs_Profile
        Vs profile.
    GGmax_and_damping_curves : class_curves.Multiple_GGmax_Damping_Curves
        G/Gmax and damping curves.
    G_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        HH or MKZ parameters for G/Gmax curves.
    xi_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        HH or MKZ parameters for damping curves.
    '''
    max_mat_num = np.max(vs_profile._material_number)
    if G_param is not None and G_param.n_layer < max_mat_num:
        raise ValueError('Not enough sets of parameters in `G_param` for '
                         '`vs_profile`.')
    if xi_param is not None and xi_param.n_layer < max_mat_num:
        raise ValueError('Not enough sets of parameters in `xi_param` for '
                         '`vs_profile`.')
    if GGmax_and_damping_curves is not None \
    and GGmax_and_damping_curves.n_layer < max_mat_num:
        raise ValueError('Not enough sets of curves in '
                         '`GGmax_and_damping_curves` for `vs_profile`.')
    return None
