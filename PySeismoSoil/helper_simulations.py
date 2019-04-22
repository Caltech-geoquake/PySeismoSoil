# Author: Jian Shi

import numpy as np

def check_layer_count(vs_profile, G_param, xi_param):
    '''
    Check that ``G_param`` and ``xi_param`` have enough sets of parameters for
    ``vs_profile``.

    Parameters
    ----------
    vs_profile : PySeismoSoil.class_Vs_profile.Vs_Profile
        Vs profile.
    G_param : PySeismoSoil.class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        HH or MKZ parameters for G/Gmax curves.
    xi_param : PySeismoSoil.class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        HH or MKZ parameters for damping curves.
    '''
    max_mat_num = np.max(vs_profile._material_number)
    if G_param.n_layer < max_mat_num:
        raise ValueError('Not enough sets of parameters in `G_param` for '
                         '`vs_profiles`.')
    if xi_param.n_layer < max_mat_num:
        raise ValueError('Not enough sets of parameters in `xi_param` for '
                         '`vs_profiles`.')

    return None
