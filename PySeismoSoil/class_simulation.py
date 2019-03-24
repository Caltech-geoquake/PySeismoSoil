# Author: Jian Shi

from . import helper_site_response as sr
from . import helper_simulations as sim

from .class_ground_motion import Ground_Motion
from .class_Vs_profile import Vs_Profile
from .class_parameters import Param_Multi_Layer
from .class_curves import Multiple_Damping_Curves, Multiple_GGmax_Curves

#%%============================================================================
class Simulation():
    '''
    Class implementatino of a base site response simulation

    Parameters
    ----------
    input_motion : class_ground_motion.Ground_Motion
        Input ground motion
    soil_profile : class_Vs_profile.Vs_Profile
        Soil profile
    G_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        Parameters that describe the G/Gmax curves
    xi_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        Parameters that describe the damping curves
    GGmax_curves : class_curves.Multiple_GGmax_Curves
        G/Gmax curves
    xi_curves : class_curves.Multiple_Damping_Curves
        Damping curves

    Attributes
    ----------
    Same as the inputs
    '''
    def __init__(self, input_motion, soil_profile, G_param=None, xi_param=None,
                 GGmax_curves=None, xi_curves=None):

        if not isinstance(input_motion, Ground_Motion):
            raise TypeError('`input_motion` must be of class `Ground_Motion`.')
        if not isinstance(soil_profile, Vs_Profile):
            raise TypeError('`soil_profile` must be of class `Vs_Profile`.')

        if type(G_param) != type(xi_param):
            raise TypeError('`G_param` and `xi_param` must be of the same type.')
        if G_param is not None and not isinstance(G_param, Param_Multi_Layer):
            raise TypeError('`G_param` must be of a subclass of '
                            '`Param_Multi_Layer`, e.g., `HH_Param_Multi_Layer` '
                            'or `MKZ_Param_Multi_Layer`.')
        if xi_param is not None and not isinstance(xi_param, Param_Multi_Layer):
            raise TypeError('`xi_param` must be of a subclass of '
                            '`Param_Multi_Layer`, e.g., `HH_Param_Multi_Layer` '
                            'or `MKZ_Param_Multi_Layer`.')

        if GGmax_curves is not None and \
        not isinstance(GGmax_curves, Multiple_GGmax_Curves):
            raise TypeError('`GGmax_curves` must be a `Multiple_GGmax_Curves` '
                            'object.')
        if xi_curves is not None and \
        not isinstance(xi_curves, Multiple_Damping_Curves):
            raise TypeError('`xi_curves` must be a `Multiple_Damping_Curves` '
                            'object.')

        self.input_motion = input_motion
        self.soil_profile = soil_profile
        self.G_param = G_param
        self.xi_param = xi_param
        self.GGmax_curves = GGmax_curves
        self.xi_curves = xi_curves

#%%============================================================================
class Linear_Simulation(Simulation):
    '''
    Linear site response simulation
    '''
    def run(self, boundary='elastic', show_fig=False, deconv=False):
        '''
        Parameters
        ----------
        boundary : {'elastic', 'rigid'}
            Boundary condition. "Elastic" means that the boundary allows waves
            to propagate through. "Rigid" means that all downgoing waves are
            reflected back to the soil medium.
        show_fig : bool
            Whether to show a figure that shows the result of the analysis
        deconv : bool
            Whether this operation is deconvolution. If True, it means that the
            `input_motion` will be propagated downwards, and the motion at the
            bottom will be collected.

        Returns
        -------
        output_motion : PySeismoSoil.class_ground_motion.Ground_Motion
            The output ground motion
        '''
        response = sr.linear_site_resp(self.soil_profile.vs_profile,
                                       self.input_motion.accel,  # unit: m/s/s
                                       boundary=boundary, show_fig=show_fig,
                                       deconv=deconv)
        return Ground_Motion(response, 'm')  # because GM internally uses SI unit

#%%============================================================================
class Nonlinear_Simulation(Simulation):
    '''
    Nonlinear site response simulation
    '''
    def __init__(self, input_motion, soil_profile, G_param, xi_param,
                 use_HH_model=True):
        if G_param is None:
            raise TypeError('`G_param` cannot be None.')
        if xi_param is None:
            raise TypeError('`xi_param` cannot be None.')
        super(Nonlinear_Simulation, self).__init__(input_motion, soil_profile,
                                                   G_param=G_param,
                                                   xi_param=xi_param)
        sim.check_layer_count(soil_profile, G_param, xi_param)

    def run():
        # TODO: add codes
        pass
