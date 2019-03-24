# Author: Jian Shi

from . import helper_site_response as sr

from .class_ground_motion import Ground_Motion
from .class_Vs_profile import Vs_Profile

#%%============================================================================
class Simulation():
    '''
    Class implementatino of a base site response simulation

    Parameters
    ----------
    input_motion : PySeismoSoil.class_ground_motion.Ground_Motion
        Input ground motion
    soil_profile : PySeismoSoil.class_Vs_profile.Vs_Profile
        Soil profile

    Attributes
    ----------
    input_motion : PySeismoSoil.class_ground_motion.Ground_Motion
        Input ground motion
    soil_profile : PySeismoSoil.class_Vs_profile.Vs_Profile
        Soil profile
    '''
    def __init__(self, input_motion, soil_profile):

        if not isinstance(input_motion, Ground_Motion):
            raise TypeError('`input_motion` need to be an object of the '
                            '`Ground_Motion` class.')
        if not isinstance(soil_profile, Vs_Profile):
            raise TypeError('`soil_profile` need to be an object of the '
                            '`Vs_Profile` class.')

        self.input_motion = input_motion
        self.soil_profile = soil_profile

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
