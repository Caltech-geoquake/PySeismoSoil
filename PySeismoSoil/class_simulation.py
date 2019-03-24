# Author: Jian Shi

from . import helper_site_response as sr

from .class_ground_motion import Ground_Motion
from .class_Vs_profile import Vs_Profile

#%%============================================================================
class Simulator():
    '''
    Class implementatino of a site response simulator

    Parameters
    ----------
    input_motion : str or PySeismoSoil.class_ground_motion.Ground_Motion
        Input ground motion. Either a file name (as a str) containing time
        and acceleration in two columns, or a Ground_Motion object.
    soil_profile : str or PySeismoSoil.class_Vs_profile.Vs_Profile
        Soil profile. Either a file name (as a str) containing at least two
        columns (thickness and Vs), or a Vs_Profile object

    Attributes
    ----------
    input_motion : PySeismoSoil.class_ground_motion.Ground_Motion
        Input ground motion
    soil_profile : PySeismoSoil.class_Vs_profile.Vs_Profile
        Soil profile
    '''
    def __init__(self, input_motion, soil_profile):

        if isinstance(input_motion, str):
            input_motion = Ground_Motion(input_motion, 'm/s/s')
        if isinstance(soil_profile, str):
            soil_profile = Vs_Profile(soil_profile)

        if not isinstance(input_motion, Ground_Motion):
            raise TypeError('`input_motion` need to be an object of the '
                            '`Ground_Motion` class.')
        if not isinstance(soil_profile, Vs_Profile):
            raise TypeError('`soil_profile` need to be an object of the '
                            '`Vs_Profile` class.')

        self.input_motion = input_motion
        self.soil_profile = soil_profile

#%%============================================================================
class Linear_Simulator(Simulator):
    '''
    Linear site response simulator
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
