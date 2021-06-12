import numpy as np

from . import helper_hh_calibration as hhc

from .class_Vs_profile import Vs_Profile
from .class_curves import Damping_Curve, Multiple_Damping_Curves


class Damping_Calibration:
    """
    A class to generate damping curves (and associated soil model parameters)
    from a given Vs profile.
    """
    def __init__(self, vs_profile):
        if not isinstance(vs_profile, Vs_Profile):
            raise TypeError('`vs_profile` must be of type Vs_Profile.')
        self.vs_profile = vs_profile

    def get_damping_curves(
            self, strain_in_pct=np.logspace(-3, 1),
            use_Darendeli_Dmin=False, show_fig=False,
    ):
        """
        Calculate damping curves using empirical formulas by Darendeli (2001).

        Parameters
        ----------
        strain_in_pct : numpy.ndarray
            Strain array. Must be a 1D numpy array. Unit: %
        use_Darendeli_Dmin : bool
            If ``True``, use the D_min values estimated by the empirical
            formulas in Darendeli (2001) as the small-strain damping ratios
            of each soil layer. If ``False``, use the damping ratio provided
            in ``vs_profile`` (or estimated using PySeismoSoil's built-in
            Vs-damping correlation).
        show_fig : bool
            Whether to show a figure of the damping curves of each layer.

        Returns
        -------
        mdc : PySeismoSoil.class_curves.Multiple_Damping_Curves
            Damping curves for all the soil layers (i.e., not including the
            rock halfspace at the bottom).
        """
        h = self.vs_profile.vs_profile[:-1, 0]
        Vs = self.vs_profile.vs_profile[:-1, 1]
        n_layer = len(Vs)

        if self.vs_profile.vs_profile.shape[1] == 5:  # there can only be 5 or 2 columns
            rho = self.vs_profile.vs_profile[:-1, 3]
        else:  # only 2 columns
            rho = hhc._calc_rho(h, Vs)

        sigma_v0 = hhc._calc_vertical_stress(h, rho)
        OCR = hhc._calc_OCR(Vs, rho, sigma_v0)
        PI = hhc._calc_PI(Vs)
        phi = 30
        _, xi, _ = hhc.produce_Darendeli_curves(
            sigma_v0, PI, OCR=OCR, K0=None, phi=phi, strain_in_pct=strain_in_pct,
        )
        assert(xi.shape[1] == n_layer)

        curve_list = []
        for j in range(n_layer):
            xi_j = xi[:, j]
            if not use_Darendeli_Dmin:
                xi_j -= xi_j[0]
                xi_j += self.vs_profile.vs_profile[j, 2]
            dc = Damping_Curve(
                np.column_stack((strain_in_pct, xi_j)),
                strain_unit='%', damping_unit='1',
                interpolate=False, check_values=True,
            )
            curve_list.append(dc)
        mdc = Multiple_Damping_Curves(curve_list)

        if show_fig:
            mdc.plot()

        return mdc

    def get_HH_x_param(self, **kwargs):
        """
        Obtain HH_x parameters for each layer (i.e., HH model parameters
        that best fit given damping values, for each layer).

        Parameters
        ----------
        kwargs :
            Keyword arguments to be passed to this method:
                PySeismoSoil.class_curves.Multiple_Damping_Curves.get_all_HH_x_params().
            Check its documentation for details:
                https://pyseismosoil.readthedocs.io/en/stable/api_docs/class_curves.html#PySeismoSoil.class_curves.Multiple_Damping_Curves.get_all_HH_x_params

        Returns
        -------
        HH_x_param : PySeismoSoil.class_parameters.HH_Param_Multi_Layer
            The best parameters for each soil layer found in the optimization.
        """
        mdc = self.get_damping_curves(
            strain_in_pct=np.geomspace(1e-4, 15, 100), show_fig=False,
        )
        HH_x_param = mdc.get_all_HH_x_params(**kwargs)
        return HH_x_param

    def get_H4_x_param(self, **kwargs):
        """
        Obtain H4_x parameters for each layer (i.e., MKZ model parameters
        that best fit given damping values, for each layer).

        Parameters
        ----------
        kwargs :
            Keyword arguments to be passed to this method:
                PySeismoSoil.class_curves.Multiple_Damping_Curves.get_all_H4_x_params().
            Check its documentation for details:
                https://pyseismosoil.readthedocs.io/en/stable/api_docs/class_curves.html#PySeismoSoil.class_curves.Multiple_Damping_Curves.get_all_H4_x_params

        Returns
        -------
        H4_x_param : PySeismoSoil.class_parameters.H4_Param_Multi_Layer
            The best parameters for each soil layer found in the optimization.
        """
        mdc = self.get_damping_curves(
            strain_in_pct=np.geomspace(1e-4, 15, 100), show_fig=False,
        )
        H4_x_param = mdc.get_all_HH_x_params(**kwargs)
        return H4_x_param
