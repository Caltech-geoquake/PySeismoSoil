from .class_Vs_profile import Vs_Profile
from .class_curves import Multiple_GGmax_Curves
from .class_parameters import HH_Param_Multi_Layer

from . import helper_generic as hlp
from . import helper_hh_calibration as hhc

class HH_Calibration:
    """
    Class implementation of the "HH calibration procedure" (HHC procedure). The
    HHC procedure generates parameters of each soil layer for the HH model.
    The users can provide only a shear-wave velocity (Vs) profile, or they can
    also provide pre-defined G/Gmax curves for the soil layers, if they have
    such laboratory measurements.

    For more information, refer to the following paper:
        J. Shi and D. Asimaki (2017) "From stiffness to strength: Formulation
        and validation of a hybrid hyperbolic nonlinear soil model for
        site‐response analyses." Bulletin of the Seismological Society of
        America. 107 (3), 1336-1355.

    Parameters
    ----------
    vs_profile : PySeismoSoil.class_Vs_profile.Vs_Profile
        The Vs profile of interest.
    GGmax_curves : PySeismoSoil.class_curves.Multiple_GGmax_Curves or ``None``
        The G/Gmax curves of each layer. If ``None``, HH parameters will be
        determined from the Vs profile alone. If the user supplies this
        parameter, it will be used to calibrate the MKZ model, which eventually
        goes into calibrating the HH parameters.
    Tmax_profile : numpy.ndarray or ``None``
        The profile of shear strength of each soil layer (not including the
        rock half space at the bottom). If ``None``, it will be determined
        using the empirical formula by Ladd (1991).

    Attributes
    ----------
    vs_profile : PySeismoSoil.class_Vs_profile.Vs_Profile
        Same as the input parameter.
    GGmax_curves : PySeismoSoil.class_curves.Multiple_GGmax_Curves or ``None``
        Same as the input parameter.
    Tmax_profile : numpy.ndarray or ``None``
        Same as the input parameter.
    """
    def __init__(self, vs_profile, *, GGmax_curves=None, Tmax_profile=None):
        if not isinstance(vs_profile, Vs_Profile):
            raise TypeError('`vs_profile` must be of type Vs_Profile.')
        if GGmax_curves is not None:
            if not isinstance(GGmax_curves, Multiple_GGmax_Curves):
                raise TypeError(
                    'If `GGmax_curves` is not `None`, it must be '
                    'of type Multiple_GGmax_Curves.'
                )
            if GGmax_curves.n_layer != vs_profile.n_layer:
                raise ValueError(
                    'The number of layers implied in `GGmax_curves` '
                    'and `vs_profile` must be the same.'
                )
        if Tmax_profile is not None:
            hlp.assert_1D_numpy_array(Tmax_profile, '`Tmax_profile`')
            if len(Tmax_profile) != vs_profile.n_layer:
                raise ValueError(
                    'The length of `Tmax_profile` needs to '
                    'equal to the number of layers (not including '
                    'the rock half space) in `vs_profile`.'
                )
        self.vs_profile = vs_profile
        self.GGmax_curves = GGmax_curves
        self.Tmax_profile = Tmax_profile

    def fit(self, show_fig=False, save_fig=False, fig_output_dir=None,
            save_HH_G_file=False, HH_G_file_dir=None, profile_name=None,
            verbose=True,
    ):
        """
        Calculate the HH parameters with the given Vs profile and/or G/Gmax
        curves.

        Parameters
        ----------
        show_fig : bool
            Whether or not to show figures G/Gmax and stress-strain curves of
            MKZ, FKZ, and HH for each layer.
        save_fig : bool
            Whether or not to save the figures to the hard drive. Only effective
            if ``show_fig`` is set to ``True``.
        fig_output_dir : str
            The output directory for the figures. Only effective if ``show_fig``
            and ``save_fig`` are both ``True``.
        save_HH_G_file : bool
            Whether or not to save the HH parameters to the hard drive (as a
            "HH_G" file).
        HH_G_file_dir : str
            The output directory for the "HH_G" file. Only effective if
            ``save_HH_G_file`` is ``True``.
        profile_name : str or ``None``
            The name of the Vs profile, such as "CE.12345". If ``None``, a
            string of current date and time will be used as the profile name.
        verbose : bool
            Whether or not to print progresses on the console.

        Returns
        -------
        HH_G_param : PySeismoSoil.class_parameters.HH_Param_Multi_Layer
            The HH parameters of each layer.
        """
        vs_profile = self.vs_profile.vs_profile
        options = dict(
            Tmax=self.Tmax_profile, show_fig=show_fig,
            save_fig=save_fig, fig_output_dir=fig_output_dir,
            save_HH_G_file=save_HH_G_file, HH_G_file_dir=HH_G_file_dir,
            profile_name=profile_name, verbose=verbose,
        )
        if self.GGmax_curves is None:
            HH_G_param_ = hhc.hh_param_from_profile(vs_profile, **options)
        else:
            curves = self.GGmax_curves.get_curve_matrix()
            HH_G_param_ = hhc.hh_param_from_curves(vs_profile, curves, **options)

        HH_G_param = HH_Param_Multi_Layer(HH_G_param_)
        return HH_G_param
