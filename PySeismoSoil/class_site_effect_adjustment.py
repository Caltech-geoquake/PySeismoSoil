import numpy as np

from PySeismoSoil import helper_site_response as sr

from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_site_factors import Site_Factors


class Site_Effect_Adjustment:
    """
    Adjusts rock-outcrop ground motions by applying site effect adjustment
    using the SAG19 site factors.

    Parameters
    ----------
    input_motion : PySeismoSoil.class_ground_motion.Ground_Motion
        Input ground motion.
    Vs30_in_meter_per_sec : float
        Vs30 values in SI unit.
    z1_in_m : float
        z1 (basin depth) in meters. If ``None``, it will be estimated from
        Vs30 using an empirical correlation (see `calc_z1_from_Vs30()`
        function in `helper_site_response.py`).
    ampl_method : {'nl_hh', 'eq_hh'}
        Which site response simulation method was used to calculate the
        amplification factors. 'nl_hh' uses the results from nonlinear site
        response simulation, which is recommended.

    Attributes
    ----------
    input_motion : PySeismoSoil.class_ground_motion.Ground_Motion
        Input ground motion.
    Vs30 : float
        Vs30 of the site. (Unit: m/s)
    z1 : float
        z1 (basin depth) of the site. (Unit: m/s)
    PGA_in_g : float
        Peak ground acceleration of the input motion. (Unit: g)
    """

    def __init__(
            self,
            input_motion,
            Vs30_in_meter_per_sec,
            z1_in_m=None,
            ampl_method='nl_hh',
            lenient=False,
    ):
        if not isinstance(input_motion, Ground_Motion):
            raise TypeError('`input_motion` must be of class `Ground_Motion`.')
        if not isinstance(Vs30_in_meter_per_sec, (int, float, np.number)):
            msg = '`Vs30_in_meter_per_sec` must be int, float, or numpy.number.'
            raise TypeError(msg)
        if not isinstance(z1_in_m, (int, float, np.number, type(None))):
            msg = '`z1_in_m` must be int, float, numpy.number, or None.'
            raise TypeError(msg)
        if ampl_method not in {'nl_hh', 'eq_hh'}:
            raise ValueError("Currently, only 'nl_hh' and 'eq_hh' are valid.")

        if z1_in_m is None:
            z1_in_m = sr.calc_z1_from_Vs30(Vs30_in_meter_per_sec)
        PGA_in_g = input_motion.pga_in_g

        site_factor = Site_Factors(
            Vs30_in_meter_per_sec,
            z1_in_m,
            PGA_in_g,
            lenient=lenient,
        )

        self.input_motion = input_motion
        self.Vs30 = Vs30_in_meter_per_sec
        self.z1 = z1_in_m
        self.PGA_in_g = PGA_in_g
        self.site_factor = site_factor
        self._lenient = lenient
        self._ampl_method = ampl_method

    def run(self, show_fig=False, return_fig_obj=False, **kwargs_to_plot):
        """
        Run the site effect adjustment by querying the SAG19 site factors.

        Parameters
        ----------
        show_fig : bool
            Whether to show a figure demonstrating how the adjustment
            works.
        return_fig_obj : bool
            Whether to return the figure and axes objects.
        **kwargs_to_plot :
            Keyword arguments to pass to ``matplotlib.pyplot.plot()``.

        Returns
        -------
        output_motion : PySeismoSoil.class_ground_motion.Ground_Motion
            Output ground motion with site effects included.
        fig : matplotlib.figure.Figure, optional
            The figure object.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            The axes object.
        """
        sf = self.site_factor
        af = sf.get_amplification(method=self._ampl_method, Fourier=True)
        phf = sf.get_phase_shift(method='eq_hh')  # only `eq_hh` is valid

        if not np.allclose(af.freq, phf.freq):
            print(
                'Warning in Site_Effect_Adjustment.run(): the frequency '
                'arrays of the amplification factor '
                'and the phase factor are not identical---something may '
                'be wrong in class_site_factors.py.',
            )
        if af.iscomplex:
            print(
                'Warning in Site_Effect_Adjustment.run(): the '
                'amplification factor is complex, rather than '
                'real---something may be wrong in class_site_factors.py',
            )
        if phf.iscomplex:
            print(
                'Warning in Site_Effect_Adjustment.run(): the phase '
                'factor is complex, rather than '
                'real---something may be wrong in class_site_factors.py',
            )

        freq = af.freq
        amp_tf = af.spectrum
        phase_tf = phf.spectrum

        accel_in = self.input_motion.accel  # acceleration in m/s/s
        result = sr.amplify_motion(
            accel_in,
            (freq, (amp_tf, phase_tf)),
            show_fig=show_fig,
            return_fig_obj=show_fig,
            extrap_tf=True,
            **kwargs_to_plot,
        )
        if show_fig:
            accel_out, fig, ax = result
            ax[0].set_ylabel('Accel. [m/s/s]')
            ax[0].set_title(
                '$V_{S30}$=%.1fm/s, $z_1$=%.1fm, '
                r'$\mathrm{PGA}_{\mathrm{input}}$=%.3g$g$'
                % (self.Vs30, self.z1, self.PGA_in_g),
            )
            ax[1].set_ylabel('Amplif. factor')
            ax[2].set_ylabel('Phase factor [rad]')
        else:
            accel_out = result

        output_motion = Ground_Motion(accel_out, unit='m')
        if return_fig_obj:
            if not show_fig:
                fig, ax = None, None
            return output_motion, fig, ax
        else:
            return output_motion
