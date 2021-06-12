import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .class_ground_motion import Ground_Motion
from .class_Vs_profile import Vs_Profile
from .class_frequency_spectrum import Frequency_Spectrum

from . import helper_generic as hlp
from . import helper_site_response as sr


class Simulation_Results:
    """
    Site response simulation results: output ground motion, transfer function,
    acceleration/velocity/displacement time histories (of every layer).

    Parameters
    ----------
    input_accel : Ground_Motion
        Input ground motion.
    accel_on_surface : Ground_Motion
        Output ground motion.
    rediscretized_profile : Vs_Profile
        Vs profile (the re-discretized version that ensures proper
        representation of wave shapes).
    max_a_v_d : numpy.ndarray
        Maximum acceleration, velocity, displacement (during ground shaking)
        at all layer boundaries.
    max_strain_stress : numpy.ndarray
        Maximum strain and stress (during ground shaking) at layer midpoints.
    trans_func : Frequency_Spectrum
        Transfer function (between the output and input motions). It can
        be complex-valued or real-valued (i.e., amplitudes only).
    trans_func_smoothed : Frequency_Spectrum or ``None``
        The smoothed transfer function (between the output and input motions).
        It is by default real-valued (i.e., amplitudes only).
    time_history_accel : numpy.ndarray
        Time histories of accelerations of all layers (at layer boundaries).
    time_history_veloc : numpy.ndarray
        Time histories of velocities of all layers (at layer boundaries).
    time_history_displ : numpy.ndarray
        Time histories of displacements of all layers (at layer boundaries).
    time_history_stress : numpy.ndarray
        Time histories of shear stresses of all layers (at layer midpoints).
    time_history_strain : numpy.ndarray
        Time histories of shear strains of all layers (at layer midpoints).
    motion_name : str or ``None``
        The name of the input motion to be used as an identifier in the
        file names. If ``None``, the current time stamp will used.
    output_dir : str or ``None``
        Directory to which to save the output files. If ``None``, the current
        time stamp will be used.

    Attributes
    ----------
    Attributes same as inputs
    """
    def __init__(
            self,
            input_accel,
            accel_on_surface,
            rediscretized_profile,
            *,
            max_a_v_d=None,
            max_strain_stress=None,
            trans_func=None,
            trans_func_smoothed=None,
            time_history_accel=None,
            time_history_veloc=None,
            time_history_displ=None,
            time_history_stress=None,
            time_history_strain=None,
            motion_name=None,
            output_dir=None,
    ):
        if not isinstance(input_accel, Ground_Motion):
            raise TypeError('`input_accel` needs to be of Ground_Motion type.')
        if not isinstance(accel_on_surface, Ground_Motion):
            raise TypeError('`accel_on_surface` needs to be of Ground_Motion type.')
        if not isinstance(rediscretized_profile, Vs_Profile):
            raise TypeError('`rediscretized_profile` needs to be of Vs_Profile type.')
        if not isinstance(trans_func, (Frequency_Spectrum, type(None))):
            raise TypeError(
                '`trans_func` needs to be either None or of Frequency_Spectrum type.'
            )
        if not isinstance(trans_func_smoothed, (Frequency_Spectrum, type(None))):
            raise TypeError(
                '`trans_func_smoothed` should be either None or of Frequency_Spectrum type.'
            )

        n_layer = rediscretized_profile.n_layer
        n_time_pts = accel_on_surface.npts

        if time_history_accel is not None:
            hlp.assert_2D_numpy_array(time_history_accel, '`time_history_accel`')
            assert(time_history_accel.shape == (n_time_pts, n_layer + 1))
        if time_history_veloc is not None:
            hlp.assert_2D_numpy_array(time_history_veloc, '`time_history_veloc`')
            assert(time_history_veloc.shape == (n_time_pts, n_layer + 1))
        if time_history_displ is not None:
            hlp.assert_2D_numpy_array(time_history_displ, '`time_history_displ`')
            assert(time_history_displ.shape == (n_time_pts, n_layer + 1))
        if time_history_stress is not None:
            hlp.assert_2D_numpy_array(time_history_stress, '`time_history_stress`')
            assert(time_history_stress.shape == (n_time_pts, n_layer))
        if time_history_strain is not None:
            hlp.assert_2D_numpy_array(time_history_strain, '`time_history_strain`')
            assert(time_history_strain.shape == (n_time_pts, n_layer))

        if max_a_v_d is not None and max_strain_stress is not None:
            hlp.assert_2D_numpy_array(max_a_v_d, '`max_a_v_d`')
            hlp.assert_2D_numpy_array(max_strain_stress, '`max_strain_stress`')
            assert(max_a_v_d.shape == (n_layer + 1, 4))
            assert(max_strain_stress.shape == (n_layer, 3))
        else:  # only when both are not `None` do we consider using them
            max_a_v_d = None
            max_strain_stress = None

        current_time = hlp.get_current_time(for_filename=True)
        if motion_name is None:
            motion_name = 'accel_%s' % current_time
        if output_dir is None:
            output_dir = os.path.join('./', 'sim_%s' % current_time)

        self.input_accel = input_accel
        self.accel_on_surface = accel_on_surface
        self.rediscretized_profile = rediscretized_profile
        self.max_a_v_d = max_a_v_d
        self.max_strain_stress = max_strain_stress
        self.trans_func = trans_func
        self.trans_func_smoothed = trans_func_smoothed
        self.time_history_accel = time_history_accel
        self.time_history_veloc = time_history_veloc
        self.time_history_displ = time_history_displ
        self.time_history_stress = time_history_stress
        self.time_history_strain = time_history_strain
        self.motion_name = motion_name
        self.output_dir = output_dir

    def plot(
            self,
            dpi=100,
            save_fig=False,
            amplif_func_ylog=True,
            output_dir=None,
    ):
        """
        Plots simulation results: output vs input motions, transfer functions
        and maximum acceleration, velocity, displacement, strain, and stress
        profiles.

        Parameters
        ----------
        dpi : float
            Figure resolution.
        save_fig : bool
            Whether to save figure to ``output_dir``.
        amplif_func_ylog : bool
            Whether to show the Y axis of the amplification function in log
            scale.
        output_dir : str
            The directory to save the plots. This overrides the ``output_dir``
            parameter when constructing the this class.

        Returns
        -------
        figs : list
            A list of three figure objects.
        axes : list
            A list of axes objects (or axes lists, if multiple subplots).
        """
        #-------- Plot output/input motions and transfer functions ------------
        accel_in = self.input_accel.accel
        accel_out = self.accel_on_surface.accel
        if self.trans_func is not None:
            freq = self.trans_func.freq
            ampl_func = self.trans_func.amplitude
            phase_func = self.trans_func.phase
        else:
            freq = None
            ampl_func = None
            phase_func = None
        if self.trans_func_smoothed is not None:
            ampl_func_smoothed = self.trans_func_smoothed.amplitude
        else:
            ampl_func_smoothed = None

        fig1, axes1 \
            = sr._plot_site_amp(accel_in, accel_out, freq, ampl_func,
                                amplif_func_1col_smoothed=ampl_func_smoothed,
                                phase_func_1col=phase_func, dpi=dpi,
                                amplif_func_ylog=amplif_func_ylog)
        axes1[0].set_ylabel('Accel. [m/s/s]')

        #-------- Plot maximum accel/veloc/displ/strain/stress profiles -------
        if self.max_a_v_d is not None and self.max_strain_stress is not None:
            max_layer_boundary_depth = np.max(self.max_a_v_d[:, 0])

            fig2 = plt.figure(figsize=(8.5, 5.5), dpi=dpi)

            ax21 = plt.subplot(151)
            plt.plot(self.max_a_v_d[:, 1], self.max_a_v_d[:, 0], ls='-', marker='.')
            plt.ylim(max_layer_boundary_depth, 0)
            plt.xlabel('Max. accel. [m/s/s]')
            plt.ylabel('Depth [m]')
            plt.grid(ls=':', lw=0.5)
            ax21.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=4, nbins='auto'))

            ax22 = plt.subplot(152)
            plt.plot(self.max_a_v_d[:, 2]*100, self.max_a_v_d[:, 0], ls='-', marker='.')
            plt.ylim(max_layer_boundary_depth, 0)
            plt.xlabel('Max. veloc. [cm/s]')
            plt.grid(ls=':', lw=0.5)
            ax22.get_yaxis().set_ticklabels([])
            ax22.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=4, nbins='auto'))

            ax23 = plt.subplot(153)
            plt.plot(self.max_a_v_d[:, 3]*100, self.max_a_v_d[:, 0], ls='-', marker='.')
            plt.ylim(max_layer_boundary_depth, 0)
            plt.xlabel('Max. displ. [cm]')
            plt.grid(ls=':', lw=0.5)
            ax23.get_yaxis().set_ticklabels([])
            ax23.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=4, nbins='auto'))

            ax24 = plt.subplot(154)
            plt.plot(self.max_strain_stress[:, 1]*100, self.max_strain_stress[:, 0],
                     ls='-', marker='.')
            plt.ylim(max_layer_boundary_depth, 0)
            plt.xlabel('$\gamma_{\max}$ [%]')
            plt.grid(ls=':', lw=0.5)
            ax24.get_yaxis().set_ticklabels([])
            ax24.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=4, nbins='auto'))

            ax25 = plt.subplot(155)
            plt.plot(self.max_strain_stress[:, 2]/1000., self.max_strain_stress[:, 0],
                     ls='-', marker='.')
            plt.ylim(max_layer_boundary_depth, 0)
            plt.xlabel(r'$\tau_{\max}$ [kPa]')
            plt.grid(ls=':', lw=0.5)
            ax25.get_yaxis().set_ticklabels([])
            ax25.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=4, nbins='auto'))

            plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.3)
        else:
            fig2 = None
            ax21, ax22, ax23, ax24, ax25 = None, None, None, None, None

        figs = [fig1, fig2]
        axes = [axes1, [ax21, ax22, ax23, ax24, ax25]]

        if save_fig:
            if output_dir is None:
                output_dir = self.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            fn_fig1 = os.path.join(output_dir,
                                   '%s_ground_motions.png' % self.motion_name)
            fn_fig2 = os.path.join(self.output_dir,
                                   '%s_max_profiles.png' % self.motion_name)
            fig1.savefig(fn_fig1, dpi=dpi, bbox_inches='tight')
            if fig2 is not None:
                fig2.savefig(fn_fig2, dpi=dpi, bbox_inches='tight')

        return figs, axes

    def to_txt(self, save_full_time_history=True, verbose=False, output_dir=None):
        """
        Save simulation results (output time history, transfer function, the
        profile of maximum acceleration/velocity/displacement/stress/train, etc.)
        as text files to the hard drive.

        Parameters
        ----------
        save_full_time_history : bool
            Whether to save full time histories (every time step, every layer)
            of accel/veloc/displ/strain/stress to hard drive. They can take
            a lot of disk space. Only effective if the full time histories
            are not ``None``.
        verbose : bool
            Whether to show on the console where the files are saved to.
        output_dir : str
            The directory to save the files. This overrides the ``output_dir``
            parameter when constructing the this class.
        """
        if output_dir is None:
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        od = output_dir  # shorten the variable name
        motion_name = self.motion_name

        fn_TF_raw = os.path.join(od, '%s_nonlinear_TF_raw.txt' % motion_name)
        fn_TF_smoothed = os.path.join(od, '%s_nonlinear_TF_smoothed.txt' % motion_name)
        fn_surface_accel = os.path.join(od, '%s_accel_on_surface.txt' % motion_name)
        fn_new_profile = os.path.join(od, '%s_re-discretized_profile.txt' % motion_name)
        fn_out_a = os.path.join(od, '%s_time_history_accel.txt' % motion_name)
        fn_out_v = os.path.join(od, '%s_time_history_veloc.txt' % motion_name)
        fn_out_d = os.path.join(od, '%s_time_history_displ.txt' % motion_name)
        fn_out_gamma = os.path.join(od, '%s_time_history_strain.txt' % motion_name)
        fn_out_tau = os.path.join(od, '%s_time_history_stress.txt' % motion_name)
        fn_max_avd = os.path.join(od, '%s_max_a_v_d.txt' % motion_name)
        fn_max_gt = os.path.join(od, '%s_max_gamma_tau.txt' % motion_name)

        fmt_dict = dict(delimiter='\t', fmt='%.6g')

        np.savetxt(fn_surface_accel, self.accel_on_surface.accel, **fmt_dict)

        np.savetxt(fn_new_profile, self.rediscretized_profile.vs_profile, **fmt_dict)
        if self.max_a_v_d is not None:
            np.savetxt(fn_max_avd, self.max_a_v_d, **fmt_dict)
        if self.max_strain_stress is not None:
            np.savetxt(fn_max_gt, self.max_strain_stress, **fmt_dict)

        if save_full_time_history:
            if self.time_history_accel is not None:
                np.savetxt(fn_out_a, self.time_history_accel, **fmt_dict)
            if self.time_history_veloc is not None:
                np.savetxt(fn_out_v, self.time_history_veloc, **fmt_dict)
            if self.time_history_displ is not None:
                np.savetxt(fn_out_d, self.time_history_displ, **fmt_dict)
            if self.time_history_strain is not None:
                np.savetxt(fn_out_gamma, self.time_history_strain, **fmt_dict)
            if self.time_history_stress is not None:
                np.savetxt(fn_out_tau, self.time_history_stress, **fmt_dict)

        if self.trans_func is not None:
            np.savetxt(fn_TF_raw, self.trans_func.spectrum_2col, **fmt_dict)

        if self.trans_func_smoothed is not None:
            np.savetxt(fn_TF_smoothed, self.trans_func_smoothed.amplitude_2col,
                       **fmt_dict)

        if verbose:
            print('Simulation results saved to %s' % od)

        return None
