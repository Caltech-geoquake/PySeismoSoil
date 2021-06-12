import os
import glob
import stat
import shutil
import subprocess
import numpy as np
import pkg_resources

from . import helper_generic as hlp
from . import helper_site_response as sr
from . import helper_simulations as sim
from . import helper_signal_processing as sig

from .class_ground_motion import Ground_Motion
from .class_Vs_profile import Vs_Profile
from .class_parameters import Param_Multi_Layer
from .class_curves import Multiple_GGmax_Damping_Curves
from .class_simulation_results import Simulation_Results
from .class_frequency_spectrum import Frequency_Spectrum


class Simulation:
    """
    Class implementation of a base site response simulation.

    Parameters
    ----------
    soil_profile : class_Vs_profile.Vs_Profile
        Soil profile.
    input_motion : class_ground_motion.Ground_Motion
        Input ground motion.
    boundary : {'elastic', 'rigid'}
        Boundary condition. 'Elastic' means that the input motion is the
        "rock outcrop" motion, and 'rigid' means that the input motion is
        the recorded motion at the bottom of the Vs profile.
    G_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        Parameters that describe the G/Gmax curves.
    xi_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        Parameters that describe the damping curves.
    GGmax_and_damping_curves : class_curves.Multiple_GGmax_Damping_Curves
        G/Gmax and damping curves.

    Attributes
    ----------
    Attributes same as the inputs
    """
    def __init__(
            self, soil_profile, input_motion, *, boundary='elastic',
            G_param=None, xi_param=None, GGmax_and_damping_curves=None,
    ):
        if not isinstance(soil_profile, Vs_Profile):
            raise TypeError('`soil_profile` must be of class `Vs_Profile`.')
        if not isinstance(input_motion, Ground_Motion):
            raise TypeError('`input_motion` must be of class `Ground_Motion`.')

        if boundary not in ['elastic', 'rigid']:
            raise ValueError('`boundary` should be "elastic" or "rigid".')

        if type(G_param) != type(xi_param):
            raise TypeError('`G_param` and `xi_param` must be of the same type.')
        if G_param is not None and not isinstance(G_param, Param_Multi_Layer):
            raise TypeError(
                '`G_param` must be of a subclass of '
                '`Param_Multi_Layer`, e.g., `HH_Param_Multi_Layer` '
                'or `MKZ_Param_Multi_Layer`.'
            )
        if xi_param is not None and not isinstance(xi_param, Param_Multi_Layer):
            raise TypeError(
                '`xi_param` must be of a subclass of '
                '`Param_Multi_Layer`, e.g., `HH_Param_Multi_Layer` '
                'or `MKZ_Param_Multi_Layer`.'
            )

        if (
            GGmax_and_damping_curves is not None and
            not isinstance(GGmax_and_damping_curves, Multiple_GGmax_Damping_Curves)
        ):
            raise TypeError(
                '`GGmax_and_damping_curves` must be a '
                '`Multiple_GGmax_Curves` object.'
            )

        self.input_motion = input_motion
        self.soil_profile = soil_profile
        self.boundary = boundary
        self.G_param = G_param
        self.xi_param = xi_param
        self.GGmax_and_damping_curves = GGmax_and_damping_curves


class Linear_Simulation(Simulation):
    """
    Linear site response simulation.

    Parameters
    ----------
    soil_profile : class_Vs_profile.Vs_Profile
        Soil profile.
    input_motion : class_ground_motion.Ground_Motion
        Input ground motion.
    boundary : {'elastic', 'rigid'}
        Boundary condition. 'Elastic' means that the input motion is the
        "rock outcrop" motion, and 'rigid' means that the input motion is
        the recorded motion at the bottom of the Vs profile.

    Attributes
    ----------
    Attributes same as the inputs
    """
    def run(self, every_layer=True, deconv=False, show_fig=False,
            save_fig=False, motion_name=None, save_txt=False,
            save_full_time_history=False, output_dir=None, verbose=True,
    ):
        """
        Parameters
        ----------
        every_layer : bool
            If ``True``, use the algorithm that can produce ground motion time
            histories of every soil layer. If ``False``, use a simpler and
            faster algorithm to produce the motion on the ground surface only.
        deconv : bool
            Whether this operation is deconvolution. If ``True``, it means
            that the ``input_motion`` will be propagated downwards, and the
            motion at the bottom will be collected. Only effective if
            ``every_layer`` is set to ``False``.
        show_fig : bool
            Whether to show figures of the simulation results.
        save_fig : bool
            Whether to save figures to ``output_dir``. Only effective when
            ``show_fig`` is set to ``True``.
        motion_name : str or ``None``
            Name of the input ground motion. For example, "Northridge". If not
            provided (i.e., ``None``), the current time stamp will be used.
        save_txt : bool
            Whether to save the results as text files to ``output_dir``.
        save_full_time_history : bool
            When saving simulation results, whether to save the full time
            histories (i.e., every time step, every depth) of the acceleration,
            velocity, displacement, stress, and strain. Only effective if
            ``every_layer`` is ``True``.
        output_dir : str
            Directory for saving the figures and/or result files.
        verbose : bool
            Whether to show simulation progress.

        Returns
        -------
        sim_results : Simulation_Results
            An object that contains all the simulation results.
        """
        if verbose:
            print('Linear site response simulation running... ', end='')

        if every_layer:
            results = sim.linear(
                self.soil_profile.vs_profile,
                self.input_motion.accel,
                boundary=self.boundary,
            )
            (
                new_profile, freq_array, tf, accel_on_surface, out_a, out_v,
                out_d, out_gamma, out_tau, max_avd, max_gt,
            ) = results

            sim_results = Simulation_Results(
                self.input_motion,
                Ground_Motion(accel_on_surface, unit='m'),
                Vs_Profile(new_profile, density_unit='g/cm^3'),
                max_a_v_d=max_avd,
                max_strain_stress=max_gt,
                trans_func=Frequency_Spectrum(tf, df=freq_array[1]-freq_array[0]),
                time_history_accel=out_a,
                time_history_veloc=out_v,
                time_history_displ=out_d,
                time_history_strain=out_gamma,
                time_history_stress=out_tau,
                motion_name=motion_name,
                output_dir=output_dir,
            )
            if show_fig:
                sim_results.plot(save_fig=save_fig, amplif_func_ylog=False)
            # END IF
        else:  # `every_layer` is `False`
            response, tf = sr.linear_site_resp(
                self.soil_profile.vs_profile,
                self.input_motion.accel,  # unit: m/s/s
                boundary=self.boundary,
                show_fig=show_fig,
                deconv=deconv,
            )
            trans_func = Frequency_Spectrum(tf[1], df=tf[0][1] - tf[0][0])
            sim_results = Simulation_Results(
                self.input_motion,
                Ground_Motion(response, unit='m'),
                self.soil_profile,
                trans_func=trans_func,
            )

        if save_txt:
            sim_results.to_txt(save_full_time_history=save_full_time_history)
        # END IF

        if verbose:
            print('done.')

        return sim_results


class Equiv_Linear_Simulation(Simulation):
    """
    Equivalent linear site response simulation.

    Parameters
    ----------
    soil_profile : class_Vs_profile.Vs_Profile
        Soil profile.
    input_motion : class_ground_motion.Ground_Motion
        Input ground motion.
    GGmax_and_damping_curves : class_curves.Multiple_GGmax_Damping_Curves
        G/Gmax and damping curves of every layer.
    boundary : {'elastic', 'rigid'}
        Boundary condition. 'Elastic' means that the input motion is the
        "rock outcrop" motion, and 'rigid' means that the input motion is
        the recorded motion at the bottom of the Vs profile.
    """
    def __init__(
            self, soil_profile, input_motion, GGmax_and_damping_curves,
            boundary='elastic',
    ):
        if GGmax_and_damping_curves is None:
            raise TypeError('`GGmax_and_damping_curves` cannot be None.')
        super(Equiv_Linear_Simulation, self).__init__(
            soil_profile,
            input_motion,
            GGmax_and_damping_curves=GGmax_and_damping_curves,
            boundary=boundary,
        )
        sim.check_layer_count(
            soil_profile, GGmax_and_damping_curves=GGmax_and_damping_curves,
        )

    def run(self, verbose=True, show_fig=False, save_fig=False,
            motion_name=None, save_txt=False, save_full_time_history=False,
            output_dir=None,
    ):
        """
        Start equivalent linear simulation.

        Parameters
        ----------
        verbose : bool
            Whether to print iteration progress on the console.
        show_fig : bool
            Whether to show figures of the simulation results (input and
            output motions, maximum accel/veloc/displ/strain/stress profiles)
        save_fig : bool
            Whether to save figures to ``output_dir``. Only effective when
            ``show_fig`` is set to ``True``.
        motion_name : str or ``None``
            Name of the input ground motion. For example, "Northridge". If not
            provided (i.e., ``None``), the current time stamp will be used.
        save_txt : bool
            Whether to save the results as text files to ``output_dir``.
        save_full_time_history : bool
            When saving simulation results, whether to save the full time
            histories (i.e., every time step, every depth) of the acceleration,
            velocity, displacement, stress, and strain.
        output_dir : str
            Directory for saving the figures and/or result files.

        Returns
        -------
        sim_results : Simulation_Results
            An object that contains all the simulation results.
        """
        vs_profile = self.soil_profile.vs_profile
        input_accel = self.input_motion.accel
        curve = self.GGmax_and_damping_curves.get_curve_matrix()

        results = sim.equiv_linear(
            vs_profile, input_accel, curve,
            boundary=self.boundary, verbose=verbose,
        )

        (
            new_profile, freq_array, tf, accel_on_surface, out_a, out_v,
            out_d, out_gamma, out_tau, max_avd, max_gt,
        ) = results

        sim_results = Simulation_Results(
            self.input_motion,
            Ground_Motion(accel_on_surface, unit='m'),
            Vs_Profile(new_profile, density_unit='g/cm^3'),
            max_a_v_d=max_avd,
            max_strain_stress=max_gt,
            trans_func=Frequency_Spectrum(tf, df=freq_array[1]-freq_array[0]),
            time_history_accel=out_a,
            time_history_veloc=out_v,
            time_history_displ=out_d,
            time_history_strain=out_gamma,
            time_history_stress=out_tau,
            motion_name=motion_name,
            output_dir=output_dir,
        )

        if show_fig:
            sim_results.plot(save_fig=save_fig, amplif_func_ylog=False)

        if save_txt:
            sim_results.to_txt(
                save_full_time_history=save_full_time_history, verbose=verbose,
            )

        return sim_results


class Nonlinear_Simulation(Simulation):
    """
    Nonlinear site response simulation.

    Parameters
    ----------
    soil_profile : class_Vs_profile.Vs_Profile
        Soil profile.
    input_motion : class_ground_motion.Ground_Motion
        Input ground motion.
    G_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        Parameters that describe the G/Gmax curves.
    xi_param : class_parameters.HH_Param_Multi_Layer or MKZ_Param_Multi_Layer
        Parameters that describe the damping curves.
    boundary : {'elastic', 'rigid'}
        Boundary condition. 'Elastic' means that the input motion is the
        "rock outcrop" motion, and 'rigid' means that the input motion is
        the recorded motion at the bottom of the Vs profile.

    Attributes
    ----------
    Attributes same as the inputs
    """
    def __init__(
            self, soil_profile, input_motion, *, G_param, xi_param,
            boundary='elastic',
    ):
        if G_param is None:
            raise TypeError('`G_param` cannot be None.')
        if xi_param is None:
            raise TypeError('`xi_param` cannot be None.')
        super(Nonlinear_Simulation, self).__init__(
            soil_profile, input_motion,
            G_param=G_param, xi_param=xi_param, boundary=boundary,
        )
        sim.check_layer_count(soil_profile, G_param=G_param, xi_param=xi_param)

    def run(self, sim_dir=None, motion_name=None, save_txt=False,
            save_full_time_history=True, show_fig=False, save_fig=False,
            remove_sim_dir=False, verbose=True,
    ):
        """
        Start nonlinear simulation.

        Parameters
        ----------
        sim_dir : str
            Directory for storing temporary input files and storing permenant
            output files/figures.
        motion_name : str or ``None``
            Name of the input ground motion. For example, "Northridge". If not
            provided (i.e., ``None``), the current time stamp will be used.
        save_txt : bool
            Whether to save the simulation results as text files to ``sim_dir``.
        save_full_time_history : bool
            When saving simulation results, whether to save the full time
            histories (i.e., every time step, every depth) of the acceleration,
            velocity, displacement, stress, and strain.
        show_fig : bool
            Whether to show figures of the simulation results (input and
            output motions, maximum accel/veloc/displ/strain/stress profiles)
        save_fig : bool
            Whether to save figures to ``sim_dir``. Only effective when
            ``show_fig`` is set to ``True``.
        remove_sim_dir : bool
            Whether to remove ``sim_dir`` from the hard drive after simulations,
            only effective when ``save_txt`` and ``save_fig`` are both set to
            ``False``.
        verbose : bool
            Whether to show simulation progress on the console.

        Returns
        -------
        sim_results : Simulation_Results
            An object that contains all the simulation results.
        """
        if verbose:
            print('Nonlinear simulation running...')

        # Mapping from user input to Fortran kernel:
        #
        #   User input   |  MATLAB SeismoSoil  |  `n_bound` for Fortran
        # ---------------+---------------------+ ------------------------
        #     rigid      |       borehole      |          3
        #    elastic     |       elastic       |          1
        #      N/A (*)   |        rigid        |          2
        #
        # "n_bound = 2" option shall not be exposed to PySeismoSoil users.
        n_bound = 3 if self.boundary == 'rigid' else 1

        if sim_dir is None:
            current_time = hlp.get_current_time(for_filename=True)
            sim_dir = './nonlinear_sim_%s' % current_time

        if os.path.exists(sim_dir):
            sim_dir += '_'
        os.makedirs(sim_dir)
        os.chmod(sim_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

        f_max = 30  # maximum frequency modeled, unit is Hz
        ppw = 10  # points per wavelength
        n_dt = 30  # number of sub-steps in one time step
        N_spr = 120  # number of Iwan springs
        N_obs = 50  # number of strain points in a curve
        n_ma = self.G_param.n_layer
        strain_in_pct = np.geomspace(0.0001, 6, num=N_obs)

        # Three coefficients (tau, alpha, beta) for modeling Q, from Table 1
        # of Liu and Archuleta (2006) BSSA
        tabk = np.array([
            [1.72333E-03, 1.66958E-02, 8.98758E-02],
            [1.80701E-03, 3.81644E-02, 6.84635E-02],
            [5.38887E-03, 9.84666E-03, 9.67052E-02],
            [1.99322E-02, -1.36803E-02, 1.20172E-01],
            [8.49833E-02, -2.85125E-02, 1.30728E-01],
            [4.09335E-01, -5.37309E-02, 1.38746E-01],
            [2.05951E+00, -6.65035E-02, 1.40705E-01],
            [1.32629E+01, -1.33696E-01, 2.14647E-01],
        ])

        #--------- Re-discretize Vs profile -----------------------------------
        new_profile = sr.stratify(self.soil_profile.vs_profile)
        new_profile[:, 3] /= 1000.0  # convert to g/cm3 to pass to NLHH

        n_layer = new_profile.shape[0] - 1  # exclude bedrock

        input_accel = self.input_motion.accel.copy()
        # On 05/26/2019, confirmed with MATLAB SeismoSoil that this is correct:
        if self.boundary == 'elastic':
            input_accel[:, 1] /= 2  # convert to incident motion

        t = input_accel[:, 0]
        nt_out = len(t)

        #--------- Create a dummy "curves" for Fortran ------------------------
        mgc, mdc = self.G_param.construct_curves(strain_in_pct=strain_in_pct)
        mgdc = Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mgc, mdc))
        curves = mgdc.get_curve_matrix()

        #--------- Prepare tabk.dat file --------------------------------------
        if hlp.detect_OS() == 'Windows':
            exec_ext = 'exe'
        elif hlp.detect_OS() == 'Darwin':
            exec_ext = 'mac'
        elif hlp.detect_OS() == 'Linux':
            exec_ext = 'unix'
        else:
            raise ValueError('Unknown operating system.')
        dir_exec_files = pkg_resources.resource_filename(
            'PySeismoSoil', 'exec_files',
        )
        shutil.copy(os.path.join(dir_exec_files, 'NLHH.%s' % exec_ext), sim_dir)
        np.savetxt(os.path.join(sim_dir, 'tabk.dat'), tabk, delimiter='\t')

        #-------- Prepare control.dat file ------------------------------------
        with open(os.path.join(sim_dir, 'control.dat'), 'w') as fp:
            fp.write(
                '%6.1f %6.0f %6.0f %6.0f %6.0f %10.0f %6.0f %6.0f %6.0f' \
                % (f_max, ppw, n_dt, n_bound, n_layer, nt_out, n_ma, N_spr, N_obs)
            )

        #-------- Write data to files for the Fortran kernel to read ----------
        np.savetxt(os.path.join(sim_dir, 'profile.dat'), new_profile)
        np.savetxt(os.path.join(sim_dir, 'incident.dat'), input_accel)
        np.savetxt(os.path.join(sim_dir, 'curve.dat'), curves)
        np.savetxt(os.path.join(sim_dir, 'HH_G.dat'), self.G_param.serialize_to_2D_array())
        np.savetxt(os.path.join(sim_dir, 'HH_x.dat'), self.xi_param.serialize_to_2D_array())

        #------- Execute Fortran kernel ---------------------------------------
        cwd = os.getcwd()
        os.chdir(sim_dir)
        if hlp.detect_OS() == 'Windows':
            subprocess.run('NLHH.exe')
        elif hlp.detect_OS() == 'Darwin':
            current_status = os.stat('NLHH.mac').st_mode
            os.chmod(
                'NLHH.mac',
                current_status | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
            subprocess.run('./NLHH.mac', stdout=True)
        elif hlp.detect_OS() == 'Linux':
            current_status = os.stat('NLHH.unix').st_mode
            os.chmod(
                'NLHH.unix',
                current_status | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
            subprocess.run('./NLHH.unix', stdout=True)
        else:
            raise ValueError('Unknown operating system.')

        if verbose:
            print('Simulation finished. Now post processing.')

        #------------ Post-process files --------------------------------------
        layer_boundary_depth = np.genfromtxt('node_depth.dat').T
        layer_midpoint_depth = np.genfromtxt('layer_depth.dat').T
        out_a = np.genfromtxt('out_a.dat')
        out_v = np.genfromtxt('out_v.dat')
        out_d = np.genfromtxt('out_d.dat')
        out_gamma = np.genfromtxt('out_gamma.dat')
        out_tau = np.genfromtxt('out_tau.dat')

        dat_files = glob.glob('*.dat')
        for dat_file in dat_files:
            os.remove(dat_file)

        if hlp.detect_OS() == 'Windows':
            os.remove('NLHH.exe')
        elif hlp.detect_OS() == 'Darwin':
            os.remove('NLHH.mac')
        elif hlp.detect_OS() == 'Linux':
            os.remove('NLHH.unix')
        else:
            raise ValueError('Unknown operating system.')

        max_a = np.max(np.abs(out_a), axis=0).T  # max of every column (i.e., layer)
        max_v = np.max(np.abs(out_v), axis=0).T
        max_d = np.max(np.abs(out_d), axis=0).T
        max_gamma = np.max(np.abs(out_gamma), axis=0).T
        max_tau = np.max(np.abs(out_tau), axis=0).T

        max_avd = np.column_stack((layer_boundary_depth, max_a, max_v, max_d))
        max_gt = np.column_stack((layer_midpoint_depth, max_gamma, max_tau))

        accel_surface = out_a[:, 0]
        accel_surface_2col = np.column_stack((t, accel_surface))
        tf_unsmoothed = sig.calc_transfer_function(
            self.input_motion.accel, accel_surface_2col, amplitude_only=False,
        )
        tf_smoothed = sig.calc_transfer_function(
            self.input_motion.accel, accel_surface_2col,
            amplitude_only=True, smooth_signal=True,
        )
        os.chdir(cwd)

        #------------ Create sim_results object and plot and/or save ----------
        sim_results = Simulation_Results(
            self.input_motion,
            Ground_Motion(accel_surface_2col, unit='m'),
            Vs_Profile(new_profile, density_unit='g/cm^3'),
            max_a_v_d=max_avd,
            max_strain_stress=max_gt,
            trans_func=Frequency_Spectrum(tf_unsmoothed),
            trans_func_smoothed=Frequency_Spectrum(tf_smoothed),
            time_history_accel=out_a,
            time_history_veloc=out_v,
            time_history_displ=out_d,
            time_history_strain=out_gamma,
            time_history_stress=out_tau,
            motion_name=motion_name,
            output_dir=sim_dir,
        )

        if verbose:
            print('Done.')

        if show_fig:
            sim_results.plot(save_fig=save_fig, amplif_func_ylog=True)

        if save_txt:
            sim_results.to_txt(
                save_full_time_history=save_full_time_history, verbose=verbose,
            )

        if not save_txt and not save_fig and remove_sim_dir:
            os.removedirs(sim_dir)
            if verbose:
                print('`sim_dir` (%s) removed.' % sim_dir)

        return sim_results
