"""
Hierarchy of subroutines::

 hh_param_from_profile
   |--- _calc_rho()
           |--- helper_site_response.thk2dep()
   |--- _calc_Gmax()
   |--- _calc_vertical_stress()
   |--- _calc_OCR()
   |--- _calc_K0()
   |--- _calc_PI()
   |--- _calc_shear_strength()
           |--- _calc_K0()
   |--- produce_HH_G_param()
           |---  _calc_mean_confining_stress()
           |---  produce_Darendeli_curves()
                      |--- helper_generic.assert_1D_numpy_array()
                      |--- helper_generic.check_length_or_extend_to_array()
                      |--- _calc_K0()
                      |--- _calc_mean_confining_stress()
           |---  helper_mkz_model.fit_MKZ()
           |---  _optimization_kernel()
                      |
                      |--- hlper_mkz_model.tau_MKZ()
                      |--- helper_generic.find_closest_index()
                      |--- __calc_area()
                                 |--- helper_hh_model.tau_FKZ()
                                 |--- helper_generic.find_closest_index()
                      |--- __find_x_t_and_d()
                                 |--- helper_hh_model.tau_FKZ()
                                 |--- helper_generic.find_closest_index()

Note: functions whose names have leading underscores are not user-facing, so
they are not shown in the documentation page.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from PySeismoSoil import helper_generic as hlp
from PySeismoSoil import helper_site_response as sr
from PySeismoSoil import helper_mkz_model as mkz
from PySeismoSoil import helper_hh_model as hh


def hh_param_from_profile(
        vs_profile,
        *,
        Tmax=None,
        show_fig=False,
        save_fig=False,
        fig_output_dir=None,
        save_HH_G_file=False,
        HH_G_file_dir=None,
        profile_name=None,
        verbose=True,
):
    """
    Get HH parameters of each soil layer from the Vs values of every layer.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear-wave velocity profile, as a 2D numpy array. It should have the
        following columns:

            +---------------+----------+---------+---------+-----------------+
            | Thickness (m) | Vs (m/s) | Damping | Density | Material Number |
            +===============+==========+=========+=========+=================+
            |      ...      |   ...    |   ...   |   ...   |      ...        |
            +---------------+----------+---------+---------+-----------------+

    Tmax : numpy.ndarray or ``None``
        Shear strength of each layer of soil. If ``None``, it will be
        calculated using a combination of Ladd (1991) and Mohr-Coulomb criteria.
    show_fig : bool
        Whether to show figures G/Gmax and stress-strain curves of MKZ,
        FKZ, and HH for each layer.
    save_fig : bool
        Whether to save the figures to the hard drive. Only effective
        if ``show_fig`` is set to ``True``.
    fig_output_dir : str
        The output directory for the figures. Only effective if ``show_fig``
        and ``save_fig`` are both ``True``.
    save_HH_G_file : bool
        Whether to save the HH parameters to the hard drive (as a
        "HH_G" file).
    HH_G_file_dir : str
        The output directory for the "HH_G" file. Only effective if
        ``save_HH_G_file`` is ``True``.
    profile_name : str or ``None``
        The name of the Vs profile, such as "CE.12345". If ``None``, a string
        of current date and time will be used as the profile name.
    verbose : bool
        Whether to print progresses on the console.

    Returns
    -------
    HH_G_param : numpy.ndarray
        The HH parameters of each layer. It's a 2D array of shape
        ``(9, n_layer)``. For each layer (i.e., column), the values are in
        this order:
            gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d
    """
    phi = 30.0  # friction angle (choose 30 degrees, because there is no better info)

    hlp.check_Vs_profile_format(vs_profile)
    h = vs_profile[:-1, 0]
    Vs = vs_profile[:-1, 1]  # exclude the last layer (i.e., half space)

    n_layer = len(Vs)
    if Tmax is not None:
        hlp.assert_array_length(Tmax, n_layer)

    rho = _calc_rho(h, Vs)
    Gmax = _calc_Gmax(Vs, rho)
    sigma_v0 = _calc_vertical_stress(h, rho)
    OCR = _calc_OCR(Vs, rho, sigma_v0)
    K0 = _calc_K0(OCR, phi=phi)
    PI = _calc_PI(Vs)

    if Tmax is None:
        Tmax = _calc_shear_strength(Vs, OCR, sigma_v0, K0=K0, phi=phi)

    HH_G_param = produce_HH_G_param(
        Vs,
        Gmax,
        Tmax,
        OCR,
        sigma_v0,
        K0,
        curves=None,
        PI=PI,
        phi=phi,
        show_fig=show_fig,
        save_fig=save_fig,
        fig_output_dir=fig_output_dir,
        verbose=verbose,
    )

    if save_HH_G_file:
        if HH_G_file_dir is None:
            raise ValueError('Please specify `HH_G_file_dir`.')
        if profile_name is None:
            profile_name = hlp.get_current_time(for_filename=True)
        np.savetxt(
            os.path.join(HH_G_file_dir, 'HH_G_%s.txt' % profile_name),
            HH_G_param,
            delimiter='\t',
            fmt='%.6g',
        )

    return HH_G_param


def hh_param_from_curves(
        vs_profile,
        curves,
        *,
        Tmax=None,
        show_fig=False,
        save_fig=False,
        fig_output_dir=None,
        save_HH_G_file=False,
        HH_G_file_dir=None,
        profile_name=None,
        verbose=True,
):
    """
    Get HH parameters of each soil layer from the Vs profile and G/Gmax curves.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear-wave velocity profile, as a 2D numpy array. It should have the
        following columns:
            +---------------+----------+---------+---------+-----------------+
            | Thickness (m) | Vs (m/s) | Damping | Density | Material Number |
            +===============+==========+=========+=========+=================+
            |      ...      |   ...    |   ...   |   ...   |      ...        |
            +---------------+----------+---------+---------+-----------------+

    curves : numpy.ndarray
        A 2D numpy array that represents G/Gmax and damping curves of each
        layer, in the following format:
            +------------+--------+------------+-------------+-------------+--------+-----+
            | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
            +============+========+============+=============+=============+========+=====+
            |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
            +------------+--------+------------+-------------+-------------+--------+-----+

        The damping information is neglected in this function, so users can
        supply some dummy values.
    Tmax : numpy.ndarray or ``None``
        Shear strength of each layer of soil. If ``None``, it will be
        calculated using a combination of Ladd (1991) and Mohr-Coulomb criteria.
    show_fig : bool
        Whether to show figures G/Gmax and stress-strain curves of MKZ,
        FKZ, and HH for each layer.
    save_fig : bool
        Whether to save the figures to the hard drive. Only effective
        if ``show_fig`` is set to ``True``.
    fig_output_dir : str
        The output directory for the figures. Only effective if ``show_fig``
        and ``save_fig`` are both ``True``.
    save_HH_G_file : bool
        Whether to save the HH parameters to the hard drive (as a
        "HH_G" file).
    HH_G_file_dir : str
        The output directory for the "HH_G" file. Only effective if
        ``save_HH_G_file`` is ``True``.
    profile_name : str or ``None``
        The name of the Vs profile, such as "CE.12345". If ``None``, a string
        of current date and time will be used as the profile name.
    verbose : bool
        Whether to print progresses on the console.

    Returns
    -------
    HH_G_param : numpy.ndarray
        The HH parameters of each layer. It's a 2D array of shape
        ``(9, n_layer)``. For each layer (i.e., column), the values are in
        this order:
            gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d
    """
    phi = 30.0

    hlp.check_Vs_profile_format(vs_profile)
    h = vs_profile[:-1, 0]
    Vs = vs_profile[:-1, 1]  # exclude the last layer (i.e., half space)
    n_layer = len(Vs)

    if vs_profile.shape[1] == 5:  # there can only be 5 or 2 columns
        mat = vs_profile[:-1, -1]
        rho = vs_profile[:-1, 3]
    else:  # only 2 columns
        mat = np.arange(1, n_layer + 1)
        rho = _calc_rho(h, Vs)

    if Tmax is not None:
        hlp.assert_array_length(Tmax, n_layer)

    Gmax = _calc_Gmax(Vs, rho)
    sigma_v0 = _calc_vertical_stress(h, rho)
    OCR = _calc_OCR(Vs, rho, sigma_v0)
    K0 = _calc_K0(OCR, phi=phi)

    if Tmax is None:
        Tmax = _calc_shear_strength(Vs, OCR, sigma_v0, K0=K0, phi=phi)

    curves_old = curves.copy()
    curves_expanded = None
    for j in range(n_layer):
        tmp = curves_old[:, int(mat[j]) * 4 - 4 : int(mat[j]) * 4]
        if curves_expanded is None:
            curves_expanded = tmp
        else:
            curves_expanded = np.column_stack((curves_expanded, tmp))
    curves = curves_expanded

    HH_G_param = produce_HH_G_param(
        Vs,
        Gmax,
        Tmax,
        OCR,
        sigma_v0,
        K0,
        curves=curves,
        PI=None,
        phi=None,
        show_fig=show_fig,
        save_fig=save_fig,
        fig_output_dir=fig_output_dir,
        verbose=verbose,
    )

    if save_HH_G_file:
        if HH_G_file_dir is None:
            raise ValueError('Please specify `HH_G_file_dir`.')
        if profile_name is None:
            profile_name = hlp.get_current_time(for_filename=True)
        np.savetxt(
            os.path.join(HH_G_file_dir, 'HH_G_%s.txt' % profile_name),
            HH_G_param,
            delimiter='\t',
            fmt='%.6g',
        )

    return HH_G_param


def produce_HH_G_param(
        Vs,
        Gmax,
        Tmax,
        OCR,
        sigma_v0,
        K0,
        curves=None,
        PI=None,
        phi=None,
        show_fig=False,
        save_fig=False,
        fig_output_dir=None,
        verbose=True,
):
    """
    Produce HH_G parameters from profiles of Vs, Tmax, OCR, etc.

    Parameters
    ----------
    Vs : numpy.ndarray
        Vs values of each layer. Shape: ``(n_layer, )``, where ``n_layer`` is
        the length of ``Vs``. Unit: m/s.
    Gmax : numpy.ndarray
        Initial stiffness of each layer. Shape: ``(n_layer, )``. Unit: Pa.
    Tmax : numpy.ndarray
        The shear strength of each layer. Shape: ``(n_layer, )``. Unit: Pa.
    OCR : numpy.ndarray
        Over-consolidation ratio of each layer. Shape: ``(n_layer, )``.
    sigma_v0 : numpy.ndarray
        Vertical effective confining stress of each layer. Shape:
        ``(n_layer, )``. Unit: Pa.
    K0 : numpy.ndarray or float
        Lateral soil pressure coefficient. If an array, it must have shape
        ``(n_layer, )``. If a single value, it means that all layers share
        this same value.
    curves : numpy.ndarray or ``None``
        A 2D numpy array that represents G/Gmax and damping curves of each
        layer, in the following format:
            +------------+--------+------------+-------------+-------------+--------+-----+
            | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
            +============+========+============+=============+=============+========+=====+
            |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
            +------------+--------+------------+-------------+-------------+--------+-----+

        The damping information is neglected in this function, so users can
        supply some dummy values. If ``None``, it means that the users do not
        have G/Gmax curve information, so this function will calculate the
        MKZ G/Gmax curves from the empirical formulas in Darendeli (2001).
    PI : float or numpy.ndarray or ``None``
        Plasticity index of the soils. It is not necessary (can be ``None``) if
        ``curves`` is provided (i.e., not ``None``). If an array, it must have
        shape ``(n_layer, )``. If a single value, it means that all layers
        share this same value.
    phi : float or numpy.ndarray or ``None``
        Effective internal frictional angle (in degrees). It is not necessary
        (can be ``None``) if ``curve`` is provided (i.e., not ``None``). If
        an array, it must have shape ``(n_layer, )``. If a single value, it
        means that all layers share this same value.
    show_fig : bool
        Whether to show figures G/Gmax and stress-strain curves of MKZ,
        FKZ, and HH for each layer.
    save_fig : bool
        Whether to save the figures to the hard drive. Only effective
        if ``show_fig`` is set to ``True``.
    fig_output_dir : str
        The output directory for the figures. Only effective if ``show_fig``
        and ``save_fig`` are both ``True``.
    verbose : bool
        Whether to print progresses on the console.

    Returns
    -------
    parameters : numpy.ndarray
        The HH parameters of each layer. It's a 2D array of shape
        ``(9, n_layer)``. For each layer (i.e., column), the values are in
        this order:
            gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d

    Notes
    -----
    This function is based on ``hybridParaKernel_FKZ.m``.
    """
    hlp.assert_1D_numpy_array(Vs, '`Vs`')
    n_layer = len(Vs)

    hlp.assert_array_length(Gmax, n_layer, name='`Gmax`')
    hlp.assert_array_length(Tmax, n_layer, name='`Tmax`')
    hlp.assert_array_length(OCR, n_layer, name='`OCR`')
    hlp.assert_array_length(sigma_v0, n_layer, name='`sigma_v0`')

    K0 = hlp.check_length_or_extend_to_array(K0, n_layer, name='`K0`')
    p0 = _calc_mean_confining_stress(sigma_v0, K0)

    if verbose:
        print('========== Start optimizing for HH_G parameters ===========')
    # END

    # ============= MKZ fit ===================================================
    if curves is None:  # user does not provide curves
        if verbose:
            print(
                '------ G/Gmax not provided; will generate MKZ curves using '
                'Darendeli (2001): ------',
            )
        # END

        strain_ = np.geomspace(1e-4, 10, 400)  # unit: percent
        GGmax, _, gamma_ref = produce_Darendeli_curves(
            sigma_v0,
            PI,
            OCR=OCR,
            K0=K0,
            phi=phi,
            strain_in_pct=strain_,
        )
        strain = np.tile(strain_, (n_layer, 1)).T  # strain matrix for all layers
        beta = np.ones(n_layer)
        s = 0.9190 * np.ones(n_layer)
    else:  # user provides own curves
        if verbose:
            print('------ G/Gmax provided; fitting MKZ curves to data: ------')
        hlp.assert_2D_numpy_array(curves)
        assert curves.shape[1] == n_layer * 4

        # ----------- Extract G/Gmax information ------------------------------
        strain = curves[:, 0::4]  # unit: percent
        GGmax = curves[:, 1::4]

        # ----------- Fit MKZ parameters --------------------------------------
        param_MKZ, _ = mkz.fit_MKZ(curves, show_fig=show_fig)
        gamma_ref = param_MKZ[:, 0]
        s = param_MKZ[:, 2]
        beta = param_MKZ[:, 3]

        # ----------- Show results on console ---------------------------------
        if verbose:
            print('****** MKZ parameters: ******')
        for j in range(n_layer):
            if verbose:
                print(
                    f'Layer {j}: gamma_ref = {gamma_ref[j]:.3g}, '
                    f's = {s[j]:.3g}, beta = {beta[j]:.3g}',
                )
            # END
        # END

    # ========== Stress-strain curve implied by G/Gmax ========================
    sigma = np.zeros_like(GGmax)
    for j in range(n_layer):
        sigma[0, j] = 0
        for k in range(1, GGmax.shape[0]):
            sigma[k, j] = GGmax[k, j] * Gmax[j] * strain[k, j] / 100.0
        # END FOR
    # END FOR

    # ========== Estimate mu using empirical correlations =====================
    p0 = p0 / 1000.0  # unit: Pa --> kPa

    mu = np.zeros_like(OCR)
    for j in range(n_layer):
        if Vs[j] <= 760:  # softer soil: use Vardanega & Bolton (2011) CGJ formula
            mu[j] = 1.0 / (0.000872 * Gmax[j]/Tmax[j] * OCR[j]**0.47 * p0[j]**0.28)  # noqa: E226
            if mu[j] <= 0.02:  # mu too small --> too low tau_FKZ --> sharply decreasing tau_HH
                # 0.236 is the standard error suggested in Vardanega & Bolton (2011)
                mu[j] = mu[j] * 10.0 ** (0.236 * 3)
            elif mu[j] <= 0.03:
                mu[j] = mu[j] * 10.0 ** (0.236 * 2)
            elif mu[j] <= 0.04:
                mu[j] = mu[j] * 10.0 ** (0.236 * 1)
            # END IF
        else:  # stiffer soils: set mu to 1 for lack of better information
            mu[j] = 1.0
        # END IF
    # END FOR

    # ========== Start FKZ optimization =======================================
    if verbose:
        print('-----------  FKZ optimization  -----------------------------')
    parameters = np.zeros((9, n_layer))

    lw = 1.0
    muted_blue = np.array([107, 174, 214]) / 255.0
    muted_green = np.array([120, 198, 121]) / 255.0
    muted_red = np.array([222, 45, 38]) / 255.0

    for j in range(n_layer):
        strain_j = strain[:, j]
        a, gamma_t, d = _optimization_kernel(
            strain_j / 100.0,
            gamma_ref[j],
            beta[j],
            s[j],
            Gmax[j],
            Tmax[j],
            mu[j],
        )
        if verbose:
            print(
                f'{j + 1}/{n_layer}: mu = {mu[j]:.3f}, a = {a:.1f}, '
                f'gamma_t = {gamma_t * 100:.3f}%, d = {d:.3f}',
            )
        T_FKZ = hh.tau_FKZ(
            strain_j / 100.0,
            Gmax=Gmax[j],
            mu=mu[j],
            d=d,
            Tmax=Tmax[j],
        )
        T_HH = hh.tau_HH(
            strain_j / 100.0,
            gamma_t=gamma_t,
            a=a,
            gamma_ref=gamma_ref[j],
            beta=beta[j],
            s=s[j],
            Gmax=Gmax[j],
            mu=mu[j],
            Tmax=Tmax[j],
            d=d,
        )

        parameters[0, j] = gamma_t
        parameters[1, j] = a
        parameters[2, j] = gamma_ref[j]
        parameters[3, j] = beta[j]
        parameters[4, j] = s[j]
        parameters[5, j] = Gmax[j]
        parameters[6, j] = mu[j]
        parameters[7, j] = Tmax[j]
        parameters[8, j] = d

        GGmax_HH = T_HH / (Gmax[j] * (strain_j / 100.0))

        if show_fig:
            fig = plt.figure(figsize=(4.2, 6.0))

            plt.subplot(211)
            if curves is None:
                plt.semilogx(
                    strain_j,
                    sigma[:, j] / 1000.0,
                    c=muted_blue,
                    lw=lw * 2.5,
                    label='MKZ',  # Darendeli's curve
                )
                plt.semilogx(
                    strain_j,
                    T_FKZ / 1000.0,
                    c=muted_green,
                    lw=lw * 1.75,
                    label='FKZ',
                )
            else:
                plt.semilogx(
                    strain_j,
                    sigma[:, j] / 1000.0,
                    c=muted_blue,
                    marker='o',
                    ls='-',
                    lw=lw * 2.5,
                    label=r'Given $G/G_{\max}$',
                )
                plt.semilogx(
                    strain_j,
                    T_FKZ / 1000.0,
                    c=muted_green,
                    lw=lw * 1.75,
                    label='FKZ',
                )
            plt.grid(ls=':', lw=0.5)
            plt.plot(
                [np.min(strain_j), np.max(strain_j)],
                np.array([Tmax[j], Tmax[j]]) / 1000.0,
                lw=lw,
                c='gray',
                ls='--',
            )
            plt.plot(strain_j, T_HH / 1000.0, c=muted_red, lw=lw, label='HH')
            plt.plot([gamma_t * 100] * 2, plt.ylim(), ls='--', c='gray')
            plt.ylabel('Stress [kPa]')
            plt.xlim(np.min(strain_j), np.max(strain_j))
            plt.legend(loc='upper left')

            title_txt = f'$V_S$ = {Vs[j]:.1f} m/s, '
            title_txt += r'$G_{\max}$' + f' = {Gmax[j] / 1e6:.3f} MPa,\n'
            title_txt += r'$\tau_{\mathrm{ff}}$ = '
            title_txt += f'{Tmax[j] / 1e3:.3f} kPa, '
            title_txt += r'$\gamma_{\mathrm{ref}}$ = '
            title_txt += f'{gamma_ref[j] * 100:.3f}%'

            plt.title(title_txt)

            plt.subplot(212)
            if curves is None:
                plt.semilogx(strain_j, GGmax[:, j], c=muted_blue, lw=lw * 2.5)
            else:
                plt.semilogx(
                    strain_j,
                    GGmax[:, j],
                    c=muted_blue,
                    ls='-',
                    marker='o',
                    lw=lw * 2.5,
                )
            plt.grid(ls=':', lw=0.5)
            plt.plot(
                strain_j,
                mu[j] / (1 + Gmax[j]/Tmax[j]*mu[j]*np.abs(strain_j/100.)),  # noqa: E226
                c=muted_green,
                lw=lw * 1.75,
            )
            plt.plot(strain_j, GGmax_HH, c=muted_red, lw=lw)
            plt.plot([gamma_t * 100] * 2, plt.ylim(), ls='--', c='gray')
            plt.ylabel(r'$G/G_{\max}$')
            plt.xlabel('Strain [%]')
            plt.xlim(np.min(strain_j), np.max(strain_j))
            plt.title(
                '$\\mu$ = %.3f, a = %.1f, $\\gamma_{\\mathrm{t}}$ = %.4f%%\n'
                r"d = %.4f, $p'_{\mathrm{m0}}$ = %.2f kPa"
                % (mu[j], a, gamma_t * 100, d, p0[j]),
            )

            fig.tight_layout(pad=0.5, h_pad=1.2, w_pad=0.3)

            if save_fig:
                if fig_output_dir is None:
                    raise ValueError('Please specify `fig_output_dir`.')
                # END
                fig.savefig(
                    os.path.join(
                        fig_output_dir,
                        'Stress_GGmax_of_Layer_#%d.png' % (j + 1),
                    ),
                )
            # END

    return parameters


def _calc_shear_strength(Vs, OCR, sigma_v0, K0=None, phi=30.0):
    """
    Calculate shear strength of soils.

    Parameters
    ----------
    Vs : numpy.ndarray
        A 1D array of Vs values of each layer. Unit: m/s.
    OCR : numpy.ndarray
        A 1D array of OCR (over-consolidation ratio) of each layer. Unit: 1.
    sigma_v0 : numpy.ndarray
        A 1D array of vertical overburden pressure. Unit: Pa.
    K0 : float, int, numpy.ndarray, or ``None``
        Lateral soil pressure coefficient. If a single value is given, it is
        assumed to be the value for all layers. If ``None``, it will be
        determined from OCR via an empirical formula by Mayne & Kulhawy (1982).
    phi : float, int, or numpy.ndarray
        Effective internal friction angle of soils (in degrees).

    Returns
    -------
    Tmax : numpy.ndarray
        Shear strength of soils of each layer. Unit: Pa.
    """
    dyna_coeff = 1.2  # assume a strain rate of 0.01 sec^(-1), from Vardanega & Bolton (2013)

    phi = hlp.check_length_or_extend_to_array(phi, len(Vs), name='`phi`')
    if K0 is None:
        K0 = _calc_K0(OCR, phi=phi)
    else:
        K0 = hlp.check_length_or_extend_to_array(K0, len(Vs), name='`K0`')

    Tmax = np.zeros(len(Vs))
    for j in range(len(Vs)):
        if Vs[j] <= 760:  # for softer soils, calculate undrained shear strength
            Tmax[j] = dyna_coeff * 0.28 * OCR[j] ** 0.8 * sigma_v0[j]  # Ladd (1991)
        else:  # stiffer soils: Mohr-Coulomb criterion
            sigma_h0 = K0[j] * sigma_v0[j]  # horizontal stress
            sigma_1 = np.max([sigma_v0[j], sigma_h0])  # largest principal stress
            sigma_3 = np.min([sigma_v0[j], sigma_h0])  # smallest principal stress

            # normal effective stress on the slip plane
            sigma_n = (
                (sigma_1 + sigma_3) / 2.0
                - (sigma_1 - sigma_3) / 2.0 * np.sin(np.deg2rad(phi[j]))
            )

            Tmax[j] = dyna_coeff * sigma_n * np.tan(np.deg2rad(phi[j]))
        # END
    # END

    return Tmax


def _calc_Gmax(Vs, rho):
    """
    Calculate initial stiffness of each soil layer.

    Parameters
    ----------
    Vs : numpy.ndarray
        1D array of Vs of layers. Unit: m/s.
    rho : numpy.ndarray
        1D array of mass density of layers. Unit: kg/m^3.

    Returns
    -------
    Gmax : numpy.ndarray
        1D array of initial stiffness. Unit: Pa
    """
    Gmax = rho * Vs**2
    return Gmax


def _calc_OCR(Vs, rho, sigma_v0, OCR_upper_limit=None):
    """
    Calculate OCR (over-consolidation ratio) of each layer from the Vs profile.

    Parameters
    ----------
    Vs : numpy.ndarray
        1D array of Vs of layers. Unit: m/s.
    rho : numpy.ndarray
        1D array of mass density of layers. Unit: kg/m^3.
    sigma_v0 : numpy.ndarray
        Vertical overburden stress at the mid-point of each layer. Unit: Pa.
    OCR_upper_limit : float or ``None``
        The maximum acceptable OCR value. If ``None``, there is no limit.

    Returns
    -------
    OCR : numpy.ndarray
        1D array of OCR value, for each soil layer. (Unitless.)
    """
    sigma_p0 = 0.106 * Vs**1.47  # Mayne, Robertson, Lunne (1998) "Clay stress history evaluated fromseismic piezocone tests"  # noqa: E501,E226
    sigma_p0 = sigma_p0 * 1000  # kPa --> Pa
    OCR = sigma_p0 / sigma_v0
    OCR = np.minimum(OCR, np.inf if OCR_upper_limit is None else OCR_upper_limit)
    return OCR


def _calc_vertical_stress(h, rho):
    """
    Calculate vertical overburden stress at the mid-point of each layer.

    Parameters
    ----------
    h : numpy.ndarray
        1D array of layer thickness. Unit: m.
    rho : numpy.ndarray
        1D array of mass density of each layer. Unit: kg/m^3.

    Returns
    -------
    stress : numpy.ndarray
        Vertical overburden stress at the mid-point of each layer. Unit: Pa.
    """
    g = 9.81  # unit: m/s/s
    n = len(h)
    stress = np.zeros_like(h)

    if np.mean(rho) < 1000:
        print(
            'Warning in __calc_vertical_stress(): It looks like the unit '
            'of mass density is g/cm^3. The correct unit should be kg/m^3.',
        )

    if h[-1] == 0:  # zero thickness, i.e., half space
        h[-1] = 1

    stress[0] = rho[0] * g * h[0] / 2  # divided by 2: middle of layer
    for i in range(1, n):
        stress[i] = (
            stress[i - 1]
            + rho[i - 1] * g * h[i - 1] / 2
            + rho[i] * g * h[i] / 2
        )

    return stress


def _calc_rho(h, Vs):
    """
    Calculate mass density of soils from Vs values, using the empirical formula
    by Mayne, Schneider & Martin (1999) and Burns & Mayne (1996).

    Parameters
    ----------
    h : numpy.ndarray
        The thickness of each soil layer. Unit: m.
    Vs : numpy.ndarray
        The shear-wave velocity for each layer. It needs to be a 1D numpy array.
        Unit: m/s.

    Returns
    -------
    rho : numpy.ndarray
        Mass density of each soil layer. Unit: kg/m^3.

    References
    ----------
    1. Mayne, Schneider & Martin (1999) "Small- and large-strain soil
       properties from seismic flat dilatometer tests." Pre-failure
       deformation characteristics of geomaterials, 1999 Balkema, Rotterdam.
       (https://www.marchetti-dmt.it/wp-content/uploads/bibliografia/mayne_1999_torino_SDMT_small_large_strain.pdf)
    2. Burns & Mayne (1996) "Small- and high-strain soil properties using the
       seismic piezocone." Transportation Research Record 1548, National
       Acad. Press, Washington DC, 81-88.
    """
    z = sr.thk2dep(h, midpoint=False)
    z[z == 0] = 0.0001  # avoid error of dividing by zero
    lb = 1.65  # lower bound of density: 1.65 g/cm^3

    # Note: we are using log10(z) here instead of log(z) as written Eq (2)
    # in Mayne et al. (1999). This is because Mayne et al. actually meant
    # log10 (as evident in Figure 2 in the paper), but they incorrectly used
    # the notation of "log", which in the US means "ln" (natural logarithm)
    # rather than log10.
    rho = np.maximum(lb, 1 + 1. / (0.614 + 58.7 * (np.log10(z) + 1.095) / Vs))

    rho *= 1000  # unit: g/cm^3 --> kg/m^3
    return rho


def _calc_PI(Vs):
    """
    Calculate PI (plasticity index) from Vs values.

    Parameters
    ----------
    Vs : numpy.ndarray
        The shear-wave velocity for each layer. It needs to be a 1D numpy array.
        Unit: m/s.

    Returns
    -------
    PI : numpy.ndarray
        The plasticity index for each layer. Unit: %.
    """
    PI = np.zeros_like(Vs)
    for j in range(len(Vs)):
        if Vs[j] <= 200:
            PI[j] = 10
        elif Vs[j] <= 360:
            PI[j] = 5
        else:
            PI[j] = 0

    return PI


def _calc_K0(OCR, phi=30.0):
    """
    Calculate K0 (lateral earth pressure coefficient at rest) from OCR using
    the empirical formula by Mayne & Kulhawy (1982).

    Parameters
    ----------
    OCR : float, int, or numpy.ndarray
        Over-consolidation ratio of each layer of soils. If it is a float/int,
        it means only one layer, or all the layers have the same OCR.
    phi : float, int, or numpy.ndarray
        Internal effective friction angle of soils. If it is a float/int, it
        means only one soil layer, or all the layers have the same angle.
        Unit: deg.

    Returns
    -------
    K0 : float or numpy.ndarray
        K0 value(s). If either ``OCR`` or ``phi`` is an array, ``K0`` will be
        an array of the same length.
    """
    K0 = (1 - np.sin(np.deg2rad(phi))) * OCR ** (np.sin(np.deg2rad(phi)))
    return K0


def produce_Darendeli_curves(
        sigma_v0,
        PI=20.0,
        OCR=1.0,
        K0=0.5,
        phi=30.0,
        strain_in_pct=None,
):
    """
    Produce G/Gmax and damping curves using empirical correlations by
    Darendeli (2001).

    Parameters
    ----------
    sigma_v0 : numpy.ndarray
        Effective vertical confining stress of each layer. Unit: Pa.
    PI : int, float, or numpy.ndarray
        Plasticity index of each layer. Unit: %. If a single value is given,
        it is assumed to be the PI for all layers.
    OCR : int, float, or numpy.ndarray
        Over-consolidation ratio of each layer. If a single value is given,
        it is assumed to be the value for all layers.
    K0 : int, float, numpy.ndarray, or ``None``
        Lateral soil pressure coefficient. If a single value is given, it is
        assumed to be the value for all layers. If ``None``, it will be
        determined from OCR via an empirical formula by Mayne & Kulhawy (1982).
    phi : int, float, or numpy.ndarray
        Internal effective friction angle of soils. If it is a float/int, it
        means all the layers have the same angle. Unit: deg.
    strain_in_pct : numpy.ndarray or ``None``
        The strain values at which to calculate G/Gmax and damping values. If
        ``None``, numpy.geomspace(1e-4, 10, 400) will be used. Unit: percent.

    Returns
    -------
    GGmax : numpy.ndarray
        G/Gmax curves for each layer. It is a 2D numpy array. Each column of it
        represents the G/Gmax curve of a particular layer. Unit: 1
    xi : numpy.ndarray
        Damping curves for each layer. Same shape as ``GGmax``. Unit: 1.
    gamma_r : numpy.ndarray
        Reference strain for each layer. It is a 1D numpy array, corresponding
        to each soil layer. Unit: 1.
    """
    hlp.assert_1D_numpy_array(sigma_v0)
    n_layer = len(sigma_v0)

    phi = hlp.check_length_or_extend_to_array(phi, n_layer, name='`phi`')
    PI = hlp.check_length_or_extend_to_array(PI, n_layer, name='`PI`')
    OCR = hlp.check_length_or_extend_to_array(OCR, n_layer, name='`OCR`')

    if K0 is None:
        K0 = _calc_K0(OCR, phi=phi)
    else:
        K0 = hlp.check_length_or_extend_to_array(K0, n_layer, name='`K0`')

    if strain_in_pct is None:
        gamma = np.geomspace(1e-4, 10, 400)
    else:
        gamma = strain_in_pct.copy()

    # Define all constants
    nr_cycle = 10
    frq = 1
    N = nr_cycle

    phi1 = 0.0352
    phi2 = 0.0010
    phi3 = 0.3246
    phi4 = 0.3483
    phi5 = 0.9190
    phi6 = 0.8005
    phi7 = 0.0129
    phi8 = -0.1069
    phi9 = -0.2889
    phi10 = 0.2919
    phi11 = 0.6329
    phi12 = -0.0057
    a = phi5

    c1 = -1.1143 * a**2 + 1.8618 * a + 0.2523  # from Darendeli (2001), page 226
    c2 = 0.0805 * a**2 - 0.0710 * a - 0.0095
    c3 = -0.0005 * a**2 + 0.0002 * a + 0.0003
    b = phi11 + phi12 * np.log(N)  # Darendeli (2001) Eq 9.1d

    # Confinine stress
    sigma_0 = _calc_mean_confining_stress(sigma_v0, K0)  # octahedral stress
    sigma_0 = sigma_0 / 101325.0  # unit: Pa --> atm
    n_strain_pts = len(strain_in_pct)

    # Reference strain for each layer (Eq 9.1a). Unit: percent
    gamma_r = (phi1 + phi2 * PI * OCR**phi3) * sigma_0**phi4

    GGmax = np.zeros((n_strain_pts, n_layer))
    xi = np.zeros_like(GGmax)
    for i in range(n_layer):
        GGmax[:, i] = 1. / (1 + (gamma / gamma_r[i])**a)  # G of i-th layer (Eq 9.2a)
        D_masing_1 = (  # unit: % (page 226)
            (100. / np.pi)
            * (
                4
                * (gamma - gamma_r[i] * np.log((gamma + gamma_r[i]) / gamma_r[i]))
                / (gamma**2 / (gamma + gamma_r[i])) - 2
            )
        )
        D_masing = c1 * D_masing_1 + c2 * D_masing_1**2 + c3 * D_masing_1**3  # unit: % (page 226)
        D_min = (phi6 + phi7 * PI[i] * OCR[i]**phi8) * sigma_0[i]**phi9 * (1 + phi10 * np.log(frq))  # Eq 9.1c (page 221)  # noqa: E501, LN001
        xi[:, i] = b * GGmax[:, i]**0.1 * D_masing + D_min  # Eq 9.2b (page 224). Unit: percent

    xi /= 100.0
    gamma_r /= 100.0
    return GGmax, xi, gamma_r


def _calc_mean_confining_stress(sigma_v0, K0):
    """
    Calculate mean (of three directions) confining stress.

    Parameters
    ----------
    sigma_v0 : numpy.ndarray
        (Effective) vertical stress of each layer. Unit: Pa.
    K0 : numpy.ndarray
        Lateral stress coefficient of each layer. Unit: 1.

    Returns
    -------
    sigma_m0 : numpy.ndarray
        Mean effective confining stress (of three directions). Unit: Pa.
    """
    sigma_m0 = (2 * K0 + 1) / 3.0 * sigma_v0
    return sigma_m0


def _optimization_kernel(x, x_ref, beta, s, Gmax, tau_f, mu):
    """
    Optimization process to find FKZ parameters.

    Parameters
    ----------
    x : numpy.ndarray
        An 1D array of shear strain. Unit: 1.
    x_ref : float
        The "reference strain" parameter (in MKZ) of the soil. Unit: 1.
    beta : float
        A shape parameter of the FKZ model.
    s : float
        A shape parameter of the FKZ model.
    Gmax : float
        Initial shear modulus. Unit: Pa.
    tau_f : float
        The shear strength of the current soil layer. Unit: Pa.
    mu : float
        The "shape parameter" of the FKZ model.

    Returns
    -------
    a : float
        A parameter of the HH model that defines the "speed" of transition from
        MKZ to FKZ
    gamma_t : float
        The shear strain at which the transition from MKZ to FKZ happens.
        Unit: 1
    d : float
        The "shape power" parameter in the FKZ model.

    Notes
    -----
    Based on optHybFKZ.m
    """
    T_MKZ = mkz.tau_MKZ(x, gamma_ref=x_ref, beta=beta, s=s, Gmax=Gmax)
    if mu <= 0.03:  # when mu is too small, there may be some numerical issues
        gamma_t_LB = 0.001  # therefore gamma_t lower bound is relaxed
    else:
        gamma_t_LB = 0.01

    gamma_t_UB = 3.0  # unit: percent

    index_gamma_t_LB, _ = hlp.find_closest_index(x, gamma_t_LB / 100.0)
    if T_MKZ[index_gamma_t_LB] >= 0.85 * tau_f:
        gamma_t_LB = 0.005  # for very deep layers, tau_MKZ may be larger than tau_f at gamma_t_LB

    index_gamma_t_LB, _ = hlp.find_closest_index(x, gamma_t_LB / 100.0)  # do it again
    if T_MKZ[index_gamma_t_LB] >= 0.85 * tau_f:
        gamma_t_LB = 0.001

    range_d = np.linspace(0.67, 1.39, 200)
    area = __calc_area(range_d, x, Gmax, mu, tau_f, gamma_t_LB, gamma_t_UB, T_MKZ)
    if np.min(area) < np.inf:  # it means that a proper d value is found
        gamma_t, d = __find_x_t_and_d(area, range_d, x, Gmax, mu, tau_f, T_MKZ)
    else:  # cannot find a proper d value
        range_d = np.linspace(0.67, 1.39, 400)  # increase grid density to 400
        area = __calc_area(
            range_d,
            x,
            Gmax,
            mu,
            tau_f,
            gamma_t_LB,
            gamma_t_UB,
            T_MKZ,
        )
        if np.min(area) < np.inf:
            gamma_t, d = __find_x_t_and_d(area, range_d, x, Gmax, mu, tau_f, T_MKZ)
        else:
            range_d = np.linspace(0.67, 1.39, 1000)  # increase grid density
            new_gamma_t_LB = 0.005  # further relax
            area = __calc_area(
                range_d,
                x,
                Gmax,
                mu,
                tau_f,
                new_gamma_t_LB,
                gamma_t_UB,
                T_MKZ,
            )
            if np.min(area) < np.inf:
                gamma_t, d = __find_x_t_and_d(area, range_d, x, Gmax, mu, tau_f, T_MKZ)
            else:
                d = 1.03
                gamma_t = 1e-3 / 100.0  # further ralax to 0.001%
            # END IF
        # END IF
    # END IF

    a = 100.0  # always use a fast transition
    return a, gamma_t, d


def __find_x_t_and_d(area, range_d, x, Gmax, mu, tau_f, T_MKZ):
    """
    Find the ``x_t`` (transition strain) that minimizes the "area" between
    the MKZ stress curve and the FKZ stress curve.

    Parameters
    ----------
    area : numpy.ndarray
        The "area" between the MKZ stress curve and the FKZ stress curve. It
        has the same shape as ``range_d``, because each element of ``area`` is
        the area corresponding to a ``d`` value within ``range_d``.
    range_d : numpy.ndarray
        The range of ``d`` to search from. Must be a 1D numpy array.
    x : numpy.ndarray
        An 1D array of shear strain. Unit: 1.
    Gmax : float
        Initial shear modulus. Unit: Pa.
    mu : float
        The "shape parameter" of the FKZ model.
    tau_f : float
        The shear strength of the current soil layer. Unit: Pa.
    T_MKZ : numpy.ndarray
        The MKZ stress curve, which has the same shape as ``x``. Unit: Pa.

    Returns
    -------
    x_t : float
        The ``x_t`` value that minimizes the "area" between the MKZ stress
        curve and the FKZ stress curve. Unit: 1.
    d : float
        The ``d`` value that minimizes the "area" between the MKZ stress curve
        and the FKZ stress curve. (No unit.)
    """
    j_ = np.argmin(area)
    d = range_d[j_]
    T_FKZ = hh.tau_FKZ(x, Gmax=Gmax, mu=mu, d=d, Tmax=tau_f)
    copt, _ = hlp.find_closest_index(np.abs(T_MKZ - T_FKZ), 0)
    x_t = x[copt]

    return x_t, d


def __calc_area(range_d, x, Gmax, mu, tau_f, gamma_t_LB, gamma_t_UB, T_MKZ):
    r"""
    Calculate the "area" between the MKZ stress curve and the FKZ stress curve.
    The MKZ stress curve is supplied as a parameter, and the FKZ stress curve
    is determined by ``x``, ``Gmax``, ``mu``, ``d``, ``tau_f``, and ``gamma_t``.

    Parameters
    ----------
    range_d : numpy.ndarray
        The range of ``d`` to search from. Must be a 1D numpy array.
    x : numpy.ndarray
        An 1D array of shear strain. Unit: 1.
    Gmax : float
        Initial shear modulus. Unit: Pa.
    mu : float
        The "shape parameter" of the FKZ model.
    tau_f : float
        The shear strength of the current soil layer. Unit: Pa.
    gamma_t_LB : float
        The lower bound of ``gamma_t`` (:math:`\gamma_t`), i.e., the transition
        strain. Unit: %.
    gamma_t_UB : float
        The upper bound of ``gamma_t``. Unit: %.
    T_MKZ : numpy.ndarray
        The MKZ stress curve, which has the same shape as ``x``. Unit: Pa.

    Returns
    -------
    area : numpy.ndarray
        The "area" between the MKZ stress curve and the FKZ stress curve. It
        has the same shape as ``range_d``, because each element of ``area`` is
        the area corresponding to a ``d`` value within ``range_d``.
    """
    area = np.zeros_like(range_d)
    for j in range(len(range_d)):
        d = range_d[j]
        T_FKZ = hh.tau_FKZ(x, Gmax=Gmax, mu=mu, d=d, Tmax=tau_f)
        range_gamma_t = np.geomspace(gamma_t_LB, gamma_t_UB, 200) / 100.0  # unit: 1

        copt, _ = hlp.find_closest_index(np.abs(T_MKZ - T_FKZ), 0)  # "copt" = cross-over point
        gamma_t = x[copt]
        if (gamma_t >= range_gamma_t[0]) and (gamma_t <= range_gamma_t[-1]):
            diff_T = np.abs(T_MKZ[: copt + 1] - T_FKZ[: copt + 1])
            area[j] = np.linalg.norm(diff_T) / (copt + 1.0)
        else:
            area[j] = np.inf
        # END IF
    # END FOR
    return area
