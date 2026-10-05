"""Sediment Velocity Model (SVM) class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import fsolve

from PySeismoSoil import helper_site_response as sr
from PySeismoSoil.class_Vs_profile import Vs_Profile

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


# Range of Vs30 where the SVM is applicable
MIN_APPLICABLE_VS30_M_S = 173.1
MAX_APPLICABLE_VS30_M_S = 1000

# Range of the "trial Vs30" values allowed in the Vs30 iteration
MIN_TRIAL_VS30_M_S = 130
MAX_TRIAL_VS30_M_S = 1000

# The top part of the Vs profile is a homogeneous layer of this thickness
TOP_HOMOGENEOUS_LAYER_THICKNESS_M = 2.5

# Bedrock Vs that is added at the bottom of a randomized Vs profile if the
# profile does not reach this value
BEDROCK_VS_M_S = 1000

# Tolerances of a randomized profile to be "compliant" with the base
# profile (in Vs30, in the last-layer Vs, and in z1, respectively)
VS30_COMPLIANCE_TOL_M_S = 25.0
LAST_VS_COMPLIANCE_REL_TOL = 0.05
Z1_COMPLIANCE_REL_TOL = 0.20

# Upper bounds of Vs30 (exclusive) of NEHRP site classes E, D, and C
SITE_CLASS_E_UPPER_VS30_M_S = 180
SITE_CLASS_D_UPPER_VS30_M_S = 360
SITE_CLASS_C_UPPER_VS30_M_S = 760

# Depth beyond which the inter-layer correlation coefficient in Toro (1995)
# is a constant (rho_200)
TORO_CORRELATION_REF_DEPTH_M = 200.0


class SVM:
    """
    Class implementation for the Sediment Velocity Model (SVM).

    Original paper: Shi & Asimaki (2018) "A generic velocity profile for basin
    sediments in California conditioned on Vs30". Seismological Research
    Letters, 89(4)

    Parameters
    ----------
    target_Vs30 : float
        The Vs30 value to be queried. Unit: m/s.
    z1 : float | None, default=None
        The depth to bedrock (1,000 m/s rock). Unit: m. If ``None``, it will be
        estimated from Vs30 using an empirical correlation (see documentation
        of ``helper_site_response.calc_z1_from_Vs30()``).
    Vs_cap : bool | float, default=True
        Whether to "cap" the Vs profile or not. If True, then the Vs profile is
        capped at 1000.0 m/s; if specified as another real number, Vs profile
        is capped at that value. If the resultant Vs profile does not reach
        ``Vs_cap`` at ``z1``, it will be "glued" to ``Vs_cap``, resulting in a
        velocity impedance at ``z1``. If the Vs profile exceeds ``Vs_cap`` at a
        depth shallower than ``z1``, then the smooth Vs profile is truncated at
        a depth where ``Vs = eta * Vs_cap``, then filled down to ``z1`` with
        linearly increasing Vs values.
    eta : float, default=0.90
        If Vs will reach ``Vs_cap`` (usually 1000 m/s) before the depth of
        ``z1``, the SVM Vs profile will stop at ``Vs = eta * Vs_cap``, and then
        a linear Vs gradation will be filled from ``eta * Vs_cap`` to
        ``Vs_cap``. Do not change this parameter, unless you know what you are
        doing.
    show_fig : bool, default=False
        Whether to plot the generated Vs profile.
    iterate : bool, default=False
        Whether to iteratively adjust the input Vs30 so that the actual Vs30
        (calculated from the resultant Vs profile) falls within 10 m/s of the
        ``target_Vs30``. (There is usually no need to do this.)
    verbose : bool, default=False
        Whether to print iteration progress (trial Vs30 value and calculated
        Vs30 value) on the terminal. It has no effects if ``iterate`` is
        ``False``.

    Attributes
    ----------
    Vs30 : float
        The target Vs30 value, in m/s.
    z1 : float
        The basin depth, in meters.
    base_profile : Vs_Profile
        The base Vs profile associated with the given ``Vs30`` and ``z1``.
    bedrock_Vs : float
        Bedrock Vs, either user-specified (via ``Vs_cap``), or automatically
        chosen as 1,000 m/s, or ``None`` (if ``Vs_cap`` is False).
    has_bedrock_Vs : bool
        Whether the Vs profile has a bedrock Vs value.

    Raises
    ------
    ValueError
        When values of some input arguments are not correct/valid
    """

    Vs30: float
    z1: float
    base_profile: Vs_Profile
    bedrock_Vs: float
    has_bedrock_Vs: bool

    def __init__(  # noqa: C901, PLR0915
            self,
            target_Vs30: float,
            *,
            z1: float | None = None,
            Vs_cap: bool | float = True,
            eta: float = 0.90,
            show_fig: bool = False,
            iterate: bool = False,
            verbose: bool = False,
    ) -> None:
        thk = 0.1  # hard-coded to be 10 cm, because this is small enough

        if (target_Vs30 < MIN_APPLICABLE_VS30_M_S) or (
            target_Vs30 > MAX_APPLICABLE_VS30_M_S
        ):
            print(
                '***** Warning in initializing an SVM object: your Vs30'
                ' (%.2f m/s) is out of the range of applicability of the SVM'
                f' ({MIN_APPLICABLE_VS30_M_S} m/s to'
                f' {MAX_APPLICABLE_VS30_M_S} m/s); the result may not be'
                ' as credible. *****',
            )

        if eta <= 0 or eta > 1:
            raise ValueError('`eta` must be between (0, 1].')

        # thickness of "additional" layer to be added on top
        thk_addl_layer = TOP_HOMOGENEOUS_LAYER_THICKNESS_M - thk

        # Note 1: The first layer of Vs_analyt (before adding any new layers on
        #         top) is Vs0. The final Vs profile should have a homogeneous
        #         Vs layer for the top 2.5 m, thus we should add a new layer
        #         with Vs = Vs0 whose thickness is "2.5 minus thk".
        #
        # Note 2: For shallow profiles (i.e., z1 < 50 m), we still want at
        #         least 50 layers, so we solve these following two equations:
        #
        #            thk$ = 2.5 - thk  (note: thk$ is `thk_addl_layer`)
        #            thk = (z1 - thk$)/50   (divide remaining soils into
        #                                    50 layers)
        #
        #         Then thk and thk$ can both be solved, hence we have:
        #         >>>    thk = (z1 - 2.5)/49.0

        p1 = -2.1688e-04  # these values come from curve fitting
        p2 = 0.5182
        p3 = 69.452

        # (Unused alternative fitting parameters: q1 = 8.4562e-09,
        # q2 = 2.9981, and q3 = 0.03073.)

        # updated on 2018/1/2: improved curve fitting accuracy for k_
        r1 = -59.67
        r2 = -0.2722
        r3 = 11.132

        s1 = 4.110
        s2 = -1.0521e-04
        s3 = -10.827
        s4 = -7.6187e-03

        if z1 is None:
            z1 = sr.calc_z1_from_Vs30(target_Vs30)

        # a rare case, but it does happen sometimes...
        if z1 <= TOP_HOMOGENEOUS_LAYER_THICKNESS_M:
            Vs0_ = p1 * target_Vs30**2.0 + p2 * target_Vs30 + p3

            # just one layer
            vs_profile = np.array([[z1, Vs0_], [0.0, 1000.0]])
        else:  # this is most of the cases...
            Vs30 = target_Vs30
            iteration_flag = True

            while iteration_flag is True:
                # --------  Calculate analytical Vs profile from Vs30  -------
                Vs0_ = p1 * Vs30**2.0 + p2 * Vs30 + p3

                k_ = np.exp(r1 * Vs30**r2 + r3)  # updated on 2018/1/2
                n_ = np.max([
                    1.0,
                    s1 * np.exp(s2 * Vs30) + s3 * np.exp(s4 * Vs30),
                ])

                # depth array
                z_array_analyt = np.arange(0.0, z1 - thk_addl_layer, thk)

                # thickness array (for analytical Vs)
                th_array_analyt = sr.dep2thk(z_array_analyt)

                # analytical Vs ( = Vs0*(1+k*z)^(1/n) )
                Vs_analyt = Vs0_ * (1.0 + k_ * z_array_analyt) ** (1.0 / n_)

                # the homogeneous layer with Vs = Vs0:
                array1 = np.array([thk_addl_layer, Vs_analyt[0]])

                # the other layers (i.e., Vs = Vs0*(1+k*z)^(1/n) )
                array2 = np.column_stack((th_array_analyt, Vs_analyt))

                # stack the homogeneous layer on top
                temp_Vs_profile = np.vstack((array1, array2))

                if iterate is False:
                    # abort while loop after only one run
                    iteration_flag = False
                else:
                    # -------  Check if actual Vs30 matches target Vs30 -------
                    actual_Vs30 = sr.calc_Vs30(temp_Vs_profile)
                    if verbose is True:  # print iteration progress
                        print(
                            f'  {actual_Vs30:.1f} --> {target_Vs30:.1f} |',
                            end='',
                        )

                    if target_Vs30 - 10 <= actual_Vs30 <= target_Vs30 + 10:
                        iteration_flag = False  # end iteration
                        if verbose is True:
                            print('|')
                    else:
                        # update the "trial Vs30" to offset the difference
                        Vs30_temp = Vs30 - (actual_Vs30 - target_Vs30) / 5.0

                        # if the "trial Vs30" is out of range
                        if (Vs30_temp < MIN_TRIAL_VS30_M_S) or (
                            Vs30_temp > MAX_TRIAL_VS30_M_S
                        ):
                            iteration_flag = False  # end iteration
                            if verbose is True:
                                print()
                        else:
                            # use the "trial Vs30" as the new Vs30
                            Vs30 = Vs30_temp

            # the homogeneous layer with Vs = Vs0
            array1 = np.array([thk_addl_layer, Vs_analyt[0]])

            # the other layers (i.e., Vs = Vs0*(1+k*z)^(1/n) )
            array2 = np.column_stack((th_array_analyt, Vs_analyt))

            # stack the homogeneous layer on top
            temp_Vs_profile = np.vstack((array1, array2))

            # ---------   Prepare output variables  ---------------
            # if we need to "cap" the Vs profile somehow
            if Vs_cap is not False:
                # if Vs_cap value not specified (i.e., user inputs "True")
                if Vs_cap is True:
                    Vs_cap = 1000.0  # use 1000.0 m/s as Vs_cap

                # if Vs_analyt eventually exceeds Vs_cap
                if np.where(Vs_analyt > Vs_cap)[0].size > 0:
                    # find the index from which Vs_analyt exceeds Vs_cap
                    index_Vs_cap = np.where(Vs_analyt > Vs_cap)[0][0]
                else:
                    # use NaN to denote the alternative situation
                    index_Vs_cap = np.nan

                # total number of layers in the smooth profile (Vs_analyt)
                end_index = len(Vs_analyt)

                if not np.isnan(index_Vs_cap):  # if index_Vs_cap is not NaN
                    # where Vs_analyt exceeds eta*Vs_cap
                    idx_eta_Vs_cap = np.where(Vs_analyt > Vs_cap * eta)[0][0]

                    # change Vs value where Vs > eta * Vs_cap
                    for i in range(idx_eta_Vs_cap, end_index):
                        # linearly distribute Vs increment from eta*Vs_cap
                        # to Vs_cap
                        Vs_analyt[i] = Vs_cap * eta + Vs_cap * (1 - eta) / (
                            end_index - idx_eta_Vs_cap
                        ) * (i - idx_eta_Vs_cap)

                # thickness (including a 0-m "phantom" layer)
                array3 = np.append(th_array_analyt[:-1], 0.0)

                # Vs ("phantom" layer has Vs = Vs_cap)
                array4 = np.append(Vs_analyt[:-1], Vs_cap)

                # place thickness and Vs side by side
                array5 = np.column_stack((array3, array4))

                # stack additional layer on top
                vs_profile = np.vstack((array1, array5))
            else:  # if Vs profile is not to be capped
                vs_profile = np.copy(temp_Vs_profile)

        # ----------  Show figure  -----------------
        if show_fig is True:
            title_text = (
                f'$V_{{S30}}$={target_Vs30:.1f}m/s, $z_{{1}}$={z1:.1f}m'
            )
            sr.plot_Vs_profile(vs_profile, title=title_text)

        # --------  Attributes  --------------------
        self.Vs30 = target_Vs30
        self.z1 = z1
        self._base_profile = vs_profile  # for use within class methods
        self.base_profile = Vs_Profile(vs_profile)  # for external users
        if Vs_cap is not False:
            self.bedrock_Vs = Vs_cap  # Vs_cap is already a number, not `True`
            self.has_bedrock_Vs = True
        else:
            self.has_bedrock_Vs = False
            self.bedrock_Vs = None

    def __repr__(self) -> str:
        """Return basic information of the SVM."""
        return f'Vs30 = {self.Vs30:.2g} m/s, z1 = {self.z1:.2g} m'

    def plot(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            figsize: tuple[float, float] = (2.6, 3.2),
            dpi: float = 100,
            **kwargs: dict[Any, Any],
    ) -> tuple[Figure, Axes, Line2D]:
        """
        Plot the base profile.

        Parameters
        ----------
        fig : Figure | None, default=None
            Figure object. If None, a new figure will be created.
        ax : Axes | None, default=None
            Axes object. If None, a new axes will be created.
        figsize : tuple[float, float], default=(2.6, 3.2)
            Figure size in inches, as a tuple of two numbers. The figure size
            of ``fig`` (if not ``None``) will override this parameter.
        dpi : float, default=100
            Figure resolution. The dpi of ``fig`` (if not ``None``) will
            override this parameter.
        **kwargs : dict[Any, Any]
            Other keyword arguments to be passed to
            ``helper_site_response.plot_Vs_profile()``.

        Returns
        -------
        fig : Figure
            The figure object being created or being passed into this function.
        ax : Axes
            The axes object being created or being passed into this function.
        h_line : Line2D
            The line object.
        """
        title = f'$V_{{S30}}$={self.Vs30:.1f}m/s, $z_{{1}}$={self.z1:.1f}m'
        fig, ax, h_line = sr.plot_Vs_profile(
            self._base_profile,
            title=title,
            fig=fig,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            **kwargs,
        )
        return fig, ax, h_line

    def get_discretized_profile(
            self,
            *,
            fixed_thk: float | None = None,
            Vs_increment: float | None = None,
            at_midpoint: bool = True,
            show_fig: bool = False,
    ) -> Vs_Profile:
        """
        Return the discretized Vs profile.

        The layering is determined by the user-specified layer thickness, or Vs
        increment.

        Parameters
        ----------
        fixed_thk : float | None, default=None
            The layer thickness for each layer.
        Vs_increment : float | None, default=None
            The Vs increment between adjacent layers.
        at_midpoint : bool, default=True
            Whether to return Vs values queried at the top of each layer depth.
            It is strongly recommended that you use ``True``. Using ``False``
            will produce biased Vs profiles.
        show_fig : bool, default=False
            Whether to show the figure of smooth and discretized profiles.

        Returns
        -------
        discr_prof : Vs_Profile
            Discretized Vs profile.

        Raises
        ------
        ValueError
            When the values of some input arguments are incorrect/invalid
        """
        if fixed_thk is None and Vs_increment is None:
            msg = 'You need to provide either `fixed_thk` or `Vs_increment`.'
            raise ValueError(msg)

        if fixed_thk is not None and Vs_increment is not None:
            msg = (
                'Please only provide `fixed_thk` or `Vs_increment`;'
                ' do not provide both.'
            )
            raise ValueError(msg)

        if fixed_thk is not None:
            discr_prof = self.base_profile.query_Vs_given_thk(
                fixed_thk,
                as_profile=True,
                at_midpoint=at_midpoint,
            )
        else:  # Vs_increment is not None
            max_Vs = np.max(self._base_profile[:, 1])
            if Vs_increment >= max_Vs:
                raise ValueError(
                    f'`Vs_increment` needs to < {max_Vs:.2g} m/s (the'
                    ' max Vs of the smooth profile)',
                )

            n_layers = self._base_profile.shape[0]
            discr_Vs_previous_layer = self._base_profile[0, 1]
            layer_bottom_depth_array = [0]
            thk_tmp = 0
            current_depth = 0
            for j in range(n_layers):
                thk = self._base_profile[j, 0]
                current_depth += thk
                base_Vs_j_th_layer = self._base_profile[j, 1]
                if base_Vs_j_th_layer < discr_Vs_previous_layer + Vs_increment:
                    thk_tmp += thk
                else:
                    # We need different treatments for two different cases:
                    # (1) `Vs_increment` exceeds the "natural" increment of the
                    #     base profile --- accumulate "temporary layer" whose
                    #     thickness is `thk_tmp`
                    # (2) `Vs_increment` is smaller than the "natural"
                    #     increment of the base profile --- we need to use
                    #     the natural increment as the Vs increment
                    if thk_tmp != 0:  # the first case
                        discr_Vs_previous_layer += Vs_increment
                    else:  # the second case
                        discr_Vs_previous_layer = base_Vs_j_th_layer

                    thk_tmp = 0
                    layer_bottom_depth_array.append(current_depth)

            thk_array = sr.dep2thk(
                np.array(layer_bottom_depth_array),
                include_halfspace=False,
            )
            discr_prof = self.base_profile.query_Vs_given_thk(
                thk_array,
                as_profile=True,
                at_midpoint=at_midpoint,
            )

        discr_prof = discr_prof.truncate(depth=self.z1, Vs=self.bedrock_Vs)
        prof_ = discr_prof.vs_profile

        if show_fig:
            self._plot_additional_profile(prof_, 'Discretized')

        return discr_prof

    def _plot_additional_profile(
            self, addtl_profile: np.ndarray, label: str
    ) -> None:
        """
        Plot an additional Vs profile on top of the base Vs profile.

        Parameters
        ----------
        addtl_profile : np.ndarray
            Additional Vs profile.
        label : str
            Label of the additional profile, to be shown in the legend.
        """
        title = f'$V_{{S30}}$={self.Vs30:.1f}m/s, $z_{{1}}$={self.z1:.1f}m'
        fig, ax, _ = sr.plot_Vs_profile(self._base_profile, label='Smooth')
        sr.plot_Vs_profile(
            addtl_profile,
            fig=fig,
            ax=ax,
            c='orange',
            alpha=0.85,
            label=label,
        )
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_xlim(0, np.max(np.append(addtl_profile[:, 1], 1000)) * 1.1)

    def get_randomized_profile(
            self,
            seed: float | None = None,
            *,
            show_fig: bool = False,
            use_Toros_layering: bool = False,
            use_Toros_std: bool = False,
            vs30_z1_compliance: bool = False,
            verbose: bool = True,
    ) -> Vs_Profile:
        """
        Return a randomized a 1D profile.

        Parameters
        ----------
        seed : float | None, default=None
            The seed value for setting the random state. If ``None``, a
            different random seed is used every time.
        show_fig : bool, default=False
            Whether to show the figure of smooth and randomized profiles.
        use_Toros_layering : bool, default=False
            Whether to use the layering relation in Toro (1995) instead of Eq
            (7) of Shi & Asimaki (2018).
        use_Toros_std : bool, default=False
            Whether to use the standard deviation (i.e., sigma(ln(Vs))) in Toro
            (1995) instead of Eq (9) of Shi & Asimaki (2018).
        vs30_z1_compliance : bool, default=False
            Whether to ensure that the resultant Vs30 and z1 of the randomized
            profile are compliant with the user-specified Vs30 and z1 values.
            The criteria for "compliance" are:
                1. The absolute difference between the randomized and target
                   Vs30 is < 25 m/s;
                2. The relative difference (between the randomized profile and
                   the base profile) of the last soil layer's Vs is < 5%;
                3. The relative difference of the randomized and target z1 is
                   < 20%.
        verbose : bool, default=True
            Whether to show the progress of iteratively searching for compliant
            randomized Vs profile. Only effective if ``vs30_z1_compliance`` is
            ``True``.

        Returns
        -------
        Vs_profile : Vs_Profile
            The randomized Vs profile.

        Raises
        ------
        TypeError
            If ``seed`` is not a number or not ``None``
        """
        if not isinstance(seed, (type(None), int, float, np.number)):
            raise TypeError('`seed` needs to be a number, or `None`.')

        options = {
            'seed': seed,
            'show_fig': show_fig,
            'use_Toros_std': use_Toros_std,
            'use_Toros_layering': use_Toros_layering,
        }

        if not vs30_z1_compliance:
            Vs_profile = self._helper_get_rand_profile(**options)
        else:
            iterate = True
            counter = 0
            if verbose:
                print('Iterating for compliant Vs profile:')

            while iterate:
                seed_ = None if seed is None else seed + counter
                options.update({'seed': seed_, 'show_fig': False})
                Vs_profile = self._helper_get_rand_profile(**options)
                rand_Vs30 = sr.calc_Vs30(
                    Vs_profile,
                    option_for_profile_shallower_than_30m=1,
                )
                rand_Vs_last = Vs_profile[-1, 1]
                rand_z1 = sr.calc_z1(Vs_profile)
                base_Vs30 = self.Vs30
                base_Vs_last = self._base_profile[-1, 1]
                base_z1 = sr.calc_z1(self._base_profile)

                condition_1 = (
                    np.abs(rand_Vs30 - base_Vs30) < VS30_COMPLIANCE_TOL_M_S
                )
                condition_2 = (
                    np.abs(rand_Vs_last - base_Vs_last) / base_Vs_last
                    < LAST_VS_COMPLIANCE_REL_TOL
                )
                condition_3 = (
                    np.abs(rand_z1 - base_z1) / base_z1 < Z1_COMPLIANCE_REL_TOL
                )

                if condition_1 and condition_2 and condition_3:
                    iterate = False
                    if verbose:
                        print()
                else:
                    iterate = True
                    counter += 1
                    if verbose:
                        print('.', end='\n' if counter % 80 == 0 else '')

            if show_fig:
                self._plot_additional_profile(Vs_profile, 'Stochastic')

        return Vs_Profile(Vs_profile)

    def _helper_get_rand_profile(  # noqa: C901, PLR0915
            self,
            seed: int | None = None,
            *,
            show_fig: bool = False,
            use_Toros_layering: bool = False,
            use_Toros_std: bool = False,
    ) -> np.ndarray:
        """
        Get randomized 1D profile.

        Parameters
        ----------
        seed : int | None, default=None
            The seed value for setting the random state. If ``None``, a
            different random seed is used every time.
        show_fig : bool, default=False
            Whether to show the figure of smooth and randomized profiles.
        use_Toros_layering : bool, default=False
            Whether to use the layering relation in Toro (1995) instead of Eq
            (7) of Shi & Asimaki (2018).
        use_Toros_std : bool, default=False
            Whether to use the standard deviation (i.e., sigma(ln(Vs))) in Toro
            (1995) instead of Eq (9) of Shi & Asimaki (2018).

        Returns
        -------
        Vs_profile : np.ndarray
            The randomized Vs profile.
        """
        if seed is None:
            # A new seed from the OS's entropy source. (It is < 2**31 so that
            # `2 * seed` below is a valid seed too.)
            seed = np.random.default_rng().integers(2**31)

        seed = int(seed)  # convert seed_value into int (for robustness)

        # A local random state (rather than `np.random.seed()`) leaves numpy's
        # global random state untouched, and it draws the same numbers as
        # `np.random.seed(seed)` did, so a given seed gives the same profile
        rng = np.random.RandomState(seed)

        # --------------  Part 1. Soil Layering Randomization  -------------
        z_top = [0]  # depth of layer top
        z_bot = []  # depth of layer bottom
        z_mid = []  # midpoint depth of soil layers
        thk = []  # thickness

        while len(z_bot) == 0 or z_bot[-1] < self.z1:
            if use_Toros_layering:
                # Eq (2) of Toro (1995)
                rate = 1.98 * (z_top[-1] + 10.86) ** (-0.89)

                # The parameter for the Poisson process equals to 1/rate,
                # because Toro (1995) says the unit of `rate` is 1/m, and also
                # as written in page 40 of Harmon's UIUC PhD thesis (2017),
                # "the expected layer thickness at 1000 m is 239 m", which
                # confirms that lambda_ = 1 / rate.
                lamda_ = 1 / rate
                thk_rand = -1
                while thk_rand <= 0:  # to ensure thickness is always positive
                    thk_rand = rng.poisson(lamda_)  # draw random sample
            else:

                def func(x: np.ndarray | float) -> np.ndarray:
                    return SVM._thk_depth_func(x, z_top[-1])

                if len(thk) == 0:  # the first layer
                    ier = -6  # exit flag

                    # keeps trying until fsolve() properly converges
                    while ier != 1:
                        mean_thk, _info, ier, _msg = fsolve(
                            func,
                            z_top[-1] + 4.0,
                            full_output=True,
                        )
                else:  # the rest of the layers
                    ier = -6  # exit flag

                    # keeps trying until fzero() properly converges
                    while ier != 1:
                        mean_thk, _info, ier, _msg = fsolve(
                            func,
                            z_top[-1] + 4.0,
                            full_output=True,
                        )

                # Take the 0th element because the return value is an array:
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
                mean_thk = mean_thk[0]

                z_mid_temp = z_top[-1] + mean_thk / 2.0

                # Eq (8) of Shi & Asimaki (2018)
                std_thk = 0.951 * z_mid_temp**0.628

                # randomized thickness based on mean and std
                thk_rand = rng.normal(mean_thk, std_thk)

            # make sure each layer is at least 2 meters thick; too thin
            # layers are not realistic
            thk_rand = np.max([thk_rand, 2.0])

            if isinstance(thk_rand, (np.number, float, int)):
                thk.append(thk_rand)
            else:  # a single-element 1D numpy array
                thk.append(thk_rand[0])

            z_mid.append(z_top[-1] + thk_rand / 2.0)
            z_bot.append(z_top[-1] + thk_rand)
            z_top.append(z_top[-1] + thk_rand)

        # adjust thickness of last layer so that sum(thk) = z1
        thk[-1] = self.z1 - np.sum(thk[:-1])

        # update z_mid because thk has changed
        # (z_top and z_bot are not used below, so no need to update)
        z_mid = sr.thk2dep(np.array(thk), midpoint=True)

        # ----------------   Part 2   ------------------------------------
        # Calculate baseline Vs profile based on layering & smooth profile
        baseline_Vs = np.zeros(len(thk))
        Vs_analyt = self._base_profile[:, 1]
        thk_array_analyt = self._base_profile[:, 0]
        z_array_analyt = sr.thk2dep(thk_array_analyt, midpoint=False)

        for i in range(len(thk)):  # query Vs value where z = z_mid[j]
            # Note: _find_index_closest() is used here because it is more
            # appropriate for depth arrays with small layer thicknesses.
            index_value, ____ = self._find_index_closest(
                z_array_analyt, z_mid[i]
            )
            baseline_Vs[i] = Vs_analyt[index_value]

        # ---------------    Part 3    -----------------------------------
        # Generate random values for each layer based on the baseline profile

        # ******** 3.1. Toro (1995) coefficients *********
        # ******** These values come from Table 5 of Toro (1995) or Table 2.3
        # ******** of Kamai, Abrahamson, Silva (2013) PEER report.
        if self.Vs30 < SITE_CLASS_E_UPPER_VS30_M_S:  # site class E
            sigma_lnV = 0.37
            rho_0 = 0
            Delta = 5.0
            rho_200 = 0.50
            z_0 = 0
            b = 0.744
        elif self.Vs30 < SITE_CLASS_D_UPPER_VS30_M_S:  # site class D
            sigma_lnV = 0.31
            rho_0 = 0.99
            Delta = 3.9
            rho_200 = 0.98
            z_0 = 0
            b = 0.344
        elif self.Vs30 < SITE_CLASS_C_UPPER_VS30_M_S:  # site class C
            sigma_lnV = 0.27
            rho_0 = 0.97
            Delta = 3.8
            rho_200 = 1.00
            z_0 = 0
            b = 0.293
        else:
            # Site classes B and A (These values are intended for class B
            # only, but you can still produce a result for a class A profile.
            # The result just won't make sense.)
            sigma_lnV = 0.36
            rho_0 = 0.95
            Delta = 3.4
            rho_200 = 0.42
            z_0 = 0
            b = 0.063

        # ***** 3.2. Calculate "mu" and "sigma" of Vs as a function of depth **
        #     (Note: "mu" and "sigma" here are NOT the mean value and standard
        #     deviation of Vs, but rather the two parameters of the log-normal
        #     distribution that Vs is assumed to follow.)
        if not use_Toros_std:
            sigma_lognormal_Vs = (
                -7.769e-10 * Vs_analyt**3
                + 1.597e-06 * Vs_analyt**2
                - 0.0008724 * Vs_analyt
                + 0.4233
            )
        else:
            # From page 8 of Toro (1995):
            sigma_lognormal_Vs = sigma_lnV * np.ones(Vs_analyt.shape)

        # ****** 3.3. Generate random Vs values based on Toro's equations  ****
        Vs_hat = np.zeros([len(thk), 1])  # randomly realized Vs values
        Y = np.zeros([len(thk), 1])  # this "Y" here is the "Z" in Toro (1995)
        rng = np.random.RandomState([2 * seed])

        for i in range(len(thk)):  # loop through layers
            index_value, __ = SVM._find_index_closest(z_array_analyt, z_mid[i])

            # query sigma value where z = z_mid[j]:
            sigma_ = sigma_lognormal_Vs[index_value]

            if z_mid[i] > TORO_CORRELATION_REF_DEPTH_M:
                rho_z = rho_200
            else:
                ref_depth = TORO_CORRELATION_REF_DEPTH_M
                rho_z = rho_200 * ((z_mid[i] + z_0) / (ref_depth + z_0)) ** b

            rho_thk = rho_0 * np.exp(-thk[i] / Delta)
            rho_1L = (1 - rho_z) * rho_thk + rho_z

            if i == 0:  # for the first layer
                # generate a 1-by-nr_of_rand_profiles vector
                Y[i] = rng.normal(0, 1, (1, 1))
            else:  # for other layers
                Y[i] = rho_1L * Y[i - 1] + rng.normal(0, 1, (1, 1)) * np.sqrt(
                    1 - rho_1L**2
                )

            Vs_hat[i] = baseline_Vs[i] * np.exp(Y[i] * sigma_)

        # -------------  Part 4: Adjust Vs_profile  ----------------
        #     If the last layer of Vs_profile is less than 1000 m/s, add a
        #     1000 m/s layer at the very bottom.  '''
        Vs_profile = np.column_stack((thk, Vs_hat))
        if Vs_profile[-1, 1] < BEDROCK_VS_M_S:
            Vs_profile = np.vstack((Vs_profile, [0, BEDROCK_VS_M_S]))

        # -------------  Part 5: Plot Vs profile (optional) ---------------
        if show_fig is True:
            self._plot_additional_profile(Vs_profile, 'Stochastic')

        return Vs_profile

    @staticmethod
    def _thk_depth_func(
            thk: np.ndarray | float,
            z_top: np.ndarray | float,
    ) -> np.ndarray:
        """
        Calculate "right hand side" minus "left hand side".

        This is based on the given thk (thickness, in meter) and z_top (depth
        of layer top, in meter).

        Eq (7) of Shi & Asimaki (2018) Seismological Research Letters:

                thk = 1.125 * z_mid ^(0.620)

        Since z_mid = z_top + h/2.0,

                thk = 1.125 * (z_top + h/2.0)^(0.620)
        """
        thk = np.array(thk)
        return 1.125 * (z_top + thk / 2.0) ** 0.620 - thk

    @staticmethod
    def _find_index_closest(
            array: np.ndarray, value: float
    ) -> tuple[int, float]:
        """
        Find the index in ``array`` of the closest value to ``value``.

        NaN values within ``array`` are omitted implicitly.

        Parameters
        ----------
        array : np.ndarray
            Array from which to query the index. Must be 1D numpy array. It
            does NOT need to be sorted.
        value : float
            The value of interest.

        Returns
        -------
        index : int
            The index within ``array`` where the closest value is found.
        closest_value : float
            The closest value to ``value`` within ``array``.

        Raises
        ------
        ValueError
            When the value of the input argument is incorrect/invalid
        """
        array = np.array(array)
        if len(array) == 0:
            raise ValueError('The length of `array` needs to >= 0.')

        if array.ndim > 1:
            raise ValueError('`array` must be a 1D numpy array.')

        index = np.nanargmin(np.abs(array - value))
        closest_value = array[index]

        return index, closest_value
