import numpy as np
import scipy.fftpack
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import helper_generic as hlp
from . import helper_signal_processing as sig


def calc_z1_from_Vs30(Vs30_in_meter_per_sec):
    """
    Calculate z1 (basin depth) from Vs30. The correlation used here is
    z1 = 140.511 * exp(-0.00303 * Vs30), where the units of z1 and Vs30 are
    both SI units. This formula is documented in Section 2.5 (page 30) of the
    following PhD thesis:
        Shi, Jian (2019) "Improving Site Response Analysis for Earthquake
        Ground Motion Modeling." PhD thesis, California Institute of Technology
    """
    z1_in_m = 140.511 * np.exp(-0.00303 * Vs30_in_meter_per_sec)
    return z1_in_m


def stratify(vs_profile):
    """
    Divide layers of a Vs profile as necessary, according to the Vs values
    of each layer: if the layer thickness is more than Vs / 225.0, then divide
    the layer into more sublayers.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        The Vs profile. Must be 2D numpy array with two or five columns.

    Returns
    -------
    new_profile : numpy.ndarray
        The re-discretized Vs profile with the same number of columns as the
        input profile.
    """
    hlp.check_Vs_profile_format(vs_profile)

    h = vs_profile[:, 0]
    Vs = vs_profile[:, 1]
    if vs_profile.shape[1] > 2:
        five_columns = True
        xi = vs_profile[:, 2]
        rho = vs_profile[:, 3]
        mtrl = vs_profile[:, 4]
    else:
        five_columns = False

    if h[-1] == 0:
        flag = True
        Vs_end = Vs[-1]
        h = h[:-1]
        Vs = Vs[:-1]
        if five_columns:
            xi_end = xi[-1]
            rho_end = rho[-1]
            mtrl_end = mtrl[-1]
            xi = xi[:-1]
            rho = rho[:-1]
            mtrl = mtrl[:-1]
    else:
        flag = False

    n = len(h)
    h_temp = Vs / 225.0  # max freq = 15 Hz, and 15 points per wavelength

    h2 = []
    Vs2 = []
    xi2 = []
    rho2 = []
    mtrl2 = []
    counter = 1

    for j in range(n):
        if h_temp[j] >= h[j]:  # no need to create sub-layers
            nr_sublayer = 1
            h2.append(h[j])
            Vs2.append(Vs[j])
            if five_columns:
                xi2.append(xi[j])
                rho2.append(rho[j])
                mtrl2.append(mtrl[j])
        else:  # create sub-layers
            nr_sublayer = int(np.ceil(h[j] / h_temp[j]))
            if np.allclose(np.round(h[j] / h_temp[j]), h[j] / h_temp[j]):
                h2.extend([h_temp[j]] * nr_sublayer)
            else:
                h2.extend([h[j] / nr_sublayer] * nr_sublayer)

            Vs2.extend([Vs[j]] * nr_sublayer)
            if five_columns:
                xi2.extend([xi[j]] * nr_sublayer)
                rho2.extend([rho[j]] * nr_sublayer)
                mtrl2.extend([mtrl[j]] * nr_sublayer)

        counter += nr_sublayer

    if flag:
        h2.append(0)
        Vs2.append(Vs_end)
        if five_columns:
            xi2.append(xi_end)
            rho2.append(rho_end)
            mtrl2.append(mtrl_end)

    if five_columns:
        new_profile = np.column_stack((h2, Vs2, xi2, rho2, mtrl2))
    else:
        new_profile = np.column_stack((h2, Vs2))

    return new_profile


def query_Vs_at_depth(vs_profile, depth):
    """
    Query Vs values at given ``depth`` values from a Vs profile. If the given
    depth values happen to be at layer interfaces, return the Vs of the
    layer *below* the interface.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear-wave velocity profile, containing at least two columns:
           (1) thickness of layers
           (2) shear wave velocity of layers
    depth : float or numpy.array
        Value(s) of depths to query the Vs value at. Unit should be m.

    Returns
    -------
    vs_array : float or numpy.ndarray
        Vs values corresponding to the given depths. Its type depends on
        the type of ``depth``.
    is_scalar : bool
        Whether the given ``depth`` is a scalar or not.
    has_duplicate_values : bool
        Whether ``depth`` has duplicate values.
    is_sorted : bool
        Whether ``depth`` is sorted (ascending).
    """
    #------------- Check input type, input value, etc. ------------------------
    if isinstance(depth, (int, float, np.number)):
        is_scalar = True
        depth = np.array([depth])
        is_sorted = True
        has_duplicate_values = False
    elif isinstance(depth, np.ndarray):
        is_scalar = False
        hlp.assert_1D_numpy_array(depth, name='`depth`')
        if len(depth) == 1:
            is_sorted = True
            has_duplicate_values = False
        else:
            has_duplicate_values = np.any(np.diff(depth) == 0)
            is_sorted = np.all(np.diff(depth) >= 0)
    else:
        raise TypeError('`depth` needs to be a single number or numpy array.')

    if np.any(depth < 0):
        raise ValueError('Please provide non-negative `depth` values.')

    hlp.check_two_column_format(vs_profile, at_least_two_columns=True,
                                name='`vs_profile`')

    #------------------ Start querying ----------------------------------------
    thk_ref = vs_profile[:, 0]
    vs_ref  = vs_profile[:, 1]
    dep_ref = thk2dep(thk_ref, midpoint=False)

    indices = np.searchsorted(dep_ref, depth, side='right')
    indices_ = np.maximum(indices - 1, 0)  # index cannot be negative
    vs_queried = vs_ref[indices_]

    return vs_queried, is_scalar, has_duplicate_values, is_sorted


def query_Vs_given_thk(vs_profile, thk, n_layers=None, at_midpoint=True):
    """
    Query Vs values from a thickness array "``thk``". The starting point of
    querying is the ground surface.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear-wave velocity profile, containing at least two columns:
           (1) thickness of layers
           (2) shear wave velocity of layers
    thk : float or numpy.ndarray
        Thickness array, or a single value that means a constant thickness.
    n_layers : int or None
        Number of layers to query. This parameter has no effect if ``thk``
        is a numpy array (because the number of layers can be inferred
        from ``thk``).
    at_midpoint : bool
        If ``True``, the Vs values are queried at the mid-point depths of
        each layer. If ``False``, at the top of each layer.

    Return
    ------
    vs_array : numpy.ndarray
        Vs values corresponding to the given depths. Its type depends on
        ``as_profile``.
    thk_array : numpy.ndarray
        The constructed thickness array (if ``thk`` is a scalar), or ``thk``
        itself, if ``thk`` is already an array.
    """
    if not isinstance(thk, (int, float, np.number, np.ndarray)):
        raise TypeError('`thk` needs to be a scalar or a numpy array.')

    if not isinstance(thk, (int, float, np.number)):  # is a numpy array
        thk_array = thk.copy()
    else :  # need to construct an array
        if not isinstance(n_layers, (int, np.integer)):
            raise TypeError('If `thk` is a scalar, you need to provide '
                            '`n_layers` as an integer.')
        if n_layers <= 0:
            raise ValueError('`n_layers` should be positive.')
        thk_array = thk * np.ones(n_layers)

    depth_array = thk2dep(thk_array, midpoint=at_midpoint)
    vs_queried, _, _, _ = query_Vs_at_depth(vs_profile, depth_array)

    return vs_queried, thk_array


def plot_motion(
        accel, unit='m', fig=None, ax=None, title=None, figsize=(5, 6), dpi=100,
):
    """
    Plot acceleration, velocity, and displacement time history from a file
    name of acceleration data.

    Parameters
    ----------
    accel : str or numpy.ndarray
        Acceleration time history. Can be a file name, or a 2D numpy array with
        two columns (time and accel).
    unit : str
        Unit of acceleration for displaying on the y axis label
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    title : str
        Title of the figure (optional).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    (ax1, ax2, ax3) : tuple<matplotlib.axes._subplots.AxesSubplot>
        The axes objects of the three subplots.
    """
    if isinstance(accel, str):
        if not title: title = accel
        accel = np.loadtxt(accel)
    elif isinstance(accel, np.ndarray):
        if title is None: title = 'Ground motion'
    else:
        raise TypeError('"accel" must be a str or a 2-columned numpy array.')

    hlp.check_two_column_format(accel, '`accel`')

    t = accel[:, 0]
    a = accel[:, 1]

    PGA = np.max(np.abs(a))
    pga_index = np.argmax(np.abs(a))

    v, u = num_int(np.column_stack((t,a)))

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.2)
    ax.remove()  # remove axes to make room for subplot axes

    lw = 1.00
    vl = 'top' if a[pga_index] > 0 else 'bottom'

    if unit not in ['m', 'cm']:
        raise ValueError('"unit" can only be "m" or "cm".')

    accel_unit = 'gal' if unit == 'cm' else unit + '/s/s'
    veloc_unit = unit + '/s'
    displ_unit = unit

    ax1 = fig.add_subplot(311)
    ax1.plot(t, a, 'b', linewidth=lw)
    ax1.plot(t[pga_index], a[pga_index], 'ro', mfc='none', mew=1)
    t_ = t[int(np.min((pga_index + np.round(np.size(t)/40.), np.size(t))))]
    ax1.text(t_, a[pga_index], 'PGA = %.3g ' % PGA + accel_unit, va=vl)
    ax1.grid(ls=':')
    ax1.set_xlim(np.min(t), np.max(t))
    ax1.set_ylabel('Acceleration [' + accel_unit + ']')
    ax1.set_title(title)

    ax2 = fig.add_subplot(312)
    ax2.plot(t, v[:, 1], 'b', linewidth=lw)
    ax2.grid(ls=':')
    ax2.set_xlim(np.min(t), np.max(t))
    ax2.set_ylabel('Velocity [%s]' % veloc_unit)

    ax3 = fig.add_subplot(313)
    ax3.plot(t, u[:, 1], 'b', linewidth=lw)
    ax3.set_xlabel('Time [sec]')
    ax3.grid(ls=':')
    ax3.set_xlim(np.min(t), np.max(t))
    ax3.set_ylabel('Displacement [%s]' % displ_unit)

    fig.tight_layout(pad=0.3)

    return fig, (ax1, ax2, ax3)


def num_int(accel):
    """
    Performs numerical integration on acceleration to get velocity and
    displacement.

    Parameters
    ----------
    accel : numpy.ndarray
        Acceleration time history. Should have two columns. The 0th column is
        the time array, and the 1st column is the acceleration.

    Returns
    -------
    v : numpy.ndarray
        Velocity time history. Same shape as the input.
    u : numpy.ndarray
        Displacement time history. Same shape as the input.
    """
    hlp.check_two_column_format(accel, name='`accel`')

    t = accel[:, 0]
    a = accel[:, 1]

    dt = t[1] - t[0]
    v = np.cumsum(a) * dt
    u = np.cumsum(v) * dt

    v = np.column_stack((t, v))
    u = np.column_stack((t, u))

    return v, u


def num_diff(veloc):
    """
    Perform numerical integration on velocity to get acceleration.

    Parameters
    ----------
    veloc : numpy.ndarray
        Velocity time history. Should have two columns. The 0th column is
        the time array, and the 1st column is the velocity.

    Returns
    -------
    accel : numpy.ndarray
        Acceleration time history. Same shape as the input.
    """
    hlp.check_two_column_format(veloc, name='`veloc`')

    t = veloc[:, 0]
    v = veloc[:, 1]

    a = np.diff(v) / np.diff(t)
    a = np.append(np.array([0]), a)
    accel = np.column_stack((t, a))

    return accel


def find_f0(x):
    """
    Find f_0 in a frequency spectrum (i.e., the frequency corresponding to the
    initial peak).

    Parameters
    ----------
    x : numpy.ndarray
        A two-column numpy array. The 0th column is the "reference array", such
        as the frequency array, and the 1st column is the "value array" in
        which the peak is being searched.

    Returns
    -------
    f0 : float
        The value in the 0th column of x corresponding to the initial peak
        value in the 1st column of x.
    """
    hlp.check_two_column_format(x)

    freq = x[:, 0]
    ampl = x[:, 1]

    l = len(freq)

    current_flag = 0  # 1 means d(ampl)/d(freq) > 0; -1 means < 0
    previous_flag = 1

    for i in range(l-1):
        incre = ampl[i+1] - ampl[i]
        if incre > 0:
            current_flag = 1
        elif incre == 0:
            current_flag = 0
        else:  # incre < 0
            current_flag = -1

        if (current_flag <= 0) and (previous_flag == 1):
            break

        previous_flag = current_flag

    if i == l-2:  # if the loop above finishes without breaking
        i = i+1

    f0 = freq[i]

    return f0


def response_spectra(
        accel,
        T_min=0.01,
        T_max=10,
        n_pts=60,
        damping=0.05,
        show_fig=False,
        parallel=False,
        n_cores=None,
        subsample_interval=1,
):
    """
    Single-degree-of-freedom elastic response spectra, using the "exact"
    solution to the equation of motion (Section 5.2, Dynamics of Structures,
    Second Edition, by Anil K. Chopra).

    The input acceleration must be in m/s/s.

    Re-written in Python based on the MATLAB function written by Jongwon Lee.

    Parameters
    ----------
    accel : numpy.ndarray
        Input acceleration. Must have exactly two columns (time and accel.).
    T_min : float
        Minimum period value to calculate the response spectra. Unit: sec.
    T_max : float
        Maximum period value to calculate the response spectra. Unit: sec.
    n_pts : int
        Number of points you want for the response spectra. A high number
        increases computation time.
    damping : float
        Damping of the dash pots. Do not use "percent" as unit. Unit: 1 (i.e.,
        not percent).
    show_fig : bool
        Whether to show a figure of the response spectra.
    parallel : bool
        Whether to perform the calculation in parallel.
    n_cores : int or None
        Number of cores to use in parallel. Not necessary if not ``parallel``.
    subsample_interval : int
        The interval at which to subsample the input acceleration in the time
        domain. A higher number reduces computation time, but could lead to
        less accurate results.

    Returns
    -------
    (Tn, SA, PSA, SV, PSV, SD, fn) : tuple of 1D numpy.ndarray
        Periods, spectral acceleration, pseudo spectral acceleration, spectral
        velocity, pseudo spectral velocity, spectral displacement, and
        frequencies, respectively. Units: SI.
    """
    import itertools
    import multiprocessing as mp

    hlp.check_two_column_format(accel, name='`accel`')

    t = accel[::subsample_interval,0]
    a = accel[::subsample_interval,1]

    t_shift = np.roll(t,1)
    if not np.all((t-t_shift)[1:] - (t-t_shift)[1] < 1e-7):
        raise ValueError('Time array within "accel" must be evenly spaced.')

    dt = float(t[1] - t[0])
    len_a = len(a)

    Tn = np.logspace(np.log10(T_min), np.log10(T_max), n_pts)
    wn = 2. * np.pi / Tn   # [rad/sec] Natural freq
    xi = damping     # damping ratio in decimal
    wd = wn * np.sqrt(1. - xi**2.)  # damped freq.

    len_wd = len(wd)
    u_max = np.zeros(len_wd)
    ud_max = np.zeros(len_wd)
    utdd_max = np.zeros(len_wd)
    PSV = np.zeros(len_wd)
    PSA = np.zeros(len_wd)

    # A, B, C, and D in Table 5.2.1, page 169
    A = np.exp(-xi*wn*dt)*(xi/np.sqrt(1.-xi**2.)*np.sin(wd*dt)+np.cos(wd*dt))
    B = np.exp(-xi*wn*dt)*(1./wd*np.sin(wd*dt))
    C = 1./wn**2.*(2.*xi/wn/dt + np.exp(-xi*wn*dt)*(((1.-2.*xi**2.)/wd/dt-xi/np.sqrt(1.-xi**2.))*np.sin(wd*dt) - (1+2.*xi/wn/dt)*np.cos(wd*dt)))
    D = 1./wn**2.*(1 - 2.*xi/wn/dt + np.exp(-xi*wn*dt)*((2.*xi**2.-1)/wd/dt*np.sin(wd*dt)+2.*xi/wn/dt*np.cos(wd*dt)))

    # A', B', C', and D' in Table 5.2.1, page 169
    A_ = -np.exp(-xi*wn*dt)*(wn/np.sqrt(1.-xi**2.)*np.sin(wd*dt))
    B_ = np.exp(-xi*wn*dt)*(np.cos(wd*dt) - xi/np.sqrt(1.-xi**2.)*np.sin(wd*dt))
    C_ = 1./wn**2.*(-1./dt + np.exp(-xi*wn*dt)*((wn/np.sqrt(1.-xi**2.)+xi/dt/np.sqrt(1.-xi**2.))*np.sin(wd*dt)+1./dt*np.cos(wd*dt)))
    D_ = 1./wn**2./dt*(1 - np.exp(-xi*wn*dt)*(xi/np.sqrt(1.-xi**2.)*np.sin(wd*dt) + np.cos(wd*dt)))

    if parallel:
        p = mp.Pool(n_cores)
        result = p.map(
            _time_stepping,
            itertools.product(
                range(len_wd),
                [len_a],
                [A],
                [B],
                [C],
                [D],
                [A_],
                [B_],
                [C_],
                [D_],
                [wn],
                [wd],
                [xi],
                [a],
            )
        )
    else:
        result = []
        for i in range(len_wd):
            result.append(
                _time_stepping(
                    (i, len_a, A, B, C, D, A_, B_, C_, D_, wn, wd, xi, a)
                )
            )

    utdd_max, ud_max, u_max, PSA, PSV = zip(*result)  # transpose list of tuples

    SA = np.array(utdd_max)  # (Total or absolute) spectral acceleration
    SV = np.array(ud_max)    # (Relative) spectral velocity
    SD = np.array(u_max)     # (Relative) spectral displacement
    PSA = np.array(PSA)      # (Total) pseudo-spectral acceleration
    PSV = np.array(PSV)      # (Relative) pseudo-spectral velocity

    fn = 1./Tn

    if show_fig:
        plt.figure(figsize=(8, 4))

        plt.subplot(2, 2, (1, 2))
        plt.plot(t, a, lw=1)
        plt.xlabel('Time [sec]')
        plt.ylabel('Input accel. [m/s/s]')
        plt.grid(ls=':', lw=0.5)
        plt.xlim(np.min(t), np.max(t))

        plt.subplot(2, 2, 3)
        plt.semilogx(Tn, SA, lw=1.5)
        plt.xlabel('Period [sec]')
        plt.ylabel('Spec. accel. [m/s/s]')
        plt.grid(ls=':', lw=0.5)
        plt.xlim(T_min, T_max)

        plt.subplot(2, 2, 4)
        plt.semilogx(Tn, PSA, lw=1.5)
        plt.xlabel('Period [sec]')
        plt.ylabel('Pseudo S.A. [m/s/s]')
        plt.grid(ls=':', lw=0.5)
        plt.xlim(T_min, T_max)

        plt.tight_layout(pad=0.5)

    return Tn, SA, PSA, SV, PSV, SD, fn


from numba import jit

@jit(nopython=True, nogil=True)
def _time_stepping(para):
    """ Helper function for response_spectra() """
    i, len_a, A, B, C, D, A_, B_, C_, D_, wn, wd, xi, a = para

    u_ = np.zeros(len_a)
    ud_ = np.zeros(len_a)
    for j in range(len_a-1):
        u_[j+1] = u_[j]*A[i] + ud_[j]*B[i] + (-1)*a[j]*C[i] + (-1)*a[j+1]*D[i]
        ud_[j+1] = u_[j]*A_[i] + ud_[j]*B_[i] + (-1)*a[j]*C_[i] + (-1)*a[j+1]*D_[i]

    udd_ = -(2.*wn[i]*xi*ud_+wn[i]**2.*u_+a)
    utdd_ = udd_ + a

    u_max = np.max(np.abs(u_))
    ud_max = np.max(np.abs(ud_))
    utdd_max = np.max(np.abs(utdd_))
    PSV = u_max*wn[i]
    PSA = PSV*wn[i]

    return utdd_max, ud_max, u_max, PSA, PSV


def get_xi_rho(Vs, formula_type=3):
    """
    Generate damping (xi) and density (rho) from the given 2-column Vs profile.

    Parameters
    ----------
    Vs : numpy.ndarray
        1D Vs profile information (i.e., Vs only, no thickness information).
    formula_type : {1, 2, 3}
        Type of formula to determine damping from Vs.

        1 - Use this rule:
                + Vs < 250 m/s, xi = 5%;
                + 250 <= Vs < 750 m/s, xi = 2%;
                + Vs >= 750 m/s, xi = 1%;
        2 - Use the formula proposed in Taborda & Bielak (2013):

                 Qs = 10.5-16Vs+153Vs^2-103Vs^3+34.7Vs^4-5.29Vs^5+0.31Vs^6
                           (unit of Vs: km/s)
        3 - Use the rule by Archuleta and Liu (2004) USGS report:

                + Qs = 0.06Vs (Vs <= 1000 m/s)
                + Qs = 0.14Vs (1000 < Vs <= 2000 m/s)
                + Qs = 0.16Vs (Vs > 2000 m/s)

            Note: xi = 1 / (2 * Qs)
            (https://earthquake.usgs.gov/cfusion/external_grants/reports/04HQGR0059.pdf)

    Returns
    -------
    xi : float
        Damping ratio, having the same shape as the input Vs. (unit: 1)
    rho : float
        Soil mass density, calculated with this rule:
                  +    Vs < 200 m/s, rho = 1600
                  +  200 <= Vs < 800 m/s, rho = 1800
                  +    Vs >= 800 m/s, rho = 2000
               (Unit of rho: kg/m3)
    """
    hlp.assert_1D_numpy_array(Vs, '`Vs`')

    nr = len(Vs)  # number of Vs layers
    xi = np.zeros(nr)
    Qs = np.zeros(nr)
    rho = np.zeros(nr)

    if formula_type == 1:
        for i in range(nr):
            if Vs[i] < 250:
                xi[i] = 0.05
            elif Vs[i] < 750:
                xi[i] = 0.02
            else:
                xi[i] = 0.01
    elif formula_type == 2:
        Vs_ = Vs/1000.0  # unit conversion: from m/s to km/s
        Qs = 10.5 - 16*Vs_ + 153*Vs_**2. - 103*Vs_**3. + 34.7*Vs_**4. \
             - 5.29*Vs_**5. + 0.31*Vs_**6.
        Qs[np.where(Qs==0)] = 0.5  # subsitute Qs = 0 (if any) with 0.5 to make sure xi has upper bound 1.0
        xi = 1.0 / (2.0*Qs)
    elif formula_type == 3:
        for i in range(nr):
            if Vs[i] <= 1000:
                Qs[i] = 0.06 * Vs[i]
            elif Vs[i] <= 2000:
                Qs[i] = 0.14 * Vs[i]
            else:
                Qs[i] = 0.16 * Vs[i]
        xi = 1.0 / (2.0*Qs)

    for i in range(nr):
        if Vs[i] < 200:
            rho[i] = 1600
        elif Vs[i] < 800:
            rho[i] = 1800
        else:
            rho[i] = 2000

    return xi, rho


def calc_VsZ(profile, Z, option_for_profile_shallower_than_Z=1, verbose=False):
    """
    Calculate VsZ from the given Vs profile, where VsZ is the reciprocal of the
    weighted average travel time from Z meters deep to the ground surface.

    Parameters
    ----------
    profile : numpy.ndarray
        Vs profile, which should have at least two columns.
    Z : float
        The depth from which to calculate the weighted average travel time.
    option_for_profile_shallower_than_Z : {1, 2}
        If the provided `profile` has a total depth smaller than Z, then
        1 - assume last layer extends to Z meters
        2 - only use actual total depth
    verbose : bool
        Whether to show a warning message when the Vs profile is shallower
        than 30 m.

    Returns
    -------
    VsZ : float
        VsZ.

    Notes
    -----
    Rewritten into Python from MATLAB on 3/4/2017.
    """
    thick = profile[:, 0]  # thickness of each layer
    vs = profile[:, 1]  # Vs of each layer
    sl = 1. / vs  # slowness of each layer [s/m]
    total_thickness = sum(thick)  # total thickness of the soil profile

    depth = np.zeros(len(thick) + 1)
    for i in range(len(thick)):
        depth[i + 1] = depth[i] + thick[i]

    cumul_sl = 0.0  # make sure cumul_sl is float
    for i in range(len(thick)):
        if depth[i + 1] < Z:
            cumul_sl = cumul_sl + sl[i] * thick[i]    # cumulative Vs*thickness
        if depth[i + 1] >= Z:
            cumul_sl = cumul_sl + sl[i] * (thick[i] - (depth[i + 1] - Z))
            break

    if option_for_profile_shallower_than_Z == 1:  # assume last Vs extends to Z m
        if total_thickness < Z:
            if verbose is True:
                print(
                    f"The input profile doesn't reach Z = {Z:.2f} m.\n"
                    f"Assume last Vs value goes down to {Z:.2f} m."
                )
            cumul_sl = cumul_sl + sl[-1] * (Z - total_thickness)
            VsZ = float(Z)/cumul_sl
        else:
            VsZ = float(Z)/cumul_sl
    if option_for_profile_shallower_than_Z == 2:  # only use actual depth
        VsZ = np.min([total_thickness,Z])/float(cumul_sl)  # use actual depth

    return VsZ


def calc_Vs30(profile, option_for_profile_shallower_than_30m=1, verbose=False):
    """
    Calculate Vs30 from the given Vs profile, where Vs30 is the reciprocal of
    the weighted average travel time from Z meters deep to the ground surface.

    Parameters
    ----------
    profile : numpy.ndarray
        Vs profile, which should have at least two columns.
    option_for_profile_shallower_than_30m : {1, 2}
        If the provided `profile` has a total depth smaller than 30 m, then
        1 - assume last layer extends to 30 meters
        2 - only use actual total depth
    verbose : bool
        Whether to show a warning message when the Vs profile is shallower
        than 30 m.

    Returns
    -------
    Vs30 : float
        Vs30.

    Notes
    -----
    Rewritten into Python from MATLAB on 3/4/2017.
    """
    Vs30 = calc_VsZ(
        profile,
        30.0,
        option_for_profile_shallower_than_Z=option_for_profile_shallower_than_30m,
        verbose=verbose,
    )
    return Vs30


def plot_Vs_profile(
        vs_profile,
        fig=None,
        ax=None,
        figsize=(2.6, 3.2),
        dpi=100,
        title=None,
        label=None,
        c='k',
        lw=1.75,
        max_depth=None,
        **other_kwargs,
):
    """
    Plot a Vs profile from a 2D numpy array.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear-wave velocity profile, containing at least two columns:
           (1) thickness of layers
           (2) shear wave velocity of layers
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    title : str
        The title of the figure.
    label : str
        The text label for the legend.
    c : list<float> or str
        Line color.
    ls : float
        Line width.
    max_depth : float or None
        Maximum depth of the soil profile. If None, it is automatically
        determined from `vs_profile`. Note that setting max_depth to be smaller
        than the actual depth (determined in `vs_profile`) could make the plot
        look strange.
    other_kwargs :
        Other keyword arguments to be passed to matplotlib.pyplot.plot()

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    h_line : matplotlib.line.Line2D
        The line object.
    """
    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize=figsize, dpi=dpi)
    hlp.check_two_column_format(
        vs_profile, at_least_two_columns=True, name='`vs_profile`',
    )

    thk = vs_profile[:, 0]
    vs = vs_profile[:, 1]
    if not max_depth:
        zmax = np.sum(thk) + thk[0]
    else:
        zmax = max_depth

    x, y = _gen_profile_plot_array(thk, vs, zmax)

    h_line, = ax.plot(x, y, c=c, lw=lw, label=label, **other_kwargs)
    ax.set_xlim(0, np.max(vs) * 1.1)
    ax.set_ylim(zmax, 0)  # reversed Y axis
    ax.set_xlabel('Shear-wave velocity [m/s]', fontsize=12)
    ax.set_ylabel('Depth [m]', fontsize=12)
    ax.grid(color=[0.5]*3, ls=':', lw=.5)
    ax.set_axisbelow(True)  # put grid line below data lines
    if title: ax.set_title(title)

    if int(mpl.__version__[0]) <= 1:  # if matplotlib version is earlier than 2.0.0
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=7, integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10, integer=True))
    else:  # matplotlib version is 2.0.0 or newer
        pass  # because 2.0.0+ can automatically produce nicely spaced ticks

    return fig, ax, h_line  # return figure, axes, and line handles


def calc_basin_depth(vs_profile, bedrock_Vs=1000.0):
    """
    Query the depth of the basin as indicated in ``vs_profile``.
    The basin is defined as the material whose Vs is at least `bedrock_Vs`.

    Parameters
    ----------
    vs_profile : np.ndarray
        A 2D numpy array that represents a Vs profile.
    bedrock_Vs : float
        The shear-wave velocity that you consider as the bedrock.

    Returns
    -------
    basin_depth : float
        The basin depth. If no Vs values in the profile reaches
        ``bedrock_Vs``, return total depth (bottom) of the profile.
    """
    thk = vs_profile[:, 0]
    vs = vs_profile[:, 1]

    depth = thk2dep(thk, midpoint=False)
    assert(depth[0] == 0)  # assert that `depth` means the layer top
    basin_depth = -1
    for j in range(len(vs)):
        current_depth = depth[j]
        if vs[j] >= bedrock_Vs:
            basin_depth = current_depth
            break
    else:
        basin_depth = np.sum(thk)

    return basin_depth


def calc_z1(vs_profile):
    """
    Calculate z1 (the depth to Vs = 1000 m/s) from ``vs_profile``.

    Parameters
    ----------
    vs_profile : np.ndarray
        A 2D numpy array that represents a Vs profile.

    Returns
    -------
    z1 : float
        The depth to Vs = 1000 m/s.
    """
    return calc_basin_depth(vs_profile, bedrock_Vs=1000.0)


def _gen_profile_plot_array(thk, vs, zmax):
    """
    Generates (x, y) for plotting, from Vs profile information.

    Parameters
    ----------
    thk : numpy.ndarray
        Thickness array.
    vs : numpy.array
        Shear-wave velocity array.
    zmax : float
        Maximum depth desired.

    Returns
    -------
    x : numpy.ndarray
        The first array for plotting the Vs profile by ``plt.plot(x, y)``.
    y : numpy.ndarray
        The second array for plotting the Vs profile by ``plt.plot(x, y)``.
    """
    hlp.assert_1D_numpy_array(thk, name='`thk`')
    hlp.assert_1D_numpy_array(vs, name='`vs`')

    N = len(vs)
    x = np.zeros(2 * N)
    y = np.zeros(2 * N)

    dep = thk2dep(thk)

    for i in range(2*N):
        x[i] = vs[i//2]  # results look like [0,0,1,1,2,2,3,3, ...]
        if i+1 < 2*N:
            y[i] = dep[(i+1)//2]  # results look like [0,1,1,2,2,3,3, ...]
        else:
            y[i] = zmax

    return x, y


def thk2dep(thk, midpoint=False):
    """
    Convert a soil layer thickness array into depth array.

    Parameters
    ----------
    thk : numpy.ndarray
        1D numpy array of layer thickness.
    midpoint : bool
        If ``True``, the returned depth values are at the mid points of each
        layer. If False, the returned array are at the top of layers.

    Returns
    -------
    dep : numpy.ndarray
        Depth array.
    """
    hlp.assert_1D_numpy_array(thk, name='`thk`')

    L = len(thk)
    z_top = np.zeros(L) # create an array with same length as h
    z_mid = np.zeros(L)

    for i in range(1,L):  # the first element of 'z_top' remains zero
        z_top[i] = z_top[i-1] + thk[i-1]  # the last element of 'thk' is not used at all
        z_mid[i-1] = z_top[i-1] + thk[i-1]/2.0  # the last element of 'z_mid' is NaN

    if thk[-1] == 0:  # if the last layer thickness is unknown
        z_mid = z_mid[:-1]
    else:  # if known
        z_mid[-1] = z_top[-1] + thk[-1]/2.0

    if midpoint == False:
        return z_top
    else:
        return z_mid


def dep2thk(depth_array_starting_from_0, include_halfspace=True):
    """
    Convert a soil layer depth array into thickness array.

    Parameters
    ----------
    depth_array_starting_from_0 : numpy.array
        Needs to be a 1D numpy array.
    include_halfspace : bool
        Whether to include the last layer (i.e., "half space"), which always
        have 0 thickness.

    Returns
    -------
    h : numpy.array
        Thickness array.
    """
    hlp.assert_1D_numpy_array(
        depth_array_starting_from_0,
        name='`depth_array_starting_from_0`',
    )

    if depth_array_starting_from_0[0] != 0:
        raise ValueError('The 0th element of depth array must be 0.')

    h = np.zeros(len(depth_array_starting_from_0))

    for i in range(len(h)-1):
        h[i] = depth_array_starting_from_0[i+1] - depth_array_starting_from_0[i]

    if include_halfspace:
        return h
    else:
        return h[:-1]


def linear_tf(vs_profile, show_fig=True, freq_resolution=.05, fmax=30.):
    """
    Compute linear elastic transfer function from a given Vs profile.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear wave velocity profile. Can have 2 or 5 columns. If only 2 columns
        (no damping and density info), then damping and density are calculated
        automatically using ``get_xi_rho()``.
    show_fig : bool
        Whether or not to show figures of the amplification function.
    freq_resolution : float
        Frequency resolution of the frequency spectrum.
    fmax : float
        Maximum frequency of interest.

    Returns
    -------
    freq_array : numpy.ndarray
        Frequency array, in linear scale.
    AF_ro : numpy.ndarray
        Amplification with respect to rock outcrop.
    TF_ro : numpy.ndarray
        Transfer function (complex-valued) with respect to rock outcrop.
    f0_ro : float
        Fundamental frequency of rock outcrop amplification.
    AF_in : numpy.ndarray
        Amplification with respect to incident motion.
    TF_in : numpy.ndarray
        Transfer function (complex-valued) with respect to incident motion.
    AF_bh : numpy.ndarray
        Amplification with respect to borehole motion.
    TF_bh : numpy.ndarray
        Transfer function (complex-valued) with respect to borehole motion.
    f0_bh : float
        Fundamental frequency of rock outcrop amplification.

    Notes
    -----
    Vs profile can either include a "zero thickness" layer at last or not.

    Rewritten into Python between the winter of 2016 and the spring of 2017.
    """
    hlp.check_Vs_profile_format(vs_profile)

    h = vs_profile[:, 0]
    Vs = vs_profile[:, 1]
    try:
        xi = vs_profile[:, 2]  # damping ratio (unit: 1, not percent)
        rho = vs_profile[:, 3]  # mass density (unit: kg/m/m/m)
    except IndexError:  # if index 2/3 out of bounds, i.e., vs_profile only has 2 columns
        xi, rho = get_xi_rho(Vs)  # calculate xi and rho

    h_length = len(h)

    vs_star = np.multiply(Vs, np.sqrt(1 + 2 * 1j * xi))
    alpha_star = np.zeros(h_length - 1, dtype = np.complex_)
    for k in range(h_length-1):
        alpha_star[k] = float(rho[k]) * vs_star[k] / (rho[k+1] * vs_star[k+1])

    TF_size = int(np.floor_divide(fmax, freq_resolution))  # length of transfer function
    freq_array = np.linspace(freq_resolution, freq_resolution * TF_size, num=TF_size)

    TF_ro = np.ones(TF_size, dtype=np.complex_)
    TF_in = np.ones(TF_size, dtype=np.complex_)
    TF_bh = np.ones(TF_size, dtype=np.complex_)
    j_index = np.arange(h_length-2, -1, -1)

    for i, f in enumerate(freq_array):
        omega = 2 * np.pi * f
        k_star = np.divide(omega, vs_star)
        D = np.zeros(2 * 2 * (h_length-1), dtype=np.complex_).reshape(2, 2, h_length-1)
        E = np.zeros(4, dtype=np.complex_).reshape(2, 2)
        E[0, 0] = 1
        E[1, 1] = 1
        for j in j_index:
            D[0, 0, j] = .5 * ((1 + alpha_star[j]) * np.exp(1j * k_star[j] * h[j]))
            D[0, 1, j] = .5 * ((1 - alpha_star[j]) * np.exp(-1j * k_star[j] * h[j]))
            D[1, 0, j] = .5 * ((1 - alpha_star[j]) * np.exp(1j * k_star[j] * h[j]))
            D[1, 1, j] = .5 * ((1 + alpha_star[j]) * np.exp(-1j * k_star[j] * h[j]))
            E = np.dot(E, D[:, :, j])
        TF_ro[i] = 1./(E[0, 0] + E[0, 1])
        TF_in[i] = 2./(E[0, 0] + E[0, 1])
        TF_bh[i] = 2./(E[0, 0] + E[1, 0] + E[0, 1] + E[1, 1])
    AF_ro = np.absolute(TF_ro)
    AF_in = np.absolute(TF_in)
    AF_bh = np.absolute(TF_bh)

    f0_ro = find_f0(np.column_stack((freq_array, AF_ro)))
    f0_in = find_f0(np.column_stack((freq_array, AF_in)))
    f0_bh = find_f0(np.column_stack((freq_array, AF_bh)))

    if show_fig:
        xSize = 12
        ySize = 6
        fig = plt.figure(figsize=(xSize,ySize),edgecolor='k',facecolor='w')

        x_limits = [0, fmax]
        x_limits_log = [1e-1, fmax]
        ax = fig.add_subplot(3, 4, 1)
        plt.plot(freq_array, AF_ro, 'k')
        plt.ylabel('S to R.O.')
        plt.xlim(x_limits)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.title('Amplitude')
        plt.text(
            0.55,
            0.85,
            '$\mathregular{f_0}$ = %.2f Hz' % f0_ro,
            transform=ax.transAxes,
            fontweight='bold',
        )
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 5)
        plt.plot(freq_array, AF_in, 'k')
        plt.ylabel('S to Inci.')
        plt.xlim(x_limits)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.text(
            0.55,
            0.85,
            '$\mathregular{f_0}$ = %.2f Hz' % f0_in,
            transform=ax.transAxes,
            fontweight='bold',
        )
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 9)
        plt.plot(freq_array, AF_bh, 'k')
        plt.ylabel('S to B.H.')
        plt.xlabel('Frequency [Hz]')
        plt.xlim(x_limits)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.text(
            0.55,
            0.85,
            '$\mathregular{f_0}$= %.2f Hz' % f0_bh,
            transform=ax.transAxes,
            fontweight='bold',
        )
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 2)
        plt.plot(freq_array, np.unwrap(np.angle(TF_ro)), 'k')
        plt.xlim(x_limits)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.title('Phase angle (rad)')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 6)
        plt.plot(freq_array, np.unwrap(np.angle(TF_in)), 'k')
        plt.xlim(x_limits)
        plt.grid(color=[0.5] * 3, ls=':')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 10)
        plt.plot(freq_array, np.unwrap(np.angle(TF_bh)), 'k')
        plt.xlim(x_limits)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.xlabel('Frequency [Hz]')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 3)
        plt.semilogx(freq_array, AF_ro, 'k')
        plt.xlim(x_limits_log)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.title('Amplitude')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 7)
        plt.semilogx(freq_array, AF_in, 'k')
        plt.xlim(x_limits_log)
        plt.grid(color=[0.5] * 3, ls=':')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 11)
        plt.semilogx(freq_array, AF_bh, 'k')
        plt.xlabel('Frequency [Hz]')
        plt.xlim(x_limits_log)
        plt.grid(color=[0.5] * 3, ls=':')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 4)
        plt.semilogx(freq_array, np.unwrap(np.angle(TF_ro)), 'k')
        plt.xlim(x_limits_log)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.title('Phase angle (rad)')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 8)
        plt.semilogx(freq_array, np.unwrap(np.angle(TF_in)), 'k')
        plt.xlim(x_limits_log)
        plt.grid(color=[0.5] * 3, ls=':')
        ax.set_axisbelow(True)

        ax = fig.add_subplot(3, 4, 12)
        plt.semilogx(freq_array, np.unwrap(np.angle(TF_bh)), 'k')
        plt.xlim(x_limits_log)
        plt.grid(color=[0.5] * 3, ls=':')
        plt.xlabel('Frequency [Hz]')
        ax.set_axisbelow(True)

        plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)
        plt.show()

    return freq_array, AF_ro, TF_ro, f0_ro, AF_in, TF_in, AF_bh, TF_bh, f0_bh


def amplify_motion(
        input_motion,
        transfer_function_single_sided,
        taper=False,
        extrap_tf=True,
        deconv=False,
        show_fig=False,
        dpi=100,
        return_fig_obj=False,
):
    """
    Amplify (or de-amplify) ground motions in the frequency domain. The
    mathematical process behind this function is as follows:

        (1) INPUT = fft(input)
        (2) OUTPUT = INPUT * TRANS_FUNC
        (3) output = ifft(OUTPUT)

    Parameters
    ----------
    input_motion : numpy.ndarray
        Input ground motion to be amplficied. 2D numpy array of two columns.
    transfer_function_single_sided : tuple
        Complex-valued transfer function in frequency domain. It should be a
        two-element tuple, whose 0-th element is the frequency array, and the
        last element can be one of two options:
            (1) A complex-valued transformation, which should be a 1D complex
                numpy array
            (2) A tuple of (amplitude, phase) which represents the complex
                numbers. `amplitude` and `phase` both need to be 1D arrays and
                real-valued.
        The transfer function only needs to be "single-sided" (see note below.)
    taper : bool
        Whether to taper the input acceleration (using Tukey taper).
    extrap_tf : bool
        Whether to extrapolate the transfer function if its frequency range
        does not reach the frequency range implied by the input motion.
    deconv : bool
        If `False`, a regular amplification is performed; otherwise, the
        transfer function is "deducted" from the input motion ("deconvolution").
    show_fig : bool
        Whether or not to show an illustration of how the calculation is
        carried out.
    dpi : int
        Desired DPI for the figures; only effective when ``show_fig`` is
        ``True``.
    return_fig_obj : bool
        Whether or not to return figure and axis objects to the caller.

    Returns
    -------
    response : numpy.array
        The resultant ground motion in time domain. In the same format as
        ``input_motion``.
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.

    Note
    ----
    "Single sided":
        For example, the sampling time interval of `input_motion` is 0.01 sec,
        then the Nyquist frequency is 50 Hz. Therefore, the transfer function
        needs to contain information at least up to the Nyquist frequency,
        i.e., at least 0-50 Hz, and anything above 50 Hz will not affect the
        input motion at all.
    """
    assert(type(transfer_function_single_sided) == tuple)
    assert(len(transfer_function_single_sided) == 2)

    f_array, tf_ss = transfer_function_single_sided
    hlp.assert_1D_numpy_array(f_array, name='`f_array`')

    if isinstance(tf_ss, np.ndarray):
        assert(tf_ss.ndim == 1)
        assert(len(f_array) == len(tf_ss))
        amp_ss = np.abs(tf_ss)
        phase_ss = robust_unwrap(np.angle(tf_ss))
    elif isinstance(tf_ss, tuple):
        assert(len(tf_ss) == 2)
        amp_ss, phase_ss = tf_ss
        assert(amp_ss.ndim == 1)
        assert(phase_ss.ndim == 1)
        assert(len(amp_ss) == len(f_array))
        assert(len(phase_ss) == len(f_array))
    else:
        raise TypeError(
            'The last element of `transfer_function_single_sided` '
            'needs to be either a tuple of (amplitude, phase), or '
            'a complex-valued 1D numpy array.'
        )

    df, fmax, n, half_n, ref_f_array = _get_freq_interval(input_motion)
    if extrap_tf:
        if np.max(f_array) < fmax:  # `f_array` does not cover `fmax`
            phase_slope = phase_ss[-1] / f_array[-1]  # average slope of phase
            f_array = np.append(f_array, fmax)
            amp_ss = np.append(amp_ss, amp_ss[-1])
            # extrapolate phase knowing that it is a straight line in general:
            phase_ss = np.append(phase_ss, phase_slope * fmax)
        if np.min(f_array) > df:  # `f_array` does not cover fmin (i.e., `df`)
            f_array = np.append(df, f_array)
            amp_ss = np.append(1.0, amp_ss)
            phase_ss = np.append(0.0, phase_ss)
    else:  # keep downsampling `input_motion` until f_array covers the freq range
        while np.max(f_array) < fmax:
            input_motion = input_motion[::2, :]
            if input_motion.shape[0] <= 1:
                raise ValueError(
                    'The frequency range covered by '
                    '`transfer_function_single_sided` does '
                    'not cover the frequency range implied in '
                    'the `input_motion` (even after downsampling '
                    'the `input_motion`). Please make sure to '
                    'provide the correct frequency range.'
                )
            df, fmax, n, half_n, ref_f_array = _get_freq_interval(input_motion)

    # interpolate amplitude and phase (NOT the real/imag parts)
    amp_ss_interp = np.interp(ref_f_array, f_array, amp_ss)
    phase_ss_interp = np.interp(ref_f_array, f_array, phase_ss)
    tf_ss = amp_ss_interp * np.exp(1j * phase_ss_interp)  # reconstruction
    f_array = ref_f_array

    tf_ss[0] = np.real(tf_ss[0])  # see note (1) below
    if n % 2 == 0:  # if n is even
        tf_ss[-1] = np.real(tf_ss[-1])  # see note (2) below

    t, a = input_motion[:, 0], input_motion[:, 1]

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # Note (1) The value of transfer function when freq = 0 should be either 1
    #          or 2 (a real number). Because of discretization errors, tf_ss(1)
    #          is not a real number, but very close.
    #      (2) If n is even, the "mid-point" of the Fourier spectrum is real.
    #          So the transfer function array needs to be modified so that the
    #          ifft of the product of the two is real valued.
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    #-----------Create "double-sided" transfer function---------------
    if n % 2 == 1:
        tf_append = np.flipud(tf_ss[1:half_n])
        tf_append = np.conj(tf_append)
    else:
        tf_append = np.flipud(tf_ss[1:half_n-1])
        tf_append = np.conj(tf_append)

    tf_ds = np.append(tf_ss, tf_append)  # should have identical length as 'a'

    #------------Fourier spectrum of the input motion-----------------
    if taper:
        a_tapered = sig.taper_Tukey(a)
    else:
        a_tapered = a
    A = scipy.fftpack.fft(a_tapered)

    #------------Multiplication---------------------------------------
    if not deconv:
        RESP = A * tf_ds
    else:
        RESP = A / tf_ds

    #---------Inverse Fourier transform to get the response time history------
    resp = scipy.fftpack.ifft(RESP).real  # truncate imaginary part (very small)
    response = np.column_stack((t, resp))

    #---------Plot comparisons-------------------
    if show_fig:
        accel_in = np.column_stack((t, a))
        accel_out = np.column_stack((t, resp))
        fig, ax = _plot_site_amp(
            accel_in,
            accel_out,
            f_array,
            amp_ss_interp,
            phase_func_1col=phase_ss_interp,
            amplif_func_ylog=False,
            dpi=dpi,
        )
    else:
        fig = None
        ax = None

    if not return_fig_obj:
        return response
    else:
        return response, fig, ax

#%%----------------------------------------------------------------------------
def linear_site_resp(
        soil_profile,
        input_motion,
        boundary='elastic',
        show_fig=False,
        deconv=False,
):
    """
    Perform linear site response analysis.

    Parameters
    ----------
    soil_profile : numpy.ndarray or str
        1D Vs profile profile. If it is a string, it means the file name that
        contains the data. If it is a 2D array, it has the following format:

         +---------------+----------+---------+------------------+--------------+
         | Thickness [m] | Vs [m/s] | Damping | Density [kg/m^3] | Material No. |
         +===============+==========+=========+==================+==============+
         |      ...      |   ...    |   ...   |       ...        |      ...     |
         +---------------+----------+---------+------------------+--------------+
        (Damping unit: 1)
    input_motion : numpy.array or str
        Input motion in the time domain (with two columns). If it is a string,
        it means the file name that contains the data.
    boundary : {'elastic', 'rigid'}
        Boundary condition. "Elastic" means that the boundary allows waves to
        propagate through. "Rigid" means that all downgoing waves are reflected
        back to the soil medium.
    show_fig : bool
        Whether to show a figure that shows the result of the analysis
    deconv : bool
        Whether this operation is deconvolution. If True, it means that the
        ``input_motion`` will be propagated downwards, and the motion at the
        bottom will be collected.

    Returns
    -------
    response : numpy.ndarray
        The resultant ground motion in time domain. In the same format as
        ``input_motion``.
    transfer_function : tuple<numpy.ndarray>
        The transfer function (complex-valued) that corresponding to the given
        ``soil_profile`` and ``boundary``. It is a tuple of two 1D numpy arrays.
        The 0th array is frequency (real values) and the 1st array is the
        spectrum (complex values).

    Notes
    -----
    If you want to get rock-outcrop motions, choose "elastic"; if you
    want to get bedrock motions (or "total" motions), choose "rigid". If
    you happen to want incident motions, choose "elastic", and then
    manually divide the result by 2.

    Misc
    ----
    Original version in MATLAB: June, 2013.
    Re-written into Python in 4/5/2018.
    """
    if isinstance(soil_profile, str):
        soil_profile = np.genfromtxt(soil_profile)
    if isinstance(input_motion, str):
        input_motion = np.genfromtxt(input_motion)

    df, fmax, _, _, _ = _get_freq_interval(input_motion)

    #---------Get linear transfer function (complex valued)--------------
    factor = 1.05  # to ensure f_max of TF >= f_max inferred from `input_motion`
    fmax_ = fmax * factor
    df_ = df * factor  # to ensure consistent length of the output freq array
    tmp = linear_tf(soil_profile, show_fig=False, fmax=fmax_, freq_resolution=df_)
    if boundary == 'elastic':
        f_array, tf_ss = tmp[0], tmp[2]
    elif boundary == 'rigid':
        f_array, tf_ss = tmp[0], tmp[-2]
    else:
        raise ValueError('`boundary` should be "elastic" or "rigid".')

    transfer_function = (f_array, tf_ss)
    response = amplify_motion(
        input_motion, transfer_function, show_fig=show_fig, deconv=deconv,
    )

    return response, transfer_function


def _plot_site_amp(
        accel_in_2col,
        accel_out_2col,
        freq,
        amplif_func_1col,
        amplif_func_1col_smoothed=None,
        phase_func_1col=None,
        fig=None,
        figsize=(8, 4.5),
        dpi=100,
        amplif_func_ylog=True,
        input_accel_label='Input',
        output_accel_label='Output',
        amplification_ylabel='Amplification',
        phase_shift_ylabel='Phase shift [rad]',
):
    """
    Plot site amplification simulation results: input and output ground
    motions, amplification and phase factors, and Fourier amplitudes of the
    input and output motions.

    Parameters
    ----------
    accel_in_2col : numpy.ndarray
        Input acceleration as a two-column numpy array (time and acceleration).
    accel_out_2col : numpy.ndarray
        Output acceleration as a two-column numpy array (time and acceleration).
    freq : numpy.ndarray
        Frequency array (1D numpy array).
    amplif_func_1col : numpy.ndarray
        Amplification function (1D numpy array).
    amplif_func_1col_smoothed : numpy.ndarray
        Smoothed amplification function (1D numpy array).
    phase_func_1col : numpy.ndarray
        Phase function (1D numpy array).
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    amplif_func_ylog : bool
        If ``True``, show the Y axis of the amplification function subplot in
        the logarithmic scale. Otherwise, show that in the linear scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    """
    hlp.check_two_column_format(accel_in_2col, name='`accel_in`')
    hlp.check_two_column_format(accel_out_2col, name='`accel_out`')
    if freq is not None:
        hlp.assert_1D_numpy_array(freq, name='`freq`')
    else:
        amplif_func_1col = None  # set all to `None` to avoid potential errors
        amplif_func_1col_smoothed =None
        phase_func_1col = None
    if amplif_func_1col is not None:
        hlp.assert_1D_numpy_array(amplif_func_1col, name='`amplif_func_1col`')
    if amplif_func_1col_smoothed is not None:
        hlp.assert_1D_numpy_array(
            amplif_func_1col_smoothed, name='`amplif_func_1col_smoothed`',
        )
    if phase_func_1col is not None:
        hlp.assert_1D_numpy_array(phase_func_1col, name='`phase_func_1col`')

    t_in, accel_in = accel_in_2col.T
    t_out, accel_out = accel_out_2col.T
    assert(np.allclose(t_in, t_out, atol=1e-4))
    time = t_in

    fig, _ = hlp._process_fig_ax_objects(fig, ax=None, figsize=figsize, dpi=dpi)
    ax = []

    blue = '#3182bd'
    red = '#ef3b2c'
    alpha = 0.7
    alpha_ = 0.85

    ax_ = plt.subplot2grid((2, 3), (0, 0), fig=fig, colspan=3)
    if np.max(np.abs(accel_out)) >= np.max(np.abs(accel_in)):
        plt.plot(time, accel_out, c=blue, label=output_accel_label)
        plt.plot(time, accel_in, c=red, label=input_accel_label, alpha=alpha)
    else:
        plt.plot(time, accel_in, c=red, label=input_accel_label, alpha=alpha_)
        plt.plot(time, accel_out, c=blue, label=output_accel_label, alpha=alpha_)
    plt.grid(ls=':')
    plt.legend(loc='upper right')
    plt.xlim(np.min(time), np.max(time))
    plt.xlabel('Time [sec]')
    plt.ylabel('Acceleration')
    plt.title('Time histories')
    ax.append(ax_)

    if freq is not None:
        ax_ = plt.subplot2grid((2, 3), (1, 0), fig=fig)
        plt.semilogx(freq, amplif_func_1col, c=[0.1] * 3, label='Unsmoothed')
        if amplif_func_1col_smoothed is not None:
            plt.semilogx(freq, amplif_func_1col_smoothed, c='orange', label='Smoothed')
        plt.semilogx(freq, np.ones(len(freq)), '--', c='gray')
        if amplif_func_1col_smoothed is not None:
            plt.legend(loc='best')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(amplification_ylabel)
        plt.grid(ls=':')
        if amplif_func_ylog:
            plt.yscale('log')
        ax.append(ax_)

    if phase_func_1col is not None:
        ax_ = plt.subplot2grid((2, 3), (1, 1), fig=fig)
        plt.semilogx(freq, phase_func_1col, c=[0.1] * 3)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(phase_shift_ylabel)
        plt.grid(ls=':')
        ax.append(ax_)

    freq_in, ACCEL_IN = sig.fourier_transform(accel_in_2col).T
    freq_out, ACCEL_OUT = sig.fourier_transform(accel_out_2col).T

    ax_ = plt.subplot2grid((2, 3), (1, 2), fig=fig)
    plt.loglog(freq_out, ACCEL_OUT, c=blue, label=output_accel_label)
    plt.loglog(freq_in, ACCEL_IN, c=red, label=input_accel_label, alpha=alpha)
    plt.legend(loc='best')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier amplitudes')
    plt.grid(ls=':')
    ax.append(ax_)

    fig.tight_layout(pad=0.3, h_pad=0.4)

    return fig, ax


def compare_two_accel(
        input_accel,
        output_accel,
        smooth=True,
        input_accel_label='Input',
        output_accel_label='Output',
        amplification_ylabel='Amplification',
        phase_shift_ylabel='Phase shift [rad]',
):
    '''
    Compare two acceleration time histories: plot comparison figures showing
    two time histories and the transfer function between them.

    Parameters
    ----------
    input_accel : numpy.ndarray
        Input acceleration. (2 columns: time and acceleration.)
    output_accel : numpy.ndarray
        Output acceleration. (2 columns: time and acceleration.)
    smooth : bool
        In the comparison plot, whether or not to also show the smoothed
        amplification factor.
    input_accel_label : str
        The text label for the input acceleration in the figure legend.
    output_accel_label : str
        The text label for the output acceleration in the figure legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object created in this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object created in this function.
    '''
    hlp.assert_2D_numpy_array(input_accel, name='input_accel')
    hlp.assert_2D_numpy_array(output_accel, name='output_accel')

    t_in = input_accel[:, 0]
    a_in = input_accel[:, 1]
    t_out = output_accel[:, 0]
    a_out = output_accel[:, 1]

    t_ = _align_two_time_arrays(t_in, t_out)

    a_in_ = np.interp(t_, t_in, a_in)
    a_out_ = np.interp(t_, t_out, a_out)

    a_in_2col = np.column_stack((t_, a_in_))
    a_out_2col = np.column_stack((t_, a_out_))

    fs_in = sig.fourier_transform(a_in_2col, real_val=False)
    fs_out = sig.fourier_transform(a_out_2col, real_val=False)

    freq = np.real(fs_in[:, 0])  # values in fs_in[:, 0] all look like: 1.23 + 0j
    tf = fs_out[:, 1] / fs_in[:, 1]
    amp_func = np.abs(tf)
    phase_shift = np.angle(tf)

    if smooth:
        amp_func_smoothed = sig.log_smooth(amp_func, lin_space=False)
    else:
        amp_func_smoothed = None
    # END IF-ELSE

    fig, ax = _plot_site_amp(
        a_in_2col,
        a_out_2col,
        freq,
        amp_func,
        phase_func_1col=phase_shift,
        amplif_func_1col_smoothed=amp_func_smoothed,
        input_accel_label=input_accel_label,
        output_accel_label=output_accel_label,
        amplification_ylabel=amplification_ylabel,
        phase_shift_ylabel=phase_shift_ylabel,
    )

    return fig, ax


def _align_two_time_arrays(t1, t2):
    hlp.assert_1D_numpy_array(t1)
    hlp.assert_1D_numpy_array(t2)

    if len(t1) < 2 or len(t2) < 2:
        raise ValueError('Both time arrays need to have at least 2 elements.')
    # END IF

    dt1 = t1[1] - t1[0]
    dt2 = t2[1] - t2[0]

    tmax1 = t1[-1]
    tmax2 = t2[-1]

    dt = min(dt1, dt2)  # use the smaller time interval of the two time array
    n_time = int(np.ceil(max(tmax1, tmax2) / dt))
    tmax = dt * n_time  # use the larger end time of the two as the end time

    t_output = np.linspace(dt, tmax, num=n_time)
    return t_output


def _get_freq_interval(input_motion):
    """
    Get frequency interval from a 2-columed input motion.

    Parameters
    ----------
    input_motion : numpy.ndarray
        Ground motion in two columns (time, accel).

    Returns
    -------
    df : float
        Frequency interval.
    fmax : float
        Maximum frequency resolvable from the input motion.
    n : int
        Length of signal.
    half_n : int
        Length of the frequency array below the Nyquist frequency.
    f_array : numpy.ndarray
        Frequency array.
    """
    hlp.check_two_column_format(input_motion, name='`input_motion`')

    t = input_motion[:,0]
    a = input_motion[:,1]

    if len(a) <= 1:
        raise ValueError(
            '`input_motion` only contains one data point. '
            'It needs at least two data points.'
        )

    dt = float(t[1] - t[0])  # sampling time interval
    fs = 1.0/dt  # sampling freq
    n = len(a)
    df = fs/float(n)  # freq resolution

    if n % 2 == 1:
        half_n = int(np.ceil(n/2.0))
    else:
        half_n = int(n/2.0 + 1)

    fmax = half_n * df
    f_array = np.linspace(df, fmax, num=half_n)

    return df, fmax, n, half_n, f_array


def robust_unwrap(signal, discont=3.141592653589793):
    """
    Robustly unwrap a phase signal.

    Sometimes, due to numerical discreteness, the "jump" in the signal does not
    happen immediately between two adjacent signal points, but rather over
    several points. For example::

      [-2.98, -3.01, -3.03, -3.05, -3.08, -1.55, 0.34, 2.24, 3.11, 3.08, 3.06]
                                   trough ------------------ peak

    This would mess with ``numpy.unwrap()`` and tricks it to think that the
    points between the trough and peak (shown above) don't need unwrapping.

    This function deals with such situations by adjust the values underlined by
    "----" into the range of (trough, -3.1415927], so that ``numpy.unwrap()``
    can correctly identify the "jump".

    Parameters
    ----------
    signal : numpy.ndarray
        Input array. Only allows 1D arrays.
    discont : float, optional
        Maximum discontinuity between values, default is pi. Refer to the
        documentation of ``numpy.unwrap()``.

    Returns
    -------
    unwrapped : numpy.ndarray
        Unwrapped array.

    Notes
    -----
    An underlying assumption of this algorithm is that the phase angle (if
    correctly unwrapped) should be monotonically decreasing and all negative.
    This assumption is true for the transfer functions of 1D SH wave
    propagation, because the output waves always arrive later than the input
    waves.

    Also, this algorithm only works for "clean" phase signals, not noisy ones,
    because this algorithm assumes that all "upward jumps" that are not
    achieved within "one step" (i.e., between two adjacent signal points) are
    artifacts.  This assumption is not true for noisy phase signals, since some
    "upward jumps" are just noises, not artifacts.
    """
    hlp.assert_1D_numpy_array(signal)

    n = len(signal)
    signal_ = signal.copy()

    #-------1. Find anomalies (peaks and troughs that are too far apart)-------
    trough = -1  # to store index of trough point
    peak = -1    # to store index of peak point
    drawer = []  # to keep pairs of (trough, peak) that are 1+ apart in location
    flag = 0
    for i in range(1, n):
        if signal[i] <= signal[i-1]:  # trend is decreasing
            if flag == 1:  # just starts to dip from a previous climb
                flag = 0
                if peak > trough + 1:  # only keep such anomalies
                    drawer.append((trough, peak))
            trough = i
        else:
            peak = i
            flag = 1

    #--------2. Move in-between points into (signal[trough], -3.1415927]-------
    for pair in drawer:
        i1, i2 = pair
        length = i2 - i1 - 1
        scale = -np.pi - signal[i1]  # total amount needed to reach -3.1415927
        for j in range(i1, i2):  # adjust values proportionally
            steps = float(j - i1)
            signal_[j] = signal_[i1] + steps / length * scale

    unwrapped = np.unwrap(signal_, discont=discont)

    return unwrapped


def calc_damping_from_param(param, strain_in_unit_1, func_stress):
    """
    Calculate damping values from HH or MKZ parameters.

    Parameters
    ----------
    param : dict
        Soil model parameters.
    strain_in_unit_1 : numpy.ndarray
        An 1D array of strain values. Unit: 1 (not percent).
    func_stress : Python function
        The function to calculate stress from ``strain_in_unit_1`` and ``param``.

    Returns
    -------
    damping : numpy.ndarray
        Damping values corresponding to each strain values, in the unit of "1".
    """
    if not isinstance(param, dict):
        raise TypeError('`para` needs to be a dictionary.')

    hlp.assert_1D_numpy_array(strain_in_unit_1)

    Tau = func_stress(strain_in_unit_1, **param)
    damping = calc_damping_from_stress_strain(strain_in_unit_1, Tau, param['Gmax'])

    return damping


def calc_damping_from_stress_strain(strain_in_unit_1, stress, Gmax):
    """
    Calculates the damping curve from the given stress-strain curve.

    Parameters
    ----------
    strain_in_unit_1 : numpy.array
        Strain array in the unit of 1. 1D numpy array.
    stress : numpy.array
        Stress. 1D numpy array
    Gmax : float
        Maximum shear modulus, whose unit needs to be identical to that of the
        stress curve.

    Returns
    -------
    damping : numpy.ndarray
        A 1D numpy array of damping ratios, in the unit of "1".
    """
    strain = strain_in_unit_1
    n = len(strain)

    G_Gmax = calc_GGmax_from_stress_strain(strain, stress, Gmax=Gmax)

    area = np.zeros(n)
    damping = np.zeros(n)

    area[0] = 0.5 * (strain[0] * G_Gmax[0]) * strain[0]
    damping[0] = 2. / np.pi * (2. * area[0] / G_Gmax[0] / strain[0]**2 - 1)
    for i in range(1, n):
        area[i] = area[i-1] + 0.5 * (
                strain[i-1] * G_Gmax[i-1] + strain[i] * G_Gmax[i]
        ) * (strain[i] - strain[i-1])
        damping[i] = 2. / np.pi * (2 * area[i] / G_Gmax[i] / strain[i]**2 - 1)

    damping = np.maximum(damping, 0.0)  # make sure all damping values are >= 0
    return damping


def calc_GGmax_from_stress_strain(strain_in_unit_1, stress, Gmax=None):
    """
    Calculates G/Gmax curve from stress-strain curve.

    Parameters
    ----------
    strain_in_unit_1 : numpy.ndarray
        Strain array (a 1D numpy array). Unit: 1.
    stress : numpy.ndarray
        Stress array. Its unit can be arbitrary.
    Gmax : float
        Maximum shear modulus, whose unit needs to be identical to that of the
        stress curve. If not provided, it is automatically calculated from the
        stress-strain curve.

    Returns
    -------
    GGmax : numpy.ndarray
        A 1D numpy array of G/Gmax.
    """
    hlp.assert_1D_numpy_array(strain_in_unit_1)
    hlp.assert_1D_numpy_array(stress)

    if strain_in_unit_1[0] == 0:
        raise ValueError('`strain_in_unit_1` should start with a non-zero value.')

    if Gmax is None:
        Gmax = stress[0] / strain_in_unit_1[0]

    G = stress / strain_in_unit_1  # secant modulus
    GGmax = G / Gmax

    return GGmax


def _plot_damping_curve_fit(
        damping_data_in_pct,
        param,
        func_stress,
        fig=None,
        ax=None,
        min_strain_in_pct=1e-4,
        max_strain_in_pct=10,
):
    """
    Plot damping data and curve-fit results together.

    Parameters
    ----------
    damping_data_in_pct : numpy.ndarray
        Damping data. Needs to have 2 columns (strain and damping ratio). Both
        columns need to use % as unit.
    param : dict
        HH_x parameters.
    func_stress : Python function
        The function to calculate stress from strain and model parameters.
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    min_strain_in_pct : float
        Strain limits of the curve-fit result.
    max_strain_in_pct : float
        Strain limits of the curve-fit result.
    """
    fig, ax = hlp._process_fig_ax_objects(fig, ax)

    init_damping = damping_data_in_pct[0, 1]
    ax.semilogx(
        damping_data_in_pct[:, 0],
        damping_data_in_pct[:, 1],
        marker='o',
        alpha=0.8,
        label='data',
    )

    min_strain_in_1 = min_strain_in_pct / 100.0
    max_strain_in_1 = max_strain_in_pct / 100.0
    strain = np.logspace(np.log10(min_strain_in_1), np.log10(max_strain_in_1))
    damping_curve_fit = calc_damping_from_param(param, strain, func_stress)

    ax.semilogx(
        strain * 100,
        damping_curve_fit * 100 + init_damping,
        label='curve fit',
        alpha=0.8,
    )
    ax.legend(loc='best')
    ax.grid(ls=':')
    ax.set_xlabel('Strain [%]')
    ax.set_ylabel('Damping ratio [%]')

    return fig, ax


def fit_all_damping_curves(
        curves,
        func_fit_single_layer,
        func_stress,
        use_scipy=True,
        pop_size=800,
        n_gen=100,
        lower_bound_power=-4,
        upper_bound_power=6,
        eta=0.1,
        seed=0,
        show_fig=False,
        verbose=False,
        parallel=False,
        n_cores=None,
        save_fig=False,
        fig_filename=None,
        dpi=100,
        save_txt=False,
        txt_filename=None,
        sep='\t',
        func_serialize=None,
):
    """
    Perform damping curve fitting for multiple damping curves using the genetic
    algorithm provided in DEAP.

    Parameters
    ----------
    curves : numpy.ndarray or list<numpy.array>
        Can either be a 2D array in the "curve" format, or a list of individual
        damping curves.
        The "curve" format is as follows:
         +------------+--------+------------+-------------+-------------+--------+-----+
         | strain [%] | G/Gmax | strain [%] | damping [%] |  strain [%] | G/Gmax | ... |
         +============+========+============+=============+=============+========+=====+
         |    ...     |  ...   |    ...     |    ...      |    ...      |  ...   | ... |
         +------------+--------+------------+-------------+-------------+--------+-----+

        The G/Gmax information is redundant for this function.

    func_fit_single_layer : Python function
        A function which fits the model parameters to a single layer in
        ``curves``, such as ``hh.fit_HH_x_single_layer`` or
        ``mkz.fit_H4_x_single_layer``.
    func_stress : Python function
        A function to calculate the shear stress from model parameters.
    use_scipy : bool
        Whether to use the "differential_evolution" algorithm in scipy
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
        to perform the optimization. If ``False``, use the algorithm in the
        DEAP package.
    pop_size : int
        The number of individuals in a generation.
    n_gen : int
        Number of generations that the evolution lasts.
    lower_bound_power : float
        The 10-based power of the lower bound of all the 9 parameters. For
        example, if your desired lower bound is 0.26, then set this parameter
        to be numpy.log10(0.26).
    upper_bound_power : float
        The 10-based power of the upper bound of all the 9 parameters.
    eta : float
        Crowding degree of the mutation or crossover. A high ``eta`` will produce
        children resembling to their parents, while a low ``eta`` will produce
        solutions much more different.
    seed : int
        Seed value for the random number generator.
    show_fig : bool
        Whether to show the curve fitting results as a figure.
    verbose : bool
        Whether to display information (statistics of the loss in each
        generation) on the console.
    parallel : bool
        Whether to use parallel computing across layers, i.e., calculate
        multiple layers simultaneously.
    n_cores : int
        Number of CPU cores to use. If None, all cores are used. No effects
        if `parallel` is set to False.
    save_fig : bool
        Whether to save damping fitting figures to hard drive.
    fig_filename : str
        Full file name of the figure.
    dpi : int
        Desired figure resolution. Only effective when ``show_fig`` is ``True``.
    save_txt : bool
        Whether to save the fitted parameters as a text file.
    txt_filename : str
        The name of the text file to save the parameters to.
    sep : str
        Delimiter to separate columns of data in the output file.
    func_serialize : Python function
        The function to serialize the parameters from a dict into a list.
        Can be hh.serialize_params_to_array or mkz.serialize_params_to_array.

    Return
    ------
    params : list<dict>
        The best parameters for each layer found in the optimization.
    """
    if isinstance(curves, np.ndarray):
        _, curves_list = hlp.extract_from_curve_format(curves)
    elif isinstance(curves, list):
        if not all([isinstance(_, np.ndarray) for _ in curves]):
            raise TypeError(
                'If `curves` is a list, all its elements needs to be 2D numpy arrays.'
            )
        for j, curve in enumerate(curves):
            hlp.check_two_column_format(
                curve,
                name='Damping curve for layer #%d' % j,
                ensure_non_negative=True,
            )
        curves_list = curves
    else:
        raise TypeError(
            'Input data type of `curves` not recognized. '
            'Please check the documentation of this function.'
        )

    other_params = [
        (
            func_fit_single_layer,
            use_scipy,
            pop_size,
            n_gen,
            lower_bound_power,
            upper_bound_power,
            eta,
            seed,
            False,  # set `show_fig` to False; show all layers in subplots
            verbose,
        )
    ]

    if parallel:
        import itertools
        import multiprocessing
        p = multiprocessing.Pool(n_cores)
        params = p.map(
            _fit_single_layer_loop,
            itertools.product(curves_list, other_params),
        )
    else:
        params = []
        for curve in curves_list:
            params.append(_fit_single_layer_loop((curve, other_params[0])))

    if show_fig:
        ncol = 4
        nrow = int(np.ceil(len(curves_list) / ncol))
        fig = plt.figure(figsize=(ncol * 3, nrow * 3))
        for j, curve in enumerate(curves_list):
            ax = plt.subplot(nrow, ncol, j + 1)
            _plot_damping_curve_fit(
                curve, params[j], func_stress, fig=fig, ax=ax,
            )
        fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

    if show_fig and save_fig:
        fig.savefig(fig_filename, dpi=dpi)

    if save_txt:
        if func_serialize is None:
            raise ValueError(
                'Please provide a function to serialize the parameters into a lists.'
            )
        data_for_file = []
        for param in params:
            data_for_file.append(func_serialize(param))

        data_for_file__ = np.column_stack(tuple(data_for_file))
        np.savetxt(txt_filename, data_for_file__, fmt='%.6g', delimiter=sep)

    return params


def _fit_single_layer_loop(param):
    """
    Loop body to be passed to the parallel pool.

    Note: `func_fit_single_layer` can be:
        (1) helper_hh_model.fit_HH_x_single_layer(), or
        (2) helper_mkz_model.fit_H4_x_single_layer()
        etc.
    """
    damping_curve, other_params = param

    (
        func_fit_single_layer, use_scipy, pop_size, n_gen, lower_bound_power,
        upper_bound_power, eta, seed, show_fig, verbose,
    ) = other_params

    best_para = func_fit_single_layer(
        damping_curve,
        use_scipy=use_scipy,
        n_gen=n_gen,
        eta=eta,
        pop_size=pop_size,
        lower_bound_power=lower_bound_power,
        upper_bound_power=upper_bound_power,
        seed=seed,
        show_fig=show_fig,
        verbose=verbose,
        parallel=False,  # no par. within layers
    )

    return best_para


def ga_optimization(
        n_param,
        lower_bound,
        upper_bound,
        loss_function,
        damping_data,
        use_scipy=True,
        pop_size=100,
        n_gen=100,
        eta=0.1,
        seed=0,
        crossover_prob=0.8,
        mutation_prob=0.8,
        suppress_warnings=True,
        verbose=False,
        parallel=False,
        n_cores=None,
):
    """
    Perform a genetic algorithm (GA) process to fit the data.

    It supports any loss function (not even differentiable or parametric), as
    long as the loss function can map the model parameters to a loss value.

    The evolutionary process that this function can generate is a mutation
    and crossover within the specified bounds in a uniform fashion.

    Parameters
    ----------
    n_param : int
        Number of parameters in the model.
    lower_bound, upper_bound : float
        Lower and upper bound of the search range (i.e., range in which the
        evolution of parameter values are constraint). Note that all the
        model parameters share this range. You cannot have a different range
        for each parameter.
    loss_function : Python function
        Function to be minimized by the genetic algorithm. It should map a set
        of parameters to a loss value. It takes a tuple/list of all the
        parameters and the damping data as input, and it needs to return a
        single float.
    damping_data : numpy.ndarray
        Damping data for curve fitting. Needs to have two columns (strain and
        damping), and in the unit of 1 (not percent).
    use_scipy : bool
        Whether to use the "differential_evolution" algorithm implemented in
        scipy (https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.optimize.differential_evolution.html) to perform the optimization.
        If False, use the algorithm implemented in the DEAP package.
    pop_size : int
        The number of individuals in a generation. A larger number leads to
        potentially better curve-fitting, but a longer computing time.
    n_gen : int
        Number of generations that the evolution lasts. A larger number leads
        to potentially better curve-fitting, but a longer computing time.
        If ``use_scipy`` is True (using "differential evolution"), ``n_gen``
        means the maximum number of generations, i.e., the evolution could end
        early if no loss reduction is found.
    eta : float
        Crowding degree of the mutation or crossover. A high ``eta`` will produce
        children resembling to their parents, while a low ``eta`` will produce
        solutions much more different. (Only effective if ``use_scipy`` is
        ``False``.)
    seed : int
        Seed value for the random number generator.
    crossover_prob : float
        Probability of cross-over. "Cross-over" means producing offsprings
        from more than one parent. Larger values introduce more demographic
        diversity into the evolutionary process, which chould help escape the
        local minima, but at a cost of converging slower.
    mutation_prob : float
        Probability of mutation. Larger values introduce more
        demographic diversity into the evolutionary process, which could help
        escape the local minima, but at a cost of converging slower.
        (``mutation_prob`` is only effective when ``use_scipy`` is ``False``.)
    supress_warnings : bool
        Whether to suppress warning messages.
    verbose : bool
        Whether to display information (statistics of the loss in each
        generation) on the console.
    parallel : bool
        Whether to use parallel computing to simultaneously evaluate different
        individuals in a population. Note that different generations still
        evolve one after another. Only effective for the differential evolution
        for now. Also note that if using parallelization in differential
        evolution, you may need more generations to achieve the same
        optimization loss, because the best solution is being updated only once
        per generation.
    n_cores : int
        Number of CPU cores to use. If ``None``, all cores are used. No effects
        if ``parallel`` is set to ``False``.

    Returns
    -------
    opt_result : list or numpy.ndarray
        The optimization result: an array of parameters that gives the lowest
        loss.
    """
    if suppress_warnings:
        import warnings  # TODO: enable setting it from methods that calls this function
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    if use_scipy:
        from scipy.optimize import differential_evolution as diff_evol
        bounds = [(lower_bound, upper_bound)] * n_param
        n_cores = -1 if parallel and n_cores is None else 1
        popsize_multiplier = max(1, pop_size // n_param)
        result = diff_evol(
            loss_function,
            bounds,
            args=(damping_data,),
            recombination=crossover_prob,
            popsize=popsize_multiplier,
            seed=seed,
            maxiter=n_gen,
            disp=verbose,
            workers=n_cores,
        )
        if verbose:
            status = 'successful' if result.success else 'not successful'
            print('\nOptimization status: %s.' % status)

        opt_result = result.x

    else:
        import random

        import deap.creator
        import deap.base
        import deap.algorithms
        import deap.tools

        def loss_function__(param):  # because DEAP requires (loss, ) as output
            return (loss_function(param, damping_data), )

        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low]*size, [up]*size)]

        LB = lower_bound
        UB = upper_bound

        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)

        toolbox = deap.base.Toolbox()

        toolbox.register("attr_float", uniform, LB, UB, n_param)
        toolbox.register(
            "individual", deap.tools.initIterate, deap.creator.Individual, toolbox.attr_float,
        )
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", loss_function__)
        toolbox.register(
            "mate", deap.tools.cxSimulatedBinaryBounded, low=LB, up=UB, eta=eta,
        )
        toolbox.register(
            "mutate", deap.tools.mutPolynomialBounded,
            low=LB, up=UB, eta=eta, indpb=1.0/n_param,
        )
        toolbox.register("select", deap.tools.selTournament, tournsize=10)

        random.seed(seed)

        pop = toolbox.population(n=pop_size)
        hof = deap.tools.HallOfFame(1)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("Avg", np.mean)
        stats.register("Std", np.std)
        stats.register("Min", np.min)
        stats.register("Max", np.max)

        deap.algorithms.eaSimple(
            pop,
            toolbox,
            ngen=n_gen,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            stats=stats,
            halloffame=hof,
            verbose=verbose,
        )

        opt_result = list(hof[0])  # 0th element of "hall of fame" --> best param

    return opt_result
