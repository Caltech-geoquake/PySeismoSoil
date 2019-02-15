# Author: Jian Shi

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

from . import helper_generic as hlp

#%%----------------------------------------------------------------------------
def plot_motion(accel, unit='m', title=None, figsize=(5, 6), dpi=100):
    '''
    Plots acceleration, velocity, and displacement time history from a file
    name of acceleration data

    Parameters
    ----------
    accel : str or numpy.ndarray
        Acceleration time history. Can be a file name, or a 2D numpy array with
        two columns (time and accel)
    unit : str
        Unit of acceleration for displaying on the y axis label
    title : str
        Title of the figure (optional)
    figsize : tuple
        Figure size
    dpi : float
        DPI of the figure

    Returns
    -------
    fig :
        The figure object
    '''

    if isinstance(accel, str):
        if not title: title = accel
        accel = np.loadtxt(accel)
    elif isinstance(accel, np.ndarray):
        if not title: title = 'accel.'
    else:
        raise TypeError('"accel" must be a str or a 2-columned numpy array.')

    hlp.check_two_column_format(accel, '`accel`')

    t = accel[:,0]
    a = accel[:,1]

    PGA = np.max(np.abs(a))
    pga_index = np.argmax(np.abs(a))

    v, u = num_int(np.column_stack((t,a)))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.2)

    lw = 1.00
    vl = 'top' if a[pga_index] > 0 else 'bottom'

    if unit not in ['m', 'cm']:
        raise ValueError('"unit" can only be "m" or "cm".')

    accel_unit = 'gal' if unit == 'cm' else unit + '/s/s'
    veloc_unit = unit + '/s'
    displ_unit = unit

    plt.subplot(311)
    plt.plot(t, a, 'b', linewidth=lw)
    plt.plot(t[pga_index], a[pga_index], 'ro', mfc='none', mew=1)
    t_ = t[int(np.min((pga_index + np.round(np.size(t)/40.), np.size(t))))]
    plt.text(t_, a[pga_index], 'PGA = %.3g ' % PGA + accel_unit, va=vl)
    plt.grid(ls=':')
    plt.xlim(np.min(t), np.max(t))
    plt.ylabel('Acceleration [' + accel_unit + ']')
    plt.title(title)

    plt.subplot(312)
    plt.plot(t, v[:,1], 'b', linewidth=lw)
    plt.grid(ls=':')
    plt.xlim(np.min(t), np.max(t))
    plt.ylabel('Velocity [%s]' % veloc_unit)

    plt.subplot(313)
    plt.plot(t, u[:,1], 'b', linewidth=lw)
    plt.xlabel('Time [sec]')
    plt.grid(ls=':')
    plt.xlim(np.min(t), np.max(t))
    plt.ylabel('Displacement [%s]' % displ_unit)

    plt.tight_layout(pad=0.3)

    return fig

#%%----------------------------------------------------------------------------
def num_int(accel):
    '''
    Performs numerical integration on acceleration to get velocity and
    displacement.

    Parameter
    ---------
    accel : numpy.ndarray
        Acceleration time history. Should have two columns. The 0th column is
        the time array, and the 1st column is the acceleration.

    Returns
    -------
    v : numpy.ndarray
        Velocity time history. Same shape as the input.
    u : numpy.ndarray
        Displacement time history. Same shape as the input.
    '''

    hlp.check_two_column_format(accel, name='`accel`')

    t = accel[:, 0]
    a = accel[:, 1]

    dt = t[1] - t[0]
    v = np.cumsum(a) * dt
    u = np.cumsum(v) * dt

    v = np.column_stack((t, v))
    u = np.column_stack((t, u))

    return v, u

#%%----------------------------------------------------------------------------
def num_diff(veloc):
    '''
    Perform numerical integration on velocity to get acceleration.

    Parameter
    ----------
    veloc : numpy.ndarray
        Velocity time history. Should have two columns. The 0th column is
        the time array, and the 1st column is the velocity.

    Returns
    -------
    accel : numpy.ndarray
        Acceleration time history. Same shape as the input.
    '''

    hlp.check_two_column_format(veloc, name='`veloc`')

    t = veloc[:, 0]
    v = veloc[:, 1]

    a = np.diff(v) / np.diff(t)
    a = np.append(np.array([0]), a)
    accel = np.column_stack((t, a))

    return accel

#%%----------------------------------------------------------------------------
def find_f0(x):
    '''
    Find f_0 in a frequency spectrum (i.e., the frequency corresponding to the
    initial peak).

    Parameter
    ---------
    x : numpy.ndarray
        A two-column numpy array. The 0th column is the "reference array", such
        as the frequency array, and the 1st column is the "value array" in
        which the peak is being searched.

    Returns
    -------
    f0 : float
        The value in the 0th column of x corresponding to the initial peak
        value in the 1st column of x
    '''

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

#%%----------------------------------------------------------------------------
def response_spectra(accel, T_min=0.01, T_max=10, n_pts=60, damping=0.05,
                     show_fig=False, parallel=False, n_cores=None,
                     subsample_interval=1):
    '''
    Single-degree-of-freedom elastic response spectra, using the "exact"
    solution to the equation of motion (Section 5.2, Dynamics of Structures
    by Chopra).

    The input acceleration must be in m/s/s.

    Re-written in Python based on the MATLAB function written by Jongwon Lee.

    Parameters
    ----------
    accel : numpy.ndarray
        Input acceleration. Must have exactly two columns (time and accel.).
    T_min : float
        Minimum period value to calculate the response spectra
    T_max : float
        Maximum period value to calculate the response spectra
    n_pts : int
        Number of points you want for the response spectra. A high number
        increases computation time.
    damping : float
        Damping of the dash pots. Do not use "percent" as unit. Use 1-based.
    show_fig : bool
        Whether to show a figure of the response spectra
    parallel : bool
        Whether to perform the calculation in parallel
    n_cores : int or None
        Number of cores to use in parallel. Not necessary if not `parallel`.
    subsample_interval : int
        The interval at which to subsample the input acceleration. A higher
        number saves computation time.

    Returns
    -------
    Tn, SA, PSA, SV, PSV, SD, fn : numpy.ndarray
        Periods, spectral acceleration, pseudo spectral acceleration, spectral
        velocity, pseudo spectral velocity, spectral displacement, and
        frequencies, respectively.
    '''

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

    A = np.exp(-xi*wn*dt)*(xi/np.sqrt(1.-xi**2.)*np.sin(wd*dt)+np.cos(wd*dt))
    B = np.exp(-xi*wn*dt)*(1./wd*np.sin(wd*dt))
    C = 1./wn**2.*(2.*xi/wn/dt + np.exp(-xi*wn*dt)*(((1.-2.*xi**2.)/wd/dt-xi/np.sqrt(1.-xi**2.))*np.sin(wd*dt) - (1+2.*xi/wn/dt)*np.cos(wd*dt)))
    D = 1./wn**2.*(1 - 2.*xi/wn/dt + np.exp(-xi*wn*dt)*((2.*xi**2.-1)/wd/dt*np.sin(wd*dt)+2.*xi/wn/dt*np.cos(wd*dt)))

    A_ = -np.exp(-xi*wn*dt)*(wn/np.sqrt(1.-xi**2.)*np.sin(wd*dt))
    B_ = np.exp(-xi*wn*dt)*(np.cos(wd*dt) - xi/np.sqrt(1.-xi**2.)*np.sin(wd*dt))
    C_ = 1./wn**2.*(-1./dt + np.exp(-xi*wn*dt)*((wn/np.sqrt(1.-xi**2.)+xi/dt/np.sqrt(1.-xi**2.))*np.sin(wd*dt)+1./dt*np.cos(wd*dt)))
    D_ = 1./wn**2./dt*(1 - np.exp(-xi*wn*dt)*(xi/np.sqrt(1.-xi**2.)*np.sin(wd*dt) + np.cos(wd*dt)))

    if parallel:
        p = mp.Pool(n_cores)
        result = p.map(_time_stepping,
                       itertools.product(range(len_wd),[len_a],[A],[B],[C],[D],
                                         [A_],[B_],[C_],[D_],[wn],[wd],[xi],[a]
                                         )
                       )
    else:
        result = []
        for i in range(len_wd):
            result.append(_time_stepping((i,len_a,A,B,C,D,A_,B_,C_,D_,wn,wd,xi,a)))

    utdd_max, ud_max, u_max, PSA, PSV = zip(*result)  # transpose list of tuples

    SA = np.array(utdd_max)  # (Total or Absolute) Spectral Acceleration in
    SV = np.array(ud_max)    # (Relative) Spectral Velocity
    SD = np.array(u_max)     # (Relative) Spectral Displacement
    PSA = np.array(PSA)      # (Total) Pseudo-spectral Acceleration in
    PSV = np.array(PSV)      # (Relative) Pseudo-spectral Velocity

    fn = 1./Tn

    if show_fig:
        plt.figure(figsize=(8,4))

        plt.subplot(2,2,(1,2))
        plt.plot(t,a,lw=1)
        plt.xlabel('Time [sec]')
        plt.ylabel('Input accel. (m/s/s)')
        plt.grid(ls=':')
        plt.xlim(np.min(t),np.max(t))

        plt.subplot(2,2,3)
        plt.semilogx(Tn,SA,lw=1.5)
        plt.xlabel('Period [sec]')
        plt.ylabel('Spec. accel. (m/s/s)')
        plt.grid(ls=':')
        plt.xlim(T_min,T_max)

        plt.subplot(2,2,4)
        plt.semilogx(Tn,PSA,lw=1.5)
        plt.xlabel('Period [sec]')
        plt.ylabel('Pseudo S.A. (m/s/s)')
        plt.grid(ls=':')
        plt.xlim(T_min,T_max)

        plt.tight_layout(pad=0.5)

    return Tn, SA, PSA, SV, PSV, SD, fn

#%%----------------------------------------------------------------------------
from numba import jit

@jit(nopython=True, nogil=True)
def _time_stepping(para):
    '''
    Helper function for response_spectra()
    '''

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

#%%----------------------------------------------------------------------------
def get_xi_rho(Vs, formula_type=3):
    '''
    Generate damping (xi) and density (rho) from the given 2-column Vs profile.

    Parameters
    ----------
    Vs : numpy.ndarray
        1D Vs profile information (i.e., Vs only, no thickness information)
    formula_type : {1, 2, 3}
        Type of formula to determine damping from Vs.

        1 - Use this rule: Vs < 250 m/s, xi = 5%
                           250 <= Vs < 750 m/s, xi = 2%
                           Vs >= 750 m/s, xi = 1%
        2 - Use the formula proposed in Taborda & Bielak (2013):
                 Qs = 10.5-16Vs+153Vs^2-103Vs^3+34.7Vs^4-5.29Vs^5+0.31Vs^6
                           (unit of Vs: km/s)
        3 - Use the rule by Archuleta and Liu (2004) USGS report
            (https://earthquake.usgs.gov/cfusion/external_grants/reports/04HQGR0059.pdf)
                              0.06Vs (Vs <= 1000 m/s)
                         Qs = 0.14Vs (1000 < Vs <= 2000 m/s)
                              0.16Vs (Vs > 2000 m/s)
            Note: xi = 1 / (2 * Qs)

    Returns
    -------
    xi : float
        Damping ratio, having the same shape as the input Vs. (unit: 1)
    rho : float
        Soil mass density, calculated with this rule:
                      Vs < 200 m/s, rho = 1600
                    200 <= Vs < 800 m/s, rho = 1800
                      Vs >= 800 m/s, rho = 2000
               (Unit of rho: kg/m3)
    '''

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

#%%----------------------------------------------------------------------------
def calc_VsZ(profile, Z, option_for_profile_shallower_than_Z=1, verbose=False):
    '''
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
        Whether to show a warning message for the situation above

    Returns
    -------
    VsZ : float
        VsZ

    (Rewritten into Python from MATLAB on 3/4/2017)
    '''

    thick = profile[:,0]  # thickness of each layer
    vs = profile[:,1]  # Vs of each layer
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
            cumul_sl = cumul_sl + sl[i] * (thick[i]- (depth[i + 1] - Z))
            break

    if option_for_profile_shallower_than_Z == 1:  # assume last Vs extends to Z m
        if total_thickness < Z:
            if verbose is True:
                print("The input profile doesn't reach Z = %.2f m.\n"\
                      "Assume last Vs value goes down to %.2f m." % (Z, Z))
            cumul_sl = cumul_sl + sl[-1] * (Z - total_thickness)
            VsZ = float(Z)/cumul_sl
        else:
            VsZ = float(Z)/cumul_sl
    if option_for_profile_shallower_than_Z == 2:  # only use actual depth
        VsZ = np.min([total_thickness,Z])/float(cumul_sl)  # use actual depth

    return VsZ

#%%----------------------------------------------------------------------------
def calc_Vs30(profile, option_for_profile_shallower_than_30m=1, verbose=False):
    '''
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
        Whether to show a warning message for the situation above

    Returns
    -------
    Vs30 : float
        Vs30

    (Rewritten into Python from MATLAB on 3/4/2017)
    '''
    Vs30 = calc_VsZ(profile, 30.0,
                    option_for_profile_shallower_than_Z=\
                    option_for_profile_shallower_than_30m,
                    verbose=verbose)
    return Vs30

#%%----------------------------------------------------------------------------
def plot_Vs_profile(vs_profile, fig=None, ax=None, figsize=(2.6, 3.2), dpi=100,
                  title=None, label=None, c='k', lw=1.75, max_depth=None,
                  **other_kwargs):
    '''
    Plots a Vs profile from a 2D numpy array.

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear-wave velocity profile, containing at least two columns:
           (1) thickness of layers
           (2) shear wave velocity of layers
    fig, ax : mpl.figure.Figure, mpl.axes._subplots.AxesSubplot
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple<float>
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter). If None, the figure size will be
        automatically determined from the number of distinct categories in x.
    dpi : int
        Display resolution of the figure
    title : str
        The title of the figure
    label : str
        The text label for the legend
    c : list<float> or str
        Line color
    ls : float
        Line width
    max_depth : float or None
        Maximum depth of the soil profile. If None, it is automatically
        determined from `vs_profile`. Note that setting max_depth to be smaller
        than the actual depth (determined in `vs_profile`) could make the plot
        look strange.
    other_kwargs :
        Other keyword arguments to be passed to matplotlib.pyplot.plot()

    Returns
    -------
    fig, ax, h_line: the figure object, axes object, and line object
    '''

    if fig is None:
        fig = pl.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='k')
    else:
        pl.figure(fig.number)

    if ax is None:
        ax = plt.axes()
    else:
        ax = ax

    thk = vs_profile[:,0]
    vs = vs_profile[:,1]
    if not max_depth:
        zmax = np.sum(thk) + thk[0]
    else:
        zmax = max_depth

    x, y = _gen_profile_plot_array(thk, vs, zmax)

    h_line, = plt.plot(x, y, c=c, lw=lw, label=label, **other_kwargs)
    plt.xlim(0, np.max(vs) * 1.1)
    plt.ylim(zmax, 0)  # reversed Y axis
    plt.xlabel('Shear wave velocity [m/s]', fontsize=12)
    plt.ylabel('Depth [m]', fontsize=12)
    plt.grid(color=[0.5]*3, ls=':', lw=.5)
    ax.set_axisbelow(True)  # put grid line below data lines
    if title: ax.set_title(title)

    if int(mpl.__version__[0]) <= 1:  # if matplotlib version is earlier than 2.0.0
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=7, integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10, integer=True))
    else:  # matplotlib version is 2.0.0 or newer
        pass  # because 2.0.0+ can automatically produce nicely spaced ticks

    return fig, ax, h_line  # return figure, axes, and line handles

#%%----------------------------------------------------------------------------
def _gen_profile_plot_array(thk, vs, zmax):
    '''
    Generates (x, y) for plotting, from Vs profile information

    Parameters
    ----------
    thk : numpy.ndarray
        Thickness array
    vs : numpy.array
        Shear-wave velocity array
    zmax : float
        Maximum depth desired

    Returns
    -------
    x, y : numpy.ndarray
        Two numpy arrays for plotting the Vs profile: plt.plot(x, y)
    '''

    hlp.assert_1D_numpy_array(thk)
    hlp.assert_1D_numpy_array(vs)

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

#%%----------------------------------------------------------------------------
def thk2dep(thk, midpoint=False):
    '''
    Converts a soil layer thickness array into depth array.

    Parameters
    ----------
    thk : numpy.ndarray
        1D numpy array of layer thickness
    midpoint : bool
        Whether or not the returned depth array means the mid points of each
        layer (if False, the returned array means the top of layers).

    Returns
    -------
    dep : numpy.ndarray
        Depth array
    '''

    hlp.assert_1D_numpy_array(thk)

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

#%%----------------------------------------------------------------------------
def dep2thk(depth_array_starting_from_0):
    '''
    Converts a soil layer depth array into thickness array.

    Parameter
    ---------
    depth_array_starting_from_0 : numpy.array
        Needs to be a 1D numpy array

    Returns
    -------
    h : numpy.array
        Thickness array
    '''

    hlp.assert_1D_numpy_array(depth_array_starting_from_0)

    if depth_array_starting_from_0[0] != 0:
        raise ValueError('The 0th element of depth array must be 0.')

    h = np.zeros(len(depth_array_starting_from_0))

    for i in range(len(h)-1):
        h[i] = depth_array_starting_from_0[i+1] - depth_array_starting_from_0[i]

    return h

#%%----------------------------------------------------------------------------
def linear_tf(vs_profile, show_fig=True, freq_resolution=.05, fmax=30.):
    '''
    Computes linear elastic transfer function from a given Vs profile

    Parameters
    ----------
    vs_profile : numpy.ndarray
        Shear wave velocity profile. Can have 2 or 5 columns. If only 2 columns
        (no damping and density info), then damping and density are calculated
        automatically using get_xi_rho()
    show_fig : bool
        Whether or not to show figures of the amplification function
    freq_resolution : float
        Frequency resolution of the frequency spectrum
    fmax : float
        Maximum frequency of interest

    Returns
    -------
    freq_array : numpy.ndarray
        Frequency array, in linear scale
    AF_ro : numpy.ndarray
        Amplification with respect to rock outcrop
    TF_ro : numpy.ndarray
        Transfer function (complex-valued) with respect to rock outcrop
    f0_ro : float
        Fundamental frequency of rock outcrop amplification
    AF_in : numpy.ndarray
        Amplification with respect to incident motion
    TF_in : numpy.ndarray
        Transfer function (complex-valued) with respect to incident motion
    AF_bh : numpy.ndarray
        Amplification with respect to borehole motion
    TF_bh : numpy.ndarray
        Transfer function (complex-valued) with respect to borehole motion
    f0_bh : float
        Fundamental frequency of rock outcrop amplification

    NOTE: Vs profile can either include a "zero thickness" layer at last or not.

    (Rewritten into Python sometime during winter of 2016 to spring of 2017.)
    '''

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
    freq_array = np.linspace(0, freq_resolution * (TF_size - 1), num=TF_size)

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
        xSize = 12; ySize = 6
        fig = plt.figure(figsize=(xSize,ySize),edgecolor='k',facecolor='w')

        x_limits = [0,fmax]
        x_limits_log = [1e-1,fmax]
        ax = fig.add_subplot(3,4,1); plt.plot(freq_array,AF_ro,'k'); plt.ylabel('S to R.O.'); plt.xlim(x_limits); plt.grid(color=[0.5]*3,ls=':'); plt.title('Amplitude'); plt.text(0.55,0.85,'$\mathregular{f_0}$ = %.2f Hz'%f0_ro,transform=ax.transAxes,fontweight='bold'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,5); plt.plot(freq_array,AF_in,'k'); plt.ylabel('S to Inci.'); plt.xlim(x_limits); plt.grid(color=[0.5]*3,ls=':'); plt.text(0.55,0.85,'$\mathregular{f_0}$ = %.2f Hz'%f0_in,transform=ax.transAxes,fontweight='bold'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,9); plt.plot(freq_array,AF_bh,'k'); plt.ylabel('S to B.H.'); plt.xlabel('Frequency [Hz]'); plt.xlim(x_limits); plt.grid(color=[0.5]*3,ls=':'); plt.text(0.55,0.85,'$\mathregular{f_0}$= %.2f Hz'%f0_bh,transform=ax.transAxes,fontweight='bold'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,2); plt.plot(freq_array,np.unwrap(np.angle(TF_ro)),'k'); plt.xlim(x_limits); plt.grid(color=[0.5]*3,ls=':'); plt.title('Phase angle (rad)'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,6); plt.plot(freq_array,np.unwrap(np.angle(TF_in)),'k'); plt.xlim(x_limits); plt.grid(color=[0.5]*3,ls=':'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,10); plt.plot(freq_array,np.unwrap(np.angle(TF_bh)),'k'); plt.xlim(x_limits); plt.grid(color=[0.5]*3,ls=':'); plt.xlabel('Frequency [Hz]'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,3); plt.semilogx(freq_array,AF_ro,'k'); plt.xlim(x_limits_log); plt.grid(color=[0.5]*3,ls=':');  plt.title('Amplitude');  ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,7); plt.semilogx(freq_array,AF_in,'k'); plt.xlim(x_limits_log); plt.grid(color=[0.5]*3,ls=':'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,11); plt.semilogx(freq_array,AF_bh,'k'); plt.xlabel('Frequency [Hz]'); plt.xlim(x_limits_log); plt.grid(color=[0.5]*3,ls=':'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,4); plt.semilogx(freq_array,np.unwrap(np.angle(TF_ro)),'k'); plt.xlim(x_limits_log); plt.grid(color=[0.5]*3,ls=':'); plt.title('Phase angle (rad)'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,8); plt.semilogx(freq_array,np.unwrap(np.angle(TF_in)),'k'); plt.xlim(x_limits_log); plt.grid(color=[0.5]*3,ls=':'); ax.set_axisbelow(True)
        ax = fig.add_subplot(3,4,12); plt.semilogx(freq_array,np.unwrap(np.angle(TF_bh)),'k'); plt.xlim(x_limits_log); plt.grid(color=[0.5]*3,ls=':'); plt.xlabel('Frequency [Hz]'); ax.set_axisbelow(True)

        plt.tight_layout(pad=0.3,h_pad=0.3,w_pad=0.3)
        plt.show()

    return freq_array, AF_ro, TF_ro, f0_ro, AF_in, TF_in, AF_bh, TF_bh, f0_bh

#%%----------------------------------------------------------------------------
def calc_damping_from_stress_strain(strain_in_unit_1, stress, Gmax):
    '''
    Calculates the damping curve from the given stress-strain curve.

    Parameters
    ----------
    strain_in_unit_1 : numpy.array
        Strain array in the unit of 1. 1D numpy array
    stress : numpy.array
        Stress. 1D numpy array
    Gmax : float
        Maximum shear modulus, whose unit needs to be identical to that of the
        stress curve.

    Returns
    -------
    damping : numpy.ndarray
        A 1D numpy array of damping ratios, in the unit of "1"
    '''

    strain = strain_in_unit_1
    n = len(strain)

    G_Gmax = calc_GGmax_from_stress_strain(strain, stress, Gmax=Gmax)

    area = np.zeros(n)
    damping = np.zeros(n)

    area[0] = 0.5 * (strain[0] * G_Gmax[0]) * strain[0]
    damping[0] = 2. / np.pi * (2. * area[0] / G_Gmax[0] / strain[0]**2 - 1)
    for i in range(1, n):
        area[i] = area[i-1] + 0.5 * (strain[i-1] * G_Gmax[i-1] + \
                  strain[i] * G_Gmax[i]) * (strain[i] - strain[i-1])
        damping[i] = 2. / np.pi * (2 * area[i] / G_Gmax[i] / strain[i]**2 - 1)

    return damping

#%%----------------------------------------------------------------------------
def calc_GGmax_from_stress_strain(strain_in_unit_1, stress, Gmax=None):
    '''
    Calculates G/Gmax curve from stress-strain curve

    Parameter
    ---------
    stress_strain_curve : numpy.ndarray
        Stress-strain curve. Needs to have two columns. The strain needs to
        have unit "1".
    Gmax : float
        Maximum shear modulus, whose unit needs to be identical to that of the
        stress curve. If not provided, it is automatically calculated from the
        stress-strain curve.

    Returns
    -------
    GGmax : numpy.ndarray
        A 1D numpy array of G/Gmax
    '''
    hlp.assert_1D_numpy_array(strain_in_unit_1)
    hlp.assert_1D_numpy_array(stress)

    if strain_in_unit_1[0] == 0:
        raise ValueError('`strain_in_unit_1` should start with a non-zero value.')

    if Gmax is None:
        Gmax = stress[0] / strain_in_unit_1[0]

    G = stress / strain_in_unit_1  # secant modulus
    GGmax = G / Gmax

    return GGmax
