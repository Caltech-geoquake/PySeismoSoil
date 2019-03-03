# Author: Jian Shi

import os
import numpy as np

from . import helper_generic as hlp
from . import helper_site_response as sr

class Vs_Profile:
    '''
    Class implementation of a Vs profile

    Parameters
    ----------
    data : str or numpy.ndarray
        If str: the full file name on the hard drive containing the data.
        If numpy.ndarray: the numpy array containing the Vs profile data.
        The provided data needs to have either 2 or 5 columns.

        The correct format for a Vs profile should be:

           thickness (m) | Vs (m/s) | damping | density | material number
           --------------+----------+---------+---------+----------------
                ...      |   ...    |   ...   |   ...   |      ...

        (The "material numbers" are integer indices that map each layer to
        their G/Gmax and damping curves.)

    damping_unit : {'1', '%'}
        The unit for the damping ratio
    density_unit : {'kg/m^3', 'g/cm^3', 'kg/m3', 'g/cm3'}
        The unit for the mass density of soils
    sep : str
        Delimiter character for reading the text file. If `data` is
        supplied as a numpy array, this parameter is ignored.
    add_halfspace : bool
        If True, add a "half space" (represented by a layer of 0 m
        thickness) at the bottom, if such a layer does not already
        exist.

    **kwargs_to_genfromtxt :
        Any extra keyword arguments will be passed to numpy.genfromtxt()
        function for loading the data from the hard drive (if applicable).

    Attributes
    ----------
    vs_profile : numpy.ndarray
        The full 5-column Vs profile data. If the supplied Vs profile only has
        2 columns, damping and density and material numbers are automatically
        filled in.
    vs30 : float
        Reciprocal of the weighted average travel time through the top 30 m.
    damping_unit : str
        Same as provided
    density_unit : str
        Same as provided
    z_max : float
        Maximum depth of the profile
    '''

    #--------------------------------------------------------------------------
    def __init__(self, data, damping_unit='1', density_unit='kg/m^3', sep='\t',
                 add_halfspace=False, **kwargs_to_genfromtxt):

        if isinstance(data, str):  # "data" is a file name
            self._path_name, self._file_name = os.path.split(data)
            data_ = np.genfromtxt(data, delimiter=sep, **kwargs_to_genfromtxt)
        elif isinstance(data, np.ndarray):
            data_ = data
            self._path_name, self._file_name = None, None
        else:
            raise TypeError('"data" must be a file name or a numpy array.')

        hlp.check_Vs_profile_format(data_)

        if damping_unit not in ['1', '%']:
            raise ValueError('damping_unit must be ''1'' or ''%''.')
        if density_unit not in ['kg/m^3', 'g/cm^3', 'kg/m3', 'g/cm3']:
            raise ValueError('density_unit must be ''kg/m^3'' or ''g/cm^3''.')

        thk = data_[:, 0]
        vs  = data_[:, 1]
        nr_layers, nr_col = data_.shape

        if nr_col == 2:
            xi, rho = sr.get_xi_rho(vs, formula_type=1)
            if density_unit in ['g/cm^3', 'g/cm3']:
                rho /= 1000.0  # kg/m^3 --> g/cm^3
            if damping_unit == '%':
                xi *= 100.0  # unity --> percent

            if thk[-1] == 0:  # last layer is an "infinity" layer
                material_number = np.append(np.arange(1, nr_layers),[0])
            else:
                material_number = np.arange(1, nr_layers+1)

            full_data = np.column_stack((thk, vs, xi, rho, material_number))
        elif nr_col == 5:
            xi  = data_[:, 2]
            rho = data_[:, 3]
            if density_unit in ['kg/m^3', 'kg/m3'] and min(rho) <= 1000:
                print('WARNING: min(density) is lower than 1000 kg/m^3. Possible error.')
            elif density_unit in ['g/cm^3', 'g/cm3'] and min(rho) <= 1.0:
                print('WARNING: min(density) is lower than 1.0 g/cm^3. Possible error.')

            if damping_unit == '1' and max(xi) > 1:
                print('WARNING: max(damping) larger than 100%. Possible error.')

            material_number = data_[:, 4]
            full_data = data_.copy()
        else:
            raise ValueError('The dimension of the input data is wrong. It '
                             'should have two or five columns.')

        if add_halfspace and thk[-1] != 0:
            last_row = full_data[-1, :]
            half_space = last_row.copy()
            half_space[0] = 0  # set thickness to 0 meters
            half_space[4] = 0  # set material number to 0
            full_data = np.row_stack((full_data, half_space))
            thk, vs, xi, rho, material_number = full_data.T

        self._thk = thk
        self._vs = vs
        self._xi = xi
        self._rho = rho
        self._material_number = material_number
        self.vs_profile = full_data
        self.vs30 = sr.calc_Vs30(full_data)
        self.damping_unit = damping_unit
        self.density_unit = density_unit
        self.z_max = np.sum(thk)

    #--------------------------------------------------------------------------
    def __repr__(self):
        '''
        Defines a presentation of the basic info of a Vs profile.
        '''

        text = '\n----------+----------+-------------+------------------+--------------\n'
        text += '  Thk (m) | Vs (m/s) | damping (%%) | density (%s) | material No. \n' \
                 % self.density_unit
        text += '----------+----------+-------------+------------------+--------------\n'

        nr_layers, nr_col = self.vs_profile.shape
        for j in range(nr_layers):
            text += '{:^10}|'.format('%.2f' % self.vs_profile[j,0])
            text += '{:^10}|'.format('%.1f' % self.vs_profile[j,1])
            if self.damping_unit == '1':
                text += '{:^13}|'.format('%.3f' % (self.vs_profile[j,2]*100.0))
            else:
                text += '{:^13}|'.format('%.3f' % self.vs_profile[j,2])
            if len(self.density_unit) == 6:
                text += '{:^18}|'.format('%.1f' % self.vs_profile[j,3])
            else:
                text += '{:^17}|'.format('%.1f' % self.vs_profile[j,3])
            text += '{:^14}'.format('%d' % self.vs_profile[j,4])
            text += '\n'

        text += '----------+----------+-------------+------------------+--------------\n'
        text += '\n(Vs30 = %.1f m/s)\n' % self.vs30

        return text

    #--------------------------------------------------------------------------
    def plot(self, fig=None, ax=None, **kwargs):
        '''
        Plot Vs profile

        Parameters
        ----------
        fig, ax : (optional)
            matplotlib figure and axes objects to show the plot on
        **kwargs :
            Extra keyword arguments to be passed to the function
            helper_site_response.plot_Vs_profile()

        Returns
        -------
        fix, ax, hl
            Figure, axes, and line objects, respectively
        '''
        if self._file_name:
            title_text = self._file_name
        elif 'title' in kwargs:
            title_text = kwargs['title']
            kwargs.pop('title')
        else:
            title_text = None

        fig, ax, hl = sr.plot_Vs_profile(self.vs_profile, title=title_text,
                                         fig=fig, ax=ax, **kwargs)
        return fig, ax, hl

    #--------------------------------------------------------------------------
    def ampl_function(self, show_fig=False, boundary_condition=None,
                      also_return_f0=True, freq_resolution=.05, fmax=30.):
        '''
        Get amplification function of the Vs profile

        Parameters
        ----------
        show_fig : bool
            Whether or not show figures of the amplification function
        boundary_condition : {'ro', 'in', 'bh'}
            Type of boundary condition: 'ro' for rock outcrop, 'in' for
            incident motion, and 'bh' for borehole motion. If None, results of
            all three boundary conditions are returned.
        also_return_f0 : bool
            Whether or not to also return f_0 (fundamental frequency)
        freq_resolution : float
            Frequency resolution of the frequency spectrum
        fmax : float
            Maximum frequency of interest

        Returns (depending on the input, not all will be returned)
        ----------------------------------------------------------
        freq_array : numpy.ndarray
            Frequency array, in linear scale
        AF_ro : numpy.ndarray
            Amplification with respect to rock outcrop
        f0_ro : float
            Fundamental frequency of rock outcrop amplification
        AF_in : numpy.ndarray
            Amplification with respect to incident motion
        AF_bh : numpy.ndarray
            Amplification with respect to borehole motion
        f0_bh : float
            Fundamental frequency of rock outcrop amplification
        '''

        freq, af_ro, _, f0_ro, af_in, _, af_bh, _, f0_bh = \
                sr.linear_tf(self.vs_profile, show_fig=show_fig, fmax=fmax,
                             freq_resolution=freq_resolution)
        if not boundary_condition:
            if also_return_f0:
                return freq, af_ro, f0_ro, af_in, af_bh, f0_bh
            else:
                return freq, af_ro, af_in, af_bh
        elif boundary_condition == 'ro':
            if also_return_f0:
                return freq, af_ro, f0_ro
            else:
                return freq, af_ro
        elif boundary_condition == 'in':
            if also_return_f0:
                return freq, af_in, f0_ro
            else:
                return freq, af_in
        elif boundary_condition == 'bh':
            if also_return_f0:
                return freq, af_bh, f0_bh
            else:
                return freq, af_bh
        else:
            raise ValueError("`boundary_condition` must be one of "
                             "['ro', 'in', 'bh'].")

    #--------------------------------------------------------------------------
    def transfer_function(self, show_fig=False, boundary_condition=None,
                          also_return_f0=False, freq_resolution=.05, fmax=30.):
        '''
        Get transfer function (complex-valued) of the Vs profile

        Parameters
        ----------
        show_fig : bool
            Whether or not show figures of the transfer function
        boundary_condition : {'ro', 'in', 'bh'}
            Type of boundary condition: 'ro' for rock outcrop, 'in' for
            incident motion, and 'bh' for borehole motion. If None, results of
            all three boundary conditions are returned.
        also_return_f0 : bool
            Whether or not to also return f_0 (fundamental frequency)
        freq_resolution : float
            Frequency resolution of the frequency spectrum
        fmax : float
            Maximum frequency of interest

        Returns (depending on the input, not all will be returned)
        ----------------------------------------------------------
        freq_array : numpy.ndarray
            Frequency array, in linear scale
        TF_ro : numpy.ndarray
            Transfer function (complex-valued) with respect to rock outcrop
        f0_ro : float
            Fundamental frequency of rock outcrop amplification
        TF_in : numpy.ndarray
            Transfer function (complex-valued) with respect to incident motion
        TF_bh : numpy.ndarray
            Transfer function (complex-valued) with respect to borehole motion
        f0_bh : float
            Fundamental frequency of rock outcrop amplification
        '''

        freq, _, tf_ro, f0_ro, _, tf_in, _, tf_bh, f0_bh = \
                sr.linear_tf(self.vs_profile, show_fig=show_fig, fmax=fmax,
                             freq_resolution=freq_resolution)
        if not boundary_condition:
            if also_return_f0:
                return freq, tf_ro, f0_ro, tf_in, tf_bh, f0_bh
            else:
                return freq, tf_ro, tf_in, tf_bh
        elif boundary_condition == 'ro':
            if also_return_f0:
                return freq, tf_ro, f0_ro
            else:
                return freq, tf_ro
        elif boundary_condition == 'in':
            if also_return_f0:
                return freq, tf_in, f0_ro
            else:
                return freq, tf_in
        elif boundary_condition == 'bh':
            if also_return_f0:
                return freq, tf_bh, f0_bh
            else:
                return freq, tf_bh
        else:
            raise ValueError("`boundary_condition` must be one of "
                             "['ro', 'in', 'bh'].")

    #--------------------------------------------------------------------------
    def phase_transfer_function(self, show_fig=False):
        '''
        Calculate the phase shift transfer function

        Parameter
        ---------
        show_fig : bool
            Whether or not show figures of the amplification function

        Returns
        -------
        freq : numpy.ndarray
            Frequency array, in linear scale
        phase_ro : numpy.array
            Phase shift with respect to rock outcrop
        phase_in : numpy.array
            Phase shift with respect to incident motion
        phase_bh : numpy.array
            Phase shift with respect to borehole motion
        '''

        freq, tf_ro, _, tf_in, tf_bh, _ = self.transfer_function(show_fig)

        phase_ro = np.unwrap(np.angle(tf_ro))
        phase_in = np.unwrap(np.angle(tf_in))
        phase_bh = np.unwrap(np.angle(tf_bh))

        return freq, phase_ro, phase_in, phase_bh

    #--------------------------------------------------------------------------
    def get_f0_RO(self):
        return self.ampl_function(boundary_condition='ro', also_return_f0=True)[-1]

    #--------------------------------------------------------------------------
    def get_f0_BH(self):
        return self.ampl_function(boundary_condition='bh', also_return_f0=True)[-1]

    #--------------------------------------------------------------------------
    def get_depth_array(self):
        return sr.thk2dep(self._thk)

    #--------------------------------------------------------------------------
    def query_Vs_at_depth(self, depth, as_profile=False):
        '''
        Query Vs values at given `depth` values. If the given depth values
        happen to be at layer interfaces, return the Vs of the layer *below*
        the interface.

        Parameter
        ---------
        depth : float or numpy.array
            Value(s) of depths to query the Vs value at. Unit should be m.
        as_profile : bool
            If True, return a Vs profile object. If False, only return the
            array of Vs.

        Returns
        -------
        vs_array : float, numpy.ndarray or Vs_Profile
            Vs values corresponding to the given depths. Its type depends on
            the type of `depth`.
        '''
        vs_queried, is_scalar, has_duplicate_values, is_sorted \
            = sr.query_Vs_at_depth(self.vs_profile, depth)

        if as_profile:
            if not is_sorted:
                raise ValueError('If `as_profile` is set to True, the given '
                                 '`depth` needs to be monotonically increasing.')
            if has_duplicate_values:
                raise ValueError('If `as_profile` is set to True, the given '
                                 '`depth` should not contain duplicate values.')

        if as_profile:
            if not np.any(depth == 0):
                thk_array = sr.dep2thk(np.append([0], depth))
                vs_queried = np.append(vs_queried[0:1], vs_queried)
            else:  # `depth` has been guarenteed to be sorted with no duplicates
                thk_array = sr.dep2thk(depth)
            vs_ = np.column_stack((thk_array, vs_queried))

            # A halfspace is already implicitly added by sr.depth2thk()
            return Vs_Profile(vs_, add_halfspace=False)
        else:
            if is_scalar:
                return float(vs_queried)
            else:
                return vs_queried

    #--------------------------------------------------------------------------
    def query_Vs_given_thk(self, thk, n_layers=None, as_profile=False,
                           at_midpoint=True, add_halfspace=True):
        '''
        Query Vs values from a thickness layer `thk`. The starting point of
        querying is the ground surface.

        Parameters
        ----------
        thk : float or numpy.ndarray
            Thickness array, or a single value that means a constant thickness.
        n_layers : int or None
            Number of layers to query. This parameter has no effect if `thk`
            is a numpy array (because the number of layers can be inferred
            from `thk`). If None, it is automatically inferred from `thk`
            and the maximum depth of the profile (`z_max`).
        as_profile : bool
            If True, return a Vs profile object. If False, only return the
            array of Vs.
        at_midpoint : bool
            If True, the Vs values are queried at the mid-point depths of
            each layer. If False, at the top of each layer.
        add_halfspace : bool
            If True, add a "half space" (represented by a layer of 0 m
            thickness) at the bottom, if such a layer does not already
            exist.

        Return
        ------
        vs_array : numpy.ndarray or Vs_Profile
            Vs values corresponding to the given depths. Its type depends on
            `as_profile`.
        '''
        if n_layers is None and isinstance(thk, (int, float, np.number)):
            n_layers = int(np.ceil(self.z_max / thk))
        vs_queried, thk_array = sr.query_Vs_given_thk(self.vs_profile, thk,
                                                      n_layers=n_layers,
                                                      at_midpoint=at_midpoint)

        if not as_profile:
            return vs_queried
        else:
            vs_ = np.column_stack((thk_array, vs_queried))
            return Vs_Profile(vs_, add_halfspace=add_halfspace)

    #--------------------------------------------------------------------------
    def get_basin_depth(self, bedrock_Vs=1000):
        '''
        Query the depth of the basin as indicated in the Vs profile data.
        The basin is defined as the material whose Vs is at least `bedrock_Vs`.

        Parameter
        ---------
        bedrock_Vs : float
            The shear-wave velocity that you consider as the bedrock

        Returns
        -------
        basin_depth : float
            The basin depth. If no Vs values in the profile reaches
            `bedrock_Vs`, return total depth (bottom) of the profile.
        '''
        depth = sr.thk2dep(self._thk, midpoint=False)
        assert(depth[0] == 0)  # assert that `depth` means the layer top
        basin_depth = -1
        for j in range(len(self._vs)):
            current_depth = depth[j]
            if self._vs[j] >= bedrock_Vs:
                basin_depth = current_depth
                break
        else:
            basin_depth = np.sum(self._thk)

        return basin_depth

    #--------------------------------------------------------------------------
    def get_z1(self):
        '''
        Get z1 (the depth to Vs = 1000 m/s)
        '''
        return self.get_basin_depth(bedrock_Vs=1000)

    #--------------------------------------------------------------------------
    def get_slowness(self):
        '''
        Get "slowness" (reciprocal of wave velocity) as a 2D numpy array
        (including the thickness array)
        '''
        slowness = np.ones_like(self.vs_profile)
        slowness[:,0] = self.vs_profile[:,0]
        slowness[:,1] = 1.0 / self.vs_profile[:,1]
        return slowness

    #--------------------------------------------------------------------------
    def output_as_km_unit(self):
        '''
        Output the Vs profile in km and km/s unit.
        '''
        tmp = self.vs_profile.copy()
        tmp[:,0] /= 1000.0
        tmp[:,1] /= 1000.0
        return tmp

    #--------------------------------------------------------------------------
    def summary(self):
        print(self)
        self.plot()

    #--------------------------------------------------------------------------
    def save_vs_profile(self, fname, sep='\t',
                        precision=['%.2f', '%.2f', '%.4g', '%.5g', '%d']):

        if not isinstance(precision, list):
            raise TypeError('precision must be a list.')
        if len(precision) != 5:
            raise ValueError('Length of precision must be 5.')

        np.savetxt(fname, self.vs_profile, fmt=precision, delimiter=sep)
