# Author: Jian Shi

import time
import numpy as np
import site_response as sr
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from .class_Vs_profile import Vs_Profile

class SVM():
    '''
    Class implementation for the Sediment Velocity Model (SVM).

    Original paper:
        Shi & Asimaki (2018) "A generic velocity profile for basin sediments in
        California conditioned on Vs30". Seismological Research Letters, 89(4)

    Parameters
    ----------
    target_Vs30 : float
        The Vs30 value to be queried. Unit: m/s.
    z1000 : float
        The depth to bedrock (1000 m/s rock). Unit: m
    thk : float
        Thickness of layers of generated profile (unit: m); if not specified,
        it will be determined automatically.
    Vs_cap : bool
        Whether to "cap" the Vs profile or not.
        If True, then the Vs profile is capped at 1000.0 m/s; if specified
        as another real number, Vs profile is capped at that value.
        [***  See Footnote for more explanations  ***]
    show_fig : bool
        Whether or not to plot the generated Vs profile
    iterate : bool
        Whether or not to iteratively adjust the input Vs30 so that the actual
        Vs30 (calculated from the resultant Vs profile) falls within 10 m/s
        of the target_Vs30. False by default.
    verbose : bool
        Whether or not to print iteration progress (trial Vs30 value and
        calculated Vs30 value) on the terminal.

    Attributes
    ----------
    smooth_profile : numpy.ndarray
        The smooth Vs profile associated with the given Vs30 and z1000
    Vs30 : float
        The target Vs30 value, in m/s
    z1000 : float
        The basin depth, in meters
    '''

    def __init__(self, target_Vs30, z1000=350.0, thk=None, Vs_cap=True,
                 show_fig=False, iterate=False, verbose=False):
        '''
        Footnote
        --------
        If the resultant Vs profile does not reach Vs_cap at z1000, it will be
        "glued" to Vs_cap, resulting in a velocity impedance at z1000. If the
        Vs profile exceeds Vs_cap at a depth shallower than z1000, then the
        smooth Vs profile is truncated at a depth where Vs = 0.8*Vs_cap, then
        filled down to z1000 with linearly increasing Vs values.
        '''

        if thk is None:  # if thk is empty
            if z1000 >= 50:
                thk = 1.0  # use 1.0 m as layer thickness
            else: # if z1000 less than 50 m (i.e., shallow sedimentary soil layers)
                thk = (z1000 - 2.5)/49.0  # see a few lines below, "Note 2"
        else:
            thk = float(thk)

        if (target_Vs30 < 173.1) or (target_Vs30 > 1000):
            print('******Warning: Vs30 out of range. Result could be '
                  'problematic.******')

        thk_addl_layer = 2.5 - thk  # thickness of "additional" layer to be added on top
        '''
        Note 1: The first layer of Vs_analyt (before adding any new layers on
                top) is Vs0. The final Vs profile should have a homogeneous Vs
                layer for the top 2.5 m, thus we should add a new layer with
                Vs = Vs0 whose thickness is "2.5 minus thk".

        Note 2: For shallow profiles (i.e., z1000 < 50 m), we still want at
                least 50 layers, so we solve these following two equations:

                   thk$ = 2.5 - thk  (note: thk$ is `thk_addl_layer`)
                   thk = (z1000 - thk$)/50   (divide remaining soils into 50 layers)

                Then thk and thk$ can both be solved, hence we have:
                    thk = (z1000 - 2.5)/49.0
        '''

        p1 = -2.1688e-04  # these values come from curve fitting
        p2 = 0.5182
        p3 = 69.452

        #q1 = 8.4562e-09
        #q2 = 2.9981
        #q3 = 0.03073

        r1 = -59.67  # updated on 2018/1/2: improved curve fitting accuracy for k_
        r2 = -0.2722
        r3 = 11.132

        s1 = 4.110
        s2 = -1.0521e-04
        s3 = -10.827
        s4 = -7.6187e-03

        if z1000 <= 2.5:  # this is a rare case, but it does happen sometimes...
            Vs0_ = p1 * target_Vs30**2.0 + p2 * target_Vs30 + p3
            vs_profile = np.array([[z1000,Vs0_],[0.0, 1000.0]])  # just one layer
        else:  # this is most of the cases...
            Vs30 = target_Vs30
            iteration_flag = True

            while iteration_flag is True:
                # --------  Calculate analytical Vs profile from Vs30  ---------
                Vs0_ = p1 * Vs30**2.0 + p2 * Vs30 + p3

                k_ = np.exp(r1 * Vs30**r2 + r3)  # updated on 2018/1/2
                n_ = np.max([1.0, s1*np.exp(s2*Vs30) + s3*np.exp(s4*Vs30)])

                z_array_analyt = np.arange(0.0, z1000-thk_addl_layer, thk) # depth array
                th_array_analyt = sr.convertDepthToThickness(z_array_analyt) # thickness array (for analytical Vs)
                Vs_analyt = Vs0_ * (1. + k_ * z_array_analyt)**(1./n_)  # analytical Vs ( = Vs0*(1+k*z)^(1/n) )

                array1 = np.array([thk_addl_layer,Vs_analyt[0]])  # the homogeneous layer with Vs = Vs0
                array2 = np.column_stack((th_array_analyt,Vs_analyt))  # the other layers (i.e., Vs = Vs0*(1+k*z)^(1/n) )
                temp_Vs_profile = np.row_stack((array1,array2))  # stack the homogeneous layer on top

                if iterate == False:
                    iteration_flag = False  # abort while loop after only one run
                else:
                    # -------  Check if actual Vs30 matches target Vs30 -----------
                    actual_Vs30 = sr.calcVs30(temp_Vs_profile)  # calculate "actual" Vs30
                    if verbose is True:  # print iteration progress
                        print('  %.1f --> %.1f |' % (actual_Vs30,target_Vs30),end='')

                    if target_Vs30-10 <= actual_Vs30 <= target_Vs30+10: # within +/- 10 m/s of targer_Vs30
                        iteration_flag = False  # end iteration
                        if verbose is True:
                            print('|')
                    else:
                        Vs30_temp = Vs30 - (actual_Vs30 - target_Vs30)/5.0  # update the "trial Vs30" to offset the difference
                        if (Vs30_temp < 130) or (Vs30_temp > 1000):  # if the "trial Vs30" is out of range
                            iteration_flag = False  # end iteration
                            if verbose is True:
                                print('')
                        else:
                            Vs30 = Vs30_temp  # use the "trial Vs30" as the new Vs30
                    ## END OF ACTUAL_VS30 WITHIN [TARGET_VS30-10, TARGER_VS30+10] CHECK

            ## END OF WHILE LOOP (ITERATION UNTIL CONVERGENCE)

            array1 = np.array([thk_addl_layer,Vs_analyt[0]])  # the homogeneous layer with Vs = Vs0
            array2 = np.column_stack((th_array_analyt,Vs_analyt))  # the other layers (i.e., Vs = Vs0*(1+k*z)^(1/n) )
            temp_Vs_profile = np.row_stack((array1, array2))  # stack the homogeneous layer on top

            # ---------   Prepare output variables  ---------------
            if Vs_cap is not False:  # if we need to "cap" the Vs profile somehow
                if Vs_cap is True:  # if Vs_cap value not specified (i.e., user inputs "True")
                    Vs_cap = 1000.0  # use 1000.0 m/s as Vs_cap

                if np.where(Vs_analyt > Vs_cap)[0].size > 0:  # if Vs_analyt eventually exceeds Vs_cap
                    index_Vs_cap = np.where(Vs_analyt > Vs_cap)[0][0]  # find the index from which Vs_analyt exceeds Vs_cap
                else:
                    index_Vs_cap = np.nan  # use NaN to denote the alternative situation

                end_index = len(Vs_analyt)  # total number of layers in the smooth profile (i.e., Vs_analyt)

                if not np.isnan(index_Vs_cap):  # if index_Vs_cap is not NaN
                    eta = 0.80
                    idx_eta_Vs_cap = np.where(Vs_analyt > Vs_cap*eta)[0][0]  # where Vs_analyt exceeds eta*Vs_cap
                    for i in range(idx_eta_Vs_cap,end_index):  # change Vs value where Vs > eta*Vs_cap
                        Vs_analyt[i] = Vs_cap * eta + Vs_cap * (1 - eta) / \
                                       (end_index-idx_eta_Vs_cap) * (i-idx_eta_Vs_cap)
                            # linearly distribute Vs increment from eta*Vs_cap to Vs_cap

                array3 = np.append(th_array_analyt[:-1],0.0)  # thickness (including a 0-m "phantom" layer)
                array4 = np.append(Vs_analyt[:-1],Vs_cap)  # Vs ("phantom" layer has Vs = Vs_cap)
                array5 = np.column_stack((array3, array4))  # place thickness and Vs side by side
                vs_profile = np.row_stack((array1, array5))  # stack additional layer on top
            else:   # if Vs profile is not to be capped
                vs_profile = np.copy(temp_Vs_profile)
            ## END OF VS_CAP TRUE/FALSE CHECKING

        ## END OF "IF Z1000 <= 2.5" CHECK

        # ---------   Show figure    ----------------------
        if show_fig is True:
            sr.plotVsProfile(vs_profile,
                             title='$V_{S30}$=%.1fm/s, $z_{1000}$=%.1fm' %
                             (target_Vs30, z1000))

        # --------  Attributes   -------------------
        self.Vs30 = target_Vs30
        self.z1000 = z1000
        self.smooth_profile = vs_profile

    #%%========================================================================
    def __repr__(self):
        return 'Vs30 = %.2g m/s, z1000 = %.2g m' % (self.Vs30, self.z1000)

    #%%========================================================================
    def get_smooth_profile(self, show_fig=False):
        '''
        Return the "base profile" (i.e., smooth).

        Parameters
        ----------
        show_fig : bool
            Whether or not to plot the profile.

        Returns
        -------
        Vs_profile : numpy.array
            The smooth Vs profile corresponding to the given Vs30 value.
            Its first column is the thickness (in meter) of each soil layer,
            and its second column is Vs (in m/s).
        '''
        if show_fig:
            title = '$V_{S30}$=%.1fm/s, $z_{1000}$=%.1fm' % (self.Vs30, self.z1000)
            sr.plotVsProfile(self.smooth_profile, title=title)

        return Vs_Profile(self.smooth_profile)

    #%%========================================================================
    def get_random_profile(self, seed=None, show_fig=False,
                           use_Toros_layering=False, use_Toros_std=False):
        '''
        Returns a randomized a 1D profile.

        Parameters
        ----------
        seed : int
            The seed value for setting the random state. It not set, this
            method automatically uses the current time to generate a seed.
        show_fig : bool
            Whether or not to show the figure of smooth and randomized profiles.
        use_Toros_layering : bool
            Whether or not to use the layering relation in Toro (1995) instead
            of Eq (7) of Shi & Asimaki (2018)
        use_Toros_std : bool
            Whether or not to use the standard deviation (i.e., sigma(ln(Vs)))
            in Toro (1995) instead of Eq (9) of Shi & Asimaki (2018)

        Returns
        -------
        Vs_profile : numpy.ndarray
            The randomzed Vs profile in the same format as "smooth_profile".
        '''

        if seed == None:
            cc = time.localtime(time.time())
            seed = cc[5] * 1e7

        seed = int(seed) # convert seed_value into int (for robustness)
        np.random.seed(int(seed))

        '''--------------  Part 1. Soil Layering Randomization  -------------'''
        z_top = [0]   # depth of layer top
        z_bot = []    # depth of layer bottom
        z_mid = []    # midpoint depth of soil layers
        thk = []    # thickness

        while len(z_bot) == 0 or z_bot[-1] < self.z1000:
            if use_Toros_layering:
                rate = 1.98 * (z_top[-1] + 10.86) ** (-0.89)  # Eq (2) of Toro (1995)

                '''
                The parameter for the Poisson process equals to 1/rate, because
                Toro (1995) says the unit of `rate` is 1/m, and also as written
                in page 40 of Harmon's UIUC PhD thesis (2017), "the expected
                layer thickness at 1000 m is 239 m", which confirms that
                lambda_ = 1 / rate.
                '''
                lamda_ = 1 / rate
                thk_rand = -1
                while thk_rand <= 0:  # to ensure thickness is always positive
                    thk_rand = np.random.poisson(lamda_)  # draw random sample
            else:
                func = lambda thk: SVM._thk_depth_func(thk, z_top[-1])
                if len(thk) == 0:  # the first layer
                    ier = -6  # exit flag
                    while ier != 1:  # keeps trying until fsolve() properly converges
                        mean_thk, info, ier, msg \
                            = fsolve(func, z_top[-1] + 4.0, full_output=True)
                else:  # the rest of the layers
                    ier = -6  # exit flag
                    while ier != 1:  # keeps trying until fzero() properly converges
                        mean_thk, info, ier, msg \
                            = fsolve(func, z_top[-1] + 4.0, full_output=True)

                z_mid_temp = z_top[-1] + mean_thk/2.0
                std_thk = 0.951 * z_mid_temp ** 0.628  # Eq (8) of Shi & Asimaki (2018)
                thk_rand = np.random.normal(mean_thk, std_thk)  # randomized thk based on mean and std

            thk_rand = np.max([thk_rand, 2.0])  # make sure each layer is at least 2 meters thick; too thin layers are not realistic

            thk.append(thk_rand)
            z_mid.append(z_top[-1] + thk_rand/2.0)
            z_bot.append(z_top[-1] + thk_rand)
            z_top.append(z_top[-1] + thk_rand)

        thk[-1] = self.z1000 - np.sum(thk[:-1])  # adjust thickness of last layer so that sum(thk) = z1000
        z_mid = sr.convertThicknessToDepth(thk, midpoint=True)  # update z_mid because thk has changed (z_top and z_bot are not used below, so no need to update)

        ''' ----------------   Part 2   ------------------------------------'''
        ''' Calculate baseline Vs profile based on layering & smooth profile'''
        baseline_Vs = np.zeros(len(thk))
        Vs_analyt = self.smooth_profile[:, 1]
        thk_array_analyt = self.smooth_profile[:, 0]
        z_array_analyt = sr.convertThicknessToDepth(thk_array_analyt)

        for i in range(len(thk)):  # query Vs value where z = z_mid[j]
            index_value, ____ = self._find_index_closest(z_array_analyt,z_mid[i])
            baseline_Vs[i] = Vs_analyt[index_value]

        ''' ---------------    Part 3    -----------------------------------'''
        '''Generate random values for each layer based on the baseline profile'''

        ## ******** 3.1. Toro (1995) coefficients *********
        ## ******** These values come from Table 5 of Toro (1995) or Table 2.3
        ## ******** of Kamai, Abrahamson, Silva (2013) PEER report.
        if self.Vs30 < 180: # site class E
            sigma_lnV = 0.37
            rho_0  = 0
            Delta = 5.0
            rho_200 = 0.50
            z_0 = 0
            b = 0.744
        elif self.Vs30 < 360: # site class D
            sigma_lnV = 0.31
            rho_0 = 0.99
            Delta = 3.9
            rho_200 = 0.98
            z_0 = 0
            b = 0.344
        elif self.Vs30 < 760: # site class C
            sigma_lnV = 0.27
            rho_0 = 0.97
            Delta = 3.8
            rho_200 = 1.00
            z_0 = 0
            b = 0.293
        else: # site classes B and A (these values are intended for class B
              # only, but you can still produce a result for a class A profile.
              # The result just won't make sense.)
            sigma_lnV = 0.36
            rho_0 = 0.95
            Delta = 3.4
            rho_200 = 0.42
            z_0 = 0
            b = 0.063

        ## ***** 3.2. Calculate "mu" and "sigma" of Vs as a function of depth  ****
        ''' (Note: "mu" and "sigma" here are NOT the mean value and standard
            deviation of Vs, but rather the two parameters of the log-normal
            distribution that Vs is assumed to follow.)'''
        if not use_Toros_std:
            std_analyt_Vs = -89.7085 + 1.6434 * z_array_analyt + 0.5204 * self.Vs30  # Eq (9)
            std_analyt_Vs = np.maximum(0.5, std_analyt_Vs)  # because std < 0 when Vs30 is too low
            _, sigma_lognormal_Vs = SVM._lognstat_back_calc(Vs_analyt, std_analyt_Vs)
        else:
            sigma_lognormal_Vs = sigma_lnV * np.ones(Vs_analyt.shape) # page 8 of Toro (1995)

        ## ****** 3.3. Generate random Vs values based on Toro's equations  ******
        Vs_hat = np.zeros([len(thk),1])  # randomly realized Vs values
        Y = np.zeros([len(thk),1])  # this "Y" here is the "Z" in Toro (1995)
        np.random.seed([2 * seed]) # specify seed value to random number generator

        for i in range(0,len(thk)):  # loop through layers
            index_value, __ = SVM._find_index_closest(z_array_analyt,z_mid[i])
            sigma_ = sigma_lognormal_Vs[index_value]  # query sigma value where z = z_mid[j]

            if z_mid[i] > 200:
                rho_z = rho_200
            else:
                rho_z = rho_200 * ((z_mid[i] + z_0) / (200.0 + z_0))**b

            rho_thk = rho_0 * np.exp(-thk[i] / Delta)
            rho_1L = (1 - rho_z) * rho_thk + rho_z

            if i == 0:  # for the first layer
                Y[i] = np.random.normal(0,1,(1,1))  # generate a 1-by-nr_of_rand_profiles vector
            else:  # for other layers
                Y[i] = rho_1L * Y[i-1] + \
                       np.random.normal(0,1,(1,1)) * np.sqrt(1 - rho_1L**2)

            Vs_hat[i] = baseline_Vs[i] * np.exp(Y[i] * sigma_)

        '''-------------  Part 4: Adjust Vs_profile  ----------------'''
        ''' If the last layer of Vs_profile is less than 1000 m/s, add a
            1000 m/s layer at the very bottom.  '''
        Vs_profile = np.column_stack((thk,Vs_hat))
        if Vs_profile[-1,1] < 1000:
            Vs_profile = np.row_stack((Vs_profile,[0,1000]))

        '''-------------  Part 5: Plot Vs profile (optional) ---------------'''
        if show_fig is True:
            hf1,ha1,hl1 = sr.plotVsProfile(Vs_profile)
            ___,___,hl2 = sr.plotVsProfile(self.smooth_profile,hf1,ha1)
            plt.setp(hl1,linewidth=1.25,color='r')
            plt.legend([hl1,hl2],['Stochastic','Smooth'],loc='best',fontsize=11)
            ha1.set_title('$V_{S30}$=%.1fm/s, $z_{1000}$=%.1fm' %
                          (self.Vs30, self.z1000))

        return Vs_Profile(Vs_profile)

    #%%========================================================================
    @staticmethod
    def _thk_depth_func(thk, z_top):
        '''
        Given thk (thickness, in meter) and z_top (depth of layer top, in
        meter), returns "right hand side" minus "left hand side".

        Eq (7) of Shi & Asimaki (2018) Seismological Research Letters:

                thk = 1.125 * z_mid ^(0.620)

        Since z_mid = z_top + h/2.0,

                thk = 1.125 * (z_top + h/2.0)^(0.620)
        '''
        thk = np.array(thk)
        return 1.125 * (z_top + thk/2.0)**0.620 - thk

    #%%========================================================================
    @staticmethod
    def _lognstat_back_calc(mean_value, variance_value):
        '''
        Calculates mu and sigma of a log-normal distribution from the mean value
        and the variance value of the distribution.

        The formulas come from: http://www.mathworks.com/help/stats/lognstat.html

        Parameters
        ----------
        mean_value : float or array_like
            The mean value of the log-normal distribution
        variance_value : float or array_like
            The variance of the log-normal distribution

        Returns
        -------
        mu : float or array_like
            The mean of the associated normal distribution (i.e., ln(X), where
            X is the random variable that follows the log-normal distribution.)
        sigma : float or array_like
            The standard deviation of the associated normal distribution.
        '''

        m = mean_value
        v = variance_value

        mu = np.log(m**2.0 / np.sqrt(v + m**2.0))
        sigma = np.sqrt(np.log(v / m**2.0 + 1))

        return mu, sigma

    #%%========================================================================
    @staticmethod
    def _find_index_closest(array, value):
        '''
        Find the index in `array` which contains the closest value to `value`.
        NaN values within `array` are obmitted implicitly.

        Parameters
        ----------
        array : numpy.ndarray
            Array from which to query the index. Must be 1D numpy array. It
            does NOT need to be sorted.
        value : float
            The value of interest

        Returns
        -------
        index : int
            The index within `array` where the closest value is found
        closest_value : float
            The closest value to `value` within `array`
        '''

        array = np.array(array)
        if len(array) == 0:
            raise ValueError('The length of `array` needs to >= 0.')
        if array.ndim > 1:
            raise ValueError('`array` must be a 1D numpy array.')

        index = np.nanargmin(np.abs(array-value))
        closest_value = array[index]

        return index, closest_value