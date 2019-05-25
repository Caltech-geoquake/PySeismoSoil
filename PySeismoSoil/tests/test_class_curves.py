# Author: Jian Shi

import unittest
import numpy as np

from PySeismoSoil.class_curves import Curve, GGmax_Curve, Damping_Curve, \
                                      Stress_Curve, Multiple_Damping_Curves, \
                                      Multiple_GGmax_Curves, \
                                      Multiple_GGmax_Damping_Curves

class Test_Class_Curves(unittest.TestCase):
    '''
    Unit test for curve-related class
    '''

    def test_init(self):
        data = np.genfromtxt('./files/curve_FKSH14.txt')
        curve = Curve(data[:, 2:4])
        damping_data = curve.raw_data[:, 1]
        damping_bench = [1.6683, 1.8386, 2.4095, 3.8574, 7.4976,
                         12.686, 18.102, 21.005, 21.783, 21.052]
        self.assertTrue(np.allclose(damping_data, damping_bench))

    def test_plot(self):
        filename = './files/curve_FKSH14.txt'
        data = np.genfromtxt(filename)
        curve = Curve(data[:, 2:4])
        curve.plot(marker='.')

        curves = Multiple_Damping_Curves(filename)
        curves.plot()

    def test_HH_x_fit_single_layer(self):
        data = np.genfromtxt('./files/curve_FKSH14.txt')
        curve = Damping_Curve(data[:, 2:4])

        try:
            hhx = curve.get_HH_x_param(pop_size=1, n_gen=1, show_fig=True,
                                       use_scipy=False)
            self.assertEqual(len(hhx), 9)
            self.assertEqual(hhx.keys(), {'gamma_t', 'a', 'gamma_ref', 'beta',
                                          's', 'Gmax', 'mu', 'Tmax', 'd'})
        except ImportError:  # DEAP library may not be installed
            pass

        hhx = curve.get_HH_x_param(pop_size=1, n_gen=1, show_fig=True,
                                   use_scipy=True)
        self.assertEqual(len(hhx), 9)
        self.assertEqual(hhx.keys(), {'gamma_t', 'a', 'gamma_ref', 'beta',
                                      's', 'Gmax', 'mu', 'Tmax', 'd'})

    def test_H4_x_fit_single_layer(self):
        data = np.genfromtxt('./files/curve_FKSH14.txt')
        curve = Damping_Curve(data[:, 2:4])

        try:
            h4x = curve.get_H4_x_param(pop_size=1, n_gen=1, show_fig=True,
                                       use_scipy=False)
            self.assertEqual(len(h4x), 4)
            self.assertEqual(h4x.keys(), {'gamma_ref', 's', 'beta', 'Gmax'})
        except ImportError:  # DEAP library may not be installed
            pass

        h4x = curve.get_H4_x_param(pop_size=1, n_gen=1, show_fig=True,
                                   use_scipy=True)
        self.assertEqual(len(h4x), 4)
        self.assertEqual(h4x.keys(), {'gamma_ref', 's', 'beta', 'Gmax'})

    def test_value_check(self):
        data = np.genfromtxt('./files/curve_FKSH14.txt')[:, 2:4]
        with self.assertRaisesRegex(ValueError, 'G/Gmax values must be between'):
            GGmax_Curve(data)
        with self.assertRaisesRegex(ValueError, 'damping values must be between'):
            Damping_Curve(data * 100.0)
        with self.assertRaisesRegex(ValueError, 'should have all non-negative'):
            Stress_Curve(data * -1)

    def test_multiple_damping_curves(self):
        mdc = Multiple_Damping_Curves('./files/curve_FKSH14.txt')

        # Test __len__
        self.assertEqual(len(mdc), 5)

        # Test __getitem__
        strain_bench = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
        damping_bench = [1.6683, 1.8386, 2.4095, 3.8574, 7.4976,
                         12.686, 18.102, 21.005, 21.783, 21.052]
        layer_0_bench = np.column_stack((strain_bench, damping_bench))
        self.assertTrue(np.allclose(mdc[0].raw_data, layer_0_bench))

        # Test __setitem__
        mdc[4] = Damping_Curve(layer_0_bench)
        self.assertTrue(np.allclose(mdc[0].raw_data, mdc[4].raw_data))

        # Test __delitem__
        mdc_2 = mdc[2]
        mdc_3 = mdc[3]
        del mdc[2]
        self.assertEqual(len(mdc), 4)
        self.assertEqual(mdc.n_layer, 4)

        # Test __contains__
        self.assertFalse(mdc_2 in mdc)
        self.assertTrue(mdc_3 in mdc)

        # Test slicing
        mdc_slice = mdc[:2]
        self.assertEqual(len(mdc_slice), 2)
        self.assertTrue(isinstance(mdc_slice, Multiple_Damping_Curves))
        self.assertTrue(isinstance(mdc_slice[0], Damping_Curve))

    def test_HH_x_fit_multi_layer(self):
        mdc = Multiple_Damping_Curves('./files/curve_FKSH14.txt')
        mdc_ = mdc[:2]

        # Test differential evolution
        hhx = mdc_.get_all_HH_x_params(pop_size=1, n_gen=1, save_txt=False,
                                       use_scipy=True)
        self.assertEqual(len(hhx), 2)
        self.assertTrue(isinstance(hhx[0].data, dict))
        self.assertEqual(hhx[0].keys(), {'gamma_t', 'a', 'gamma_ref', 'beta',
                                         's', 'Gmax', 'mu', 'Tmax', 'd'})

        # Test DEAP
        try:
            hhx = mdc_.get_all_HH_x_params(pop_size=1, n_gen=1, save_txt=False,
                                           use_scipy=False)
            self.assertEqual(len(hhx), 2)
            self.assertTrue(isinstance(hhx[0].data, dict))
            self.assertEqual(hhx[0].keys(), {'gamma_t', 'a', 'gamma_ref', 'beta',
                                             's', 'Gmax', 'mu', 'Tmax', 'd'})
        except ImportError:  # DEAP library may not be installed
            pass

    def test_H4_x_fit_multi_layer(self):
        mdc = Multiple_Damping_Curves('./files/curve_FKSH14.txt')
        mdc_ = mdc[:2]

        # Test differential evolution
        h4x = mdc_.get_all_H4_x_params(pop_size=1, n_gen=1, save_txt=False,
                                       use_scipy=True)
        self.assertEqual(len(h4x), 2)
        self.assertTrue(isinstance(h4x[0].data, dict))
        self.assertEqual(h4x[0].keys(), {'gamma_ref', 's', 'beta', 'Gmax'})

        # Test DEAP
        try:
            h4x = mdc_.get_all_H4_x_params(pop_size=1, n_gen=1, save_txt=False,
                                           use_scipy=False)
            self.assertEqual(len(h4x), 2)
            self.assertTrue(isinstance(h4x[0].data, dict))
            self.assertEqual(h4x[0].keys(), {'gamma_ref', 's', 'beta', 'Gmax'})
        except ImportError:  # DEAP library may not be installed
            pass

    def test_multiple_GGmax_curve_get_curve_matrix(self):
        damping = 1.23  # choose a dummy value

        mgc = Multiple_GGmax_Curves('./files/curve_FKSH14.txt')
        curve = mgc.get_curve_matrix(damping_filler_value=damping)

        curve_benchmark = np.genfromtxt('./files/curve_FKSH14.txt')
        for j in range(curve_benchmark.shape[1]):
            if j % 4 == 3:  # original damping info is lost; use same dummy value
                curve_benchmark[:, j] = damping

        self.assertTrue(np.allclose(curve, curve_benchmark, rtol=1e-5))

    def test_multiple_damping_curve_get_curve_matrix(self):
        GGmax = 0.76  # choose a dummy value

        mdc = Multiple_Damping_Curves('./files/curve_FKSH14.txt')
        curve = mdc.get_curve_matrix(GGmax_filler_value=GGmax)

        curve_benchmark = np.genfromtxt('./files/curve_FKSH14.txt')
        for j in range(curve_benchmark.shape[1]):
            if j % 4 == 1:  # original damping info is lost; use same dummy value
                curve_benchmark[:, j] = GGmax

        self.assertTrue(np.allclose(curve, curve_benchmark, rtol=1e-5))

    def test_init_multiple_GGmax_damping_curves(self):
        # Case 1: with MGC and MDC
        mgc = Multiple_GGmax_Curves('./files/curve_FKSH14.txt')
        mdc = Multiple_Damping_Curves('./files/curve_FKSH14.txt')
        with self.assertRaisesRegex(ValueError, 'Both parameters are `None`'):
            Multiple_GGmax_Damping_Curves()
        with self.assertRaisesRegex(ValueError, 'one and only one input parameter'):
            Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mgc, mdc), data_array=2.6)
        with self.assertRaisesRegex(TypeError, 'needs to be of type'):
            Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mdc, mgc))
        mgc_ = Multiple_GGmax_Curves('./files/curve_FKSH14.txt')
        del mgc_[-1]
        with self.assertRaisesRegex(ValueError, 'same number of soil layers'):
            Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mgc_, mdc))

        mgdc = Multiple_GGmax_Damping_Curves(mgc_and_mdc=(mgc, mdc))
        matrix = mgdc.get_curve_matrix()
        benchmark = np.genfromtxt('./files/curve_FKSH14.txt')
        self.assertTrue(np.allclose(matrix, benchmark))

        # Case 2: with a numpy array
        array = np.genfromtxt('./files/curve_FKSH14.txt')
        array_ = np.column_stack((array, array[:, -1]))
        with self.assertRaisesRegex(ValueError, 'needs to be a multiple of 4'):
            Multiple_GGmax_Damping_Curves(data_array=array_)

        mgdc = Multiple_GGmax_Damping_Curves(data_array=array)
        mgc_, mdc_ = mgdc.get_MGC_MDC_objects()
        self.assertTrue(np.allclose(mgc_.get_curve_matrix(), mgc.get_curve_matrix()))
        self.assertTrue(np.allclose(mdc_.get_curve_matrix(), mdc.get_curve_matrix()))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Curves)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
