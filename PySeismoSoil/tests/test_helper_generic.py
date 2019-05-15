# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_generic as hlp

class Test_Helper_Generic(unittest.TestCase):
    '''
    Unit test for helper functions in helper_generic.py
    '''

    def test_is_int(self):
        self.assertTrue(hlp.is_int(3))
        self.assertTrue(hlp.is_int(np.array([3])[0]))
        self.assertTrue(hlp.is_int(3.0))
        self.assertTrue(hlp.is_int(np.array([3.0])[0]))
        self.assertFalse(hlp.is_int(3.1))
        self.assertFalse(hlp.is_int(np.array([3.1])[0]))
        self.assertFalse(hlp.is_int(None))
        self.assertFalse(hlp.is_int(np.array([])))
        self.assertFalse(hlp.is_int(np.array([3])))

    def test_read_two_column_stuff(self):
        # Load from file
        data, dt = hlp.read_two_column_stuff('./files/two_column_data_example.txt')
        benchmark = np.array([[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0,
                               1.1, 1.2, 1.3, 1.4, 1.5],
                              [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7]]).T
        self.assertTrue(np.allclose(data, benchmark))
        self.assertAlmostEqual(dt, benchmark[1, 0] - benchmark[0, 0])

        # Load from 1D numpy array
        data_1col = np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7])
        dt = 0.1
        data_, dt_ = hlp.read_two_column_stuff(data, dt)
        self.assertTrue(np.allclose(data_1col, data_[:, 1]))
        self.assertAlmostEqual(dt, dt_)

        # Load from 2D numpy array
        data__, dt__ \
            = hlp.read_two_column_stuff(benchmark,
                                         benchmark[1, 0] - benchmark[0, 0])
        self.assertTrue(np.allclose(data__, benchmark))
        self.assertAlmostEqual(dt__, benchmark[1, 0] - benchmark[0, 0])

    def test_check_two_column_format(self):
        with self.assertRaisesRegex(TypeError, '1.5 should be an numpy array.'):
            hlp.check_two_column_format(1.5, '1.5')
        with self.assertRaisesRegex(TypeError, '_a_ should be a 2D numpy array.'):
            hlp.check_two_column_format(np.array([1,2,3]), '_a_')
        with self.assertRaisesRegex(TypeError, '_b_ should have two columns.'):
            hlp.check_two_column_format(np.ones((2, 3)), '_b_')

    def test_mean_absolute_error(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.32, 2.12, 2.87, 3.95, 5.74])
        self.assertAlmostEqual(hlp.mean_absolute_error(y_true, y_pred), 0.272)

    def test_check_numbers_valid(self):
        self.assertEqual(hlp.check_numbers_valid(np.array([1, 2, 3])), 0)
        self.assertEqual(hlp.check_numbers_valid(np.array(['a', 'b'])), -1)
        self.assertEqual(hlp.check_numbers_valid(np.array([1, 2, np.nan])), -2)
        self.assertEqual(hlp.check_numbers_valid(np.array([1, 2, -1])), -3)

    def test_assert_1D_numpy_array(self):
        with self.assertRaisesRegex(TypeError, 'must be a 1D numpy array.'):
            hlp.assert_1D_numpy_array([1, 2, 3, 4])
        with self.assertRaisesRegex(TypeError, 'must be a 1D numpy array.'):
            hlp.assert_1D_numpy_array(np.array([[1, 2, 3, 4]]))
        with self.assertRaisesRegex(TypeError, 'must be a 1D numpy array.'):
            hlp.assert_1D_numpy_array(np.array([[1, 2], [3, 4]]))

    def test_extract_from_curve_format(self):
        data = np.genfromtxt('./files/curve_FKSH14.txt')[:, :8]
        GGmax, damping = hlp.extract_from_curve_format(data)

        strain = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
        GGmax_1 = [0.99038, 0.97403, 0.92539, 0.8188, 0.59912, 0.35256,
                   0.15261, 0.061578, 0.021241, 0.0078452]
        GGmax_2 = [0.99452, 0.98511, 0.95629, 0.88852, 0.72498, 0.48992,
                   0.24108, 0.10373, 0.036868, 0.013755]
        damping_1 = [1.6683, 1.8386, 2.4095, 3.8574, 7.4976, 12.686, 18.102,
                     21.005, 21.783, 21.052]
        damping_2 = [0.99457, 1.0872, 1.4039, 2.2497, 4.6738, 9.0012, 14.898,
                     19.02, 21.021, 20.947]
        GGmax_bench = [np.column_stack((strain, GGmax_1)),
                       np.column_stack((strain, GGmax_2))]
        damping_bench = [np.column_stack((strain, damping_1)),
                         np.column_stack((strain, damping_2))]
        self.assertTrue(np.allclose(GGmax, GGmax_bench))
        self.assertTrue(np.allclose(damping, damping_bench))

    def test_extract_from_param_format(self):
        data = np.genfromtxt('./files/HH_X_FKSH14.txt')
        param = hlp.extract_from_param_format(data)
        param_bench = [np.array([0.010161, 1, 0.10468, 39.317, 0.630114,
                                 18.7975, 149.535, 29.053, 1]),
                       np.array([0.027916, 1.01507, 0.0851825, 23.468, 0.638322,
                                 5.84163, 183.507, 29.7071, 1]),
                       np.array([0.0479335, 1.00849, 0.276801, 35.9504,
                                 0.643012, 5.04279, 193.483, 54.8234, 1]),
                       np.array([0.0516179, 1.0215, 0.153973, 21.8676,
                                 0.654707, 1.44752, 179.24, 22.4495, 1]),
                       np.array([0.0340815, 1.02711, 0.202054, 25.2326,
                                 0.667001, 3.97622, 195.136, 34.601, 1])]
        self.assertTrue(np.allclose(param, param_bench))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_Generic)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
