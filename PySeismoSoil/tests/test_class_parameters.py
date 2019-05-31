# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_hh_model as hh
import PySeismoSoil.helper_mkz_model as mkz
from PySeismoSoil.class_parameters import HH_Param, MKZ_Param, \
    HH_Param_Multi_Layer, MKZ_Param_Multi_Layer

class Test_Class_HH_Param(unittest.TestCase):

    def test_init(self):
        # Test that invalid parameter names are blocked as expected
        invalid_data = {'key': 1, 'lock': 2}
        with self.assertRaisesRegex(KeyError, 'Invalid keys exist in your input'):
            hhp = HH_Param(invalid_data)

        # Test that querying a nonexistent parameter name produces KeyError
        data = {'gamma_t': 1, 'a': 2, 'gamma_ref': 3, 'beta': 4, 's': 5,
                'Gmax': 6, 'mu': 7, 'Tmax': 8, 'd': 9}
        hhp = HH_Param(data)
        with self.assertRaisesRegex(KeyError, "'haha'"):
            hhp['haha']

    def test_get_GGmax(self):
        # Actual HH_G parameter from profile 350_750_01
        params = np.array([0.000116861, 100, 0.000314814, 1, 0.919, 9.04E+07,
                           0.0718012, 59528, 0.731508])
        HH_G = HH_Param(hh.deserialize_array_to_params(params))
        GGmax = HH_G.get_GGmax(strain_in_pct=np.logspace(-4, 1, num=50))
        GGmax_bench = [0.994964, 0.993758, 0.992265, 0.990419, 0.988137,
                       0.985319, 0.981846, 0.977568, 0.972312, 0.965866,
                       0.957985, 0.948382, 0.936729, 0.922659, 0.905775,
                       0.885662, 0.861911, 0.83415, 0.802089, 0.765572,
                       0.724631, 0.669318, 0.612187, 0.557676, 0.50573,
                       0.456335, 0.409512, 0.365315, 0.323816, 0.285101,
                       0.249252, 0.216336, 0.186389, 0.15941, 0.13535,
                       0.114114, 0.0955634, 0.0795192, 0.0657761, 0.0541104,
                       0.0442915, 0.0360909, 0.02929, 0.0236855, 0.0190929,
                       0.0153483, 0.0123084, 0.00985001, 0.00786843, 0.00627576]
        self.assertTrue(np.allclose(GGmax, GGmax_bench, atol=1e-4))

        # Actual H4_G parameter of IBRH17 (the 0th layer)
        params = np.array([0.00028511, 0, 0.919, 1.7522])
        H4_G = MKZ_Param(mkz.deserialize_array_to_params(params, from_files=True))
        GGmax = H4_G.get_GGmax(strain_in_pct=np.geomspace(0.0001, 6, num=50))
        GGmax_bench = [0.99038, 0.9882, 0.98553, 0.98228, 0.9783, 0.97346,
                       0.96758, 0.96044, 0.95182, 0.94142, 0.92895, 0.91406,
                       0.89641, 0.87562, 0.85135, 0.82331, 0.79127, 0.75514,
                       0.71502, 0.67118, 0.62415, 0.57465, 0.52361, 0.47207,
                       0.42112, 0.37179, 0.325, 0.28146, 0.24167, 0.20588,
                       0.17418, 0.14646, 0.1225, 0.10199, 0.084583, 0.069916,
                       0.057631, 0.047395, 0.038902, 0.03188, 0.026091, 0.02133,
                       0.017423, 0.01422, 0.0116, 0.0094575, 0.0077078, 0.0062797,
                       0.0051149, 0.0041652]
        self.assertTrue(np.allclose(GGmax, GGmax_bench, atol=1e-4))

    def test_get_damping(self):
        # Actual HH_x parameter from profile 350_750_01
        params = np.array([0.014766, 1.00583, 0.0410009, 21.951, 0.620032,
                           6.44725, 151.838, 13.0971, 1])
        HH_x = HH_Param(hh.deserialize_array_to_params(params))
        damping = HH_x.get_damping(strain_in_pct=np.logspace(-4, 1, num=50))
        damping_bench = [1.6815, 1.70007, 1.72351, 1.75306, 1.79029, 1.83716,
                         1.89609, 1.97005, 2.06272, 2.17854, 2.32286, 2.50202,
                         2.72343, 2.99551, 3.32759, 3.72957, 4.21139, 4.78223,
                         5.44944, 6.21726, 7.0856, 8.04891, 9.09572, 10.2088,
                         11.3664, 12.5435, 13.7147, 14.8555, 15.944, 16.9621,
                         17.8956, 18.7343, 19.4716, 20.104, 20.6311, 21.0548,
                         21.3791, 21.6099, 21.7542, 21.8197, 21.8147, 21.7478,
                         21.6271, 21.4606, 21.2554, 21.0184, 20.7555, 20.472,
                         20.1723, 19.8606]
        # The error could be high, due to curve-fitting errors of genetic algorithms
        self.assertTrue(np.allclose(damping, damping_bench, atol=7.))

        # Actual H4_x parameter from IBRH17 (the 0th layer)
        params = np.array([0.00062111, 0, 0.60001, 1.797])
        H4_x = MKZ_Param(mkz.deserialize_array_to_params(params, from_files=True))
        damping = H4_x.get_damping(strain_in_pct=np.geomspace(0.0001, 6, num=50))
        damping_bench = [2.3463, 2.3679, 2.3949, 2.4286, 2.4705, 2.5227,
                         2.5876, 2.6682, 2.768, 2.8913, 3.0433, 3.2299,
                         3.4578, 3.7348, 4.0692, 4.4697, 4.9449, 5.5026,
                         6.149, 6.8874, 7.7176, 8.6347, 9.6289, 10.686,
                         11.786, 12.909, 14.032, 15.134, 16.194, 17.195,
                         18.124, 18.971, 19.727, 20.389, 20.955, 21.425,
                         21.801, 22.089, 22.292, 22.417, 22.471, 22.462,
                         22.396, 22.282, 22.125, 21.933, 21.71, 21.463,
                         21.197, 20.914]
        self.assertTrue(np.allclose(damping, damping_bench, atol=7.))

    def test_plot_curves(self):
        data = {'gamma_t': 1, 'a': 2, 'gamma_ref': 3, 'beta': 4, 's': 5,
                'Gmax': 6, 'mu': 7, 'Tmax': 8, 'd': 9}
        hhp = HH_Param(data)
        hhp.plot_curves()

        data_ = {'gamma_ref': 1, 'beta': 2, 's': 3, 'Gmax': 4}
        mkzp = MKZ_Param(data_)
        mkzp.plot_curves()

    def test_hh_param_multi_layer(self):
        # Test that the class constructor can correctly read from a file
        HH_x = HH_Param_Multi_Layer('./files/HH_X_FKSH14.txt')
        self.assertEqual(len(HH_x), 5)
        self.assertEqual(HH_x.n_layer, 5)

        # Test that the class constructor can correctly read from a 2D array
        HH_x_array = np.genfromtxt('./files/HH_X_FKSH14.txt')
        HH_x_from_array = HH_Param_Multi_Layer(HH_x_array)
        self.assertTrue(np.allclose(HH_x.serialize_to_2D_array(),
                                    HH_x_from_array.serialize_to_2D_array()))

        # Test list operations
        del HH_x[3]
        self.assertEqual(len(HH_x), 4)
        self.assertEqual(len(HH_x), 4)

        # Test the content of list elements
        HH_x_1 = HH_x[1]
        self.assertTrue(isinstance(HH_x_1, HH_Param))
        self.assertTrue(HH_x_1.keys(), {'gamma_t', 'a', 'gamma_ref', 'beta',
                                        's', 'Gmax', 'mu', 'Tmax', 'd'})
        self.assertTrue(np.allclose(hh.serialize_params_to_array(HH_x_1),
                                    [0.027916, 1.01507, 0.0851825, 23.468,
                                     0.638322, 5.84163, 183.507, 29.7071, 1]))

    def test_mkz_param_multi_layer(self):
        # Test that the class constructor can correctly read from a file
        H4_G = MKZ_Param_Multi_Layer('./files/H4_G_IWTH04.txt')
        self.assertEqual(len(H4_G), 14)
        self.assertEqual(H4_G.n_layer, 14)

        # Test that the class constructor can correctly read from a 2D array
        H4_G_array = np.genfromtxt('./files/H4_G_IWTH04.txt')
        H4_G_from_array = MKZ_Param_Multi_Layer(H4_G_array)
        self.assertTrue(np.allclose(H4_G.serialize_to_2D_array(),
                                    H4_G_from_array.serialize_to_2D_array()))

        # Test list operations
        del H4_G[6]
        self.assertEqual(len(H4_G), 13)
        self.assertEqual(H4_G.n_layer, 13)

        # Test the content of list elements
        H4_G_1 = H4_G[1]
        self.assertTrue(isinstance(H4_G_1, MKZ_Param))
        self.assertTrue(H4_G_1.keys(), {'gamma_ref', 'beta', 's', 'Gmax'})
        self.assertTrue(np.allclose(mkz.serialize_params_to_array(H4_G_1),
                                    [0.000856, 0, 0.88832, 1.7492], atol=1e-6))

    def test_construct_curves(self):
        HH_x = HH_Param_Multi_Layer('./files/HH_X_FKSH14.txt')
        mgdc = HH_x.construct_curves()
        curves = mgdc.get_curve_matrix()
        self.assertEqual(mgdc.n_layer, HH_x.n_layer)
        self.assertEqual(curves.shape[1], HH_x.n_layer * 4)

        H4_G = MKZ_Param_Multi_Layer('./files/H4_G_IWTH04.txt')
        mgdc = H4_G.construct_curves()
        curves = mgdc.get_curve_matrix()
        self.assertEqual(mgdc.n_layer, H4_G.n_layer)
        self.assertEqual(curves.shape[1], H4_G.n_layer * 4)

    def test_param_serialize(self):
        HH_x = HH_Param({'gamma_t': 1, 'a': 2, 'gamma_ref': 3, 'beta': 4,
                         's': 5, 'Gmax': 6, 'mu': 7, 'Tmax': 8, 'd': 9})
        HH_x_array = HH_x.serialize()
        self.assertTrue(np.allclose(HH_x_array, [1, 2, 3, 4, 5, 6, 7, 8, 9]))

        H4_x = MKZ_Param({'gamma_ref': 5, 's': 6, 'beta': 7, 'Gmax': 8})
        H4_x_array = H4_x.serialize()
        self.assertTrue(np.allclose(H4_x_array, [5, 6, 7, 8]))

    def test_serialize_to_2D_array(self):
        HH_x = HH_Param_Multi_Layer('./files/HH_X_FKSH14.txt')
        HH_x_2D_array = HH_x.serialize_to_2D_array()
        HH_x_2D_array_bench = np.genfromtxt('./files/HH_X_FKSH14.txt')
        self.assertTrue(np.allclose(HH_x_2D_array, HH_x_2D_array_bench))

        H4_G = MKZ_Param_Multi_Layer('./files/H4_G_IWTH04.txt')
        H4_G_2D_array = H4_G.serialize_to_2D_array()
        H4_G_2D_array_bench = np.genfromtxt('./files/H4_G_IWTH04.txt')
        self.assertTrue(np.allclose(H4_G_2D_array, H4_G_2D_array_bench))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_HH_Param)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
