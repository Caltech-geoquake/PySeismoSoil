# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_hh_model as hh
from PySeismoSoil.class_parameters import HH_Param, MKZ_Param, HH_Param_Multi_Layer

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

    def test_plot_curves(self):
        data = {'gamma_t': 1, 'a': 2, 'gamma_ref': 3, 'beta': 4, 's': 5,
                'Gmax': 6, 'mu': 7, 'Tmax': 8, 'd': 9}
        hhp = HH_Param(data)
        hhp.plot_curves()

    def test_hh_param_multi_layer(self):
        HH_x = HH_Param_Multi_Layer('./files/HH_X_FKSH14.txt')
        self.assertEqual(len(HH_x), 5)
        self.assertEqual(HH_x.n_layer, 5)

        del HH_x[3]
        self.assertEqual(len(HH_x), 4)

        HH_x_1 = HH_x[1]
        self.assertTrue(isinstance(HH_x_1, HH_Param))
        self.assertTrue(HH_x_1.keys(), {'gamma_t', 'a', 'gamma_ref', 'beta',
                                        's', 'Gmax', 'mu', 'Tmax', 'd'})
        self.assertTrue(np.allclose(hh.serialize_params_to_array(HH_x_1),
                                    [0.0324457, 1.02664, 0.203758, 44.0942,
                                     0.615992, 8.07508, 187.808, 33.9501, 1]))

    def test_param_serialize(self):
        HH_x = HH_Param({'gamma_t': 1, 'a': 2, 'gamma_ref': 3, 'beta': 4,
                         's': 5, 'Gmax': 6, 'mu': 7, 'Tmax': 8, 'd': 9})
        HH_x_array = HH_x.serialize()
        self.assertTrue(np.allclose(HH_x_array, [1, 2, 3, 4, 5, 6, 7, 8, 9]))

        H4_x = MKZ_Param({'gamma_ref': 5, 's': 6, 'beta': 7, 'Gmax': 8})
        H4_x_array = H4_x.serialize()
        self.assertTrue(np.allclose(H4_x_array, [5, 6, 7, 8]))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_HH_Param)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
