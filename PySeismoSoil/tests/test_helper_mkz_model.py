# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_mkz_model as mkz
import PySeismoSoil.helper_site_response as sr

class Test_Helper_MKZ_Model(unittest.TestCase):
    '''
    Unit tests for helper functions in helper_mkz_model.py
    '''

    def __init__(self, methodName='runTest'):
        self.strain = np.logspace(-2, 1, num=12)
        self.atol = 1e-4
        self.param = {'gamma_ref': 0.1, 's': 0.2, 'beta': 0.3, 'Gmax': 0.4}
        self.array = np.array([1, 2, 3, 4]) / 10.0
        super(Test_Helper_MKZ_Model, self).__init__(methodName=methodName)

    def test_tau_MKZ(self):
        T = mkz.tau_MKZ(self.strain, gamma_ref=1, beta=2, s=3, Gmax=4)
        # note: benchmark results come from comparable functions in MATLAB
        self.assertTrue(np.allclose(T, [0.0400, 0.0750, 0.1404, 0.2630, 0.4913,
                                        0.9018, 1.4898, 1.5694, 0.7578, 0.2413,
                                        0.0700, 0.0200], atol=self.atol))

    def test_calc_damping_from_param(self):
        xi = sr.calc_damping_from_param(self.param, self.strain, mkz.tau_MKZ)
        self.assertTrue(np.allclose(xi, [0, 0.0072, 0.0101, 0.0119, 0.0133,
                                         0.0147, 0.0163, 0.0178, 0.0195, 0.0213,
                                         0.0232, 0.0251], atol=self.atol))

    def test_serialize_params_to_array(self):
        array = mkz.serialize_params_to_array(self.param)
        self.assertTrue(np.allclose(array, self.array))

    def test_deserialize_array_to_params(self):
        param = mkz.deserialize_array_to_params(self.array)
        self.assertEqual(param, self.param)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_MKZ_Model)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
