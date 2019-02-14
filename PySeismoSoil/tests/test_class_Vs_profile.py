# Author: Jian Shi

import unittest
import numpy as np

from PySeismoSoil.class_Vs_profile import Vs_Profile as Profile

import PySeismoSoil.helper_generic as hlp

class Test_Class_Vs_Profile(unittest.TestCase):
    '''
    Unit test for Vs_Profile class
    '''

    def test_load_data(self):

        data, _ = hlp.read_two_column_stuff('./files/two_column_data_example.txt')
        data[:, 0] *= 10
        data[:, 1] *= 100
        prof = Profile(data)

        self.assertAlmostEqual(prof.vs30, 276.9231, delta=1e-4)
        self.assertAlmostEqual(prof.get_f0_BH(), 1.05, delta=1e-2)
        self.assertAlmostEqual(prof.get_f0_RO(), 1.10, delta=1e-2)
        self.assertAlmostEqual(prof.get_Vs_at_depth(110), 700)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Vs_Profile)
    unittest.TextTestRunner(verbosity=2).run(SUITE)