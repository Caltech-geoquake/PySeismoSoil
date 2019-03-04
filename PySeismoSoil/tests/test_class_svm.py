# Author: Jian Shi

import unittest

#import PySeismoSoil.helper_site_response as sr

from PySeismoSoil.class_svm import SVM
from PySeismoSoil.class_Vs_profile import Vs_Profile

class Test_Class_SVM(unittest.TestCase):
    '''
    Unit test for SVM class
    '''

    def test_init(self):
        Vs30 = 256
        z1 = 100
        svm = SVM(Vs30, z1, show_fig=False)
        self.assertEqual(svm.Vs30, Vs30)
        self.assertEqual(svm.z1, z1)

    def test_base_profile(self):
        Vs30 = 256
        z1 = 100
        svm = SVM(Vs30, z1, show_fig=False)
        base_profile = svm.base_profile
        self.assertTrue(isinstance(base_profile, Vs_Profile))

    def test_get_discretized_profile(self):
        Vs30 = 256
        z1 = 100
        svm = SVM(Vs30, z1, show_fig=False)

        # Test fixed_thk
        discr_profile = svm.get_discretized_profile(fixed_thk=10,
                                                    show_fig=False)
        self.assertTrue(isinstance(discr_profile, Vs_Profile))

        # Test Vs_increment
        discr_profile = svm.get_discretized_profile(Vs_increment=100,
                                                    show_fig=False)
        self.assertTrue(isinstance(discr_profile, Vs_Profile))

        # Test invalid Vs_increment
        with self.assertRaisesRegex(ValueError, 'max Vs of the smooth profile'):
            svm.get_discretized_profile(Vs_increment=5000)

        # Test input parameter checking
        with self.assertRaisesRegex(ValueError, 'You need to provide either'):
            svm.get_discretized_profile(Vs_increment=None, fixed_thk=None)
        with self.assertRaisesRegex(ValueError, 'do not provide both'):
            svm.get_discretized_profile(Vs_increment=1, fixed_thk=2)

    def test_get_randomized_profile(self):
        Vs30 = 256
        z1 = 100
        svm = SVM(Vs30, z1, show_fig=False)
        random_profile = svm.get_randomized_profile(show_fig=False)
        self.assertTrue(isinstance(random_profile, Vs_Profile))

    def test_index_closest(self):
        array = [0, 1, 2, 1.1, 0.4, -3.2]
        i, val = SVM._find_index_closest(array, 2.1)
        self.assertEqual((i, val), (2, 2))

        i, val = SVM._find_index_closest(array, -9)
        self.assertEqual((i, val), (5, -3.2))

        i, val = SVM._find_index_closest(array, -0.5)
        self.assertEqual((i, val), (0, 0))

        i, val = SVM._find_index_closest([1], 10000)
        self.assertEqual((i, val), (0, 1))

        with self.assertRaisesRegex(ValueError, 'length of `array` needs to'):
            SVM._find_index_closest([], 2)

def test_svm_profiles_plotting():
    '''
    Do not indent this function, because I don't want to produce figures
    when running All_unit_test.py.
    '''
    vs30 = 256  # m/s
    z1 = 200  # m

    svm = SVM(vs30, z1, show_fig=False)
    svm.get_discretized_profile(fixed_thk=20, show_fig=True)
    svm.get_discretized_profile(Vs_increment=30, show_fig=True)
    svm.get_randomized_profile(show_fig=True)

    return None

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_SVM)
    unittest.TextTestRunner(verbosity=2).run(SUITE)

    test_svm_profiles_plotting()
