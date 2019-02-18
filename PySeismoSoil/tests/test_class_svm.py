# Author: Jian Shi

import unittest

from PySeismoSoil.class_svm import SVM
from PySeismoSoil.class_Vs_profile import Vs_Profile

class Test_Class_SVM(unittest.TestCase):
    '''
    Unit test for SVM class
    '''

    def test_init(self):
        Vs30 = 256
        z1000 = 100
        svm = SVM(Vs30, z1000, show_fig=False)
        self.assertEqual(svm.Vs30, Vs30)
        self.assertEqual(svm.z1000, z1000)

    def test_get_smooth_profile(self):
        Vs30 = 256
        z1000 = 100
        svm = SVM(Vs30, z1000, show_fig=False)
        smooth_profile = svm.get_smooth_profile(show_fig=False)
        self.assertTrue(isinstance(smooth_profile, Vs_Profile))

    def test_get_random_profile(self):
        Vs30 = 256
        z1000 = 100
        svm = SVM(Vs30, z1000, show_fig=False)
        random_profile = svm.get_random_profile(show_fig=False)
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

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_SVM)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
