# Author: Jian Shi

import unittest
import numpy as np
import matplotlib.pyplot as plt

from PySeismoSoil.class_Vs_profile import Vs_Profile

import PySeismoSoil.helper_generic as hlp

class Test_Class_Vs_Profile(unittest.TestCase):
    '''
    Unit test for Vs_Profile class
    '''

    def __init__(self, methodName='runTest'):
        data, _ = hlp.read_two_column_stuff('./files/two_column_data_example.txt')
        data[:, 0] *= 10
        data[:, 1] *= 100
        prof = Vs_Profile(data)
        self.prof = prof
        super(Test_Class_Vs_Profile, self).__init__(methodName=methodName)

    def test_plot(self):
        self.prof.plot(c='r', ls='--')

        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes()
        self.prof.plot(fig=fig, ax=ax, label='profile')
        ax.legend(loc='best')

    def test_add_halfspace(self):
        # Test case 1: already has a half space
        data = np.genfromtxt('./files/sample_profile.txt')
        prof_1 = Vs_Profile(data, add_halfspace=False)
        prof_2 = Vs_Profile(data, add_halfspace=True)

        self.assertTrue(prof_1._thk[-1] == 0)
        self.assertTrue(prof_2._thk[-1] == 0)
        self.assertTrue(prof_1._thk[-2] != 0)
        self.assertTrue(prof_2._thk[-2] != 0)  # assert only one "halfspace"

        # Test case 2: no half space
        data = np.genfromtxt('./files/two_column_data_example.txt')
        prof_1 = Vs_Profile(data, add_halfspace=False)
        prof_2 = Vs_Profile(data, add_halfspace=True)

        self.assertTrue(prof_1._thk[-1] != 0)
        self.assertTrue(prof_2._thk[-1] == 0)
        self.assertTrue(prof_1._thk[-2] != 0)
        self.assertTrue(prof_2._thk[-2] != 0)  # assert only one "halfspace"

    def test_vs30(self):
        self.assertAlmostEqual(self.prof.vs30, 276.9231, delta=1e-4)

    def test_f0_BH(self):
        self.assertAlmostEqual(self.prof.get_f0_BH(), 1.05, delta=1e-2)

    def test_f0_RO(self):
        self.assertAlmostEqual(self.prof.get_f0_RO(), 1.10, delta=1e-2)

    def test_get_z1(self):
        # Normal case: Vs reaches 1000 m/s
        data_1 = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]]).T
        prof_1 = Vs_Profile(data_1)
        self.assertAlmostEqual(prof_1.get_z1(), 12)

        # Abnormal case: Vs does not reach 1000 m/s
        data_2 = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 800, 900]]).T
        prof_2 = Vs_Profile(data_2)
        self.assertAlmostEqual(prof_2.get_z1(), 15)

    def test_truncate(self):
        data = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]]).T
        prof = Vs_Profile(data)

        # Case 1: Truncation in the middle of a layer
        new_prof = prof.truncate(depth=8, Vs=2000)
        benchmark = np.array([[5, 3, 0], [200, 500, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

        # Case 2: Truncation on the boundary of a layer
        new_prof = prof.truncate(depth=9, Vs=2000)
        benchmark = np.array([[5, 4, 0], [200, 500, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

        # Case 3: Truncation beyond the total depth of original profile
        new_prof = prof.truncate(depth=30, Vs=2000)
        benchmark = np.array([[5, 4, 3, 2, 16, 0],
                              [200, 500, 700, 1000, 1200, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

        # Case 3b: Truncation beyond the total depth of original profile
        data_ = np.array([[5, 4, 3, 2, 1, 0], [200, 500, 700, 1000, 1200, 1500]]).T
        prof = Vs_Profile(data_)
        new_prof = prof.truncate(depth=30, Vs=2000)
        benchmark = np.array([[5, 4, 3, 2, 1, 15, 0],
                              [200, 500, 700, 1000, 1200, 1500, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

        # Case 3c: Truncation beyond the total depth of original profile
        data_ = np.array([[5, 4, 3, 2, 1, 0], [200, 500, 700, 1000, 1200, 1200]]).T
        prof = Vs_Profile(data_)
        new_prof = prof.truncate(depth=30, Vs=2000)
        benchmark = np.array([[5, 4, 3, 2, 1, 15, 0],
                              [200, 500, 700, 1000, 1200, 1200, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

    def test_query_Vs_at_depth(self):
        prof = Vs_Profile('./files/profile_FKSH14.txt')

        #--------------- Query numpy array as results -------------------------
        # (1) Test ground surface
        self.assertAlmostEqual(prof.query_Vs_at_depth(0.0, as_profile=False), 120.0)

        # (2) Test depth within a layer
        self.assertAlmostEqual(prof.query_Vs_at_depth(1.0, as_profile=False), 120.0)

        # (3) Test depth at layer interface
        self.assertAlmostEqual(prof.query_Vs_at_depth(105, as_profile=False), 1030.0)
        self.assertAlmostEqual(prof.query_Vs_at_depth(106, as_profile=False), 1210.0)
        self.assertAlmostEqual(prof.query_Vs_at_depth(107, as_profile=False), 1210.0)

        # (4) Test infinite depth
        self.assertAlmostEqual(prof.query_Vs_at_depth(1e9, as_profile=False), 1210.0)

        # (5) Test depth at layer interface -- input is an array
        result = prof.query_Vs_at_depth(np.array([7,8,9]), as_profile=False)
        is_all_close = np.allclose(result, [190, 280, 280])
        self.assertTrue(is_all_close)

        # (6) Test depth at ground sufrace and interface
        result = prof.query_Vs_at_depth(np.array([0,1,2,3]), as_profile=False)
        is_all_close = np.allclose(result, [120, 120, 190, 190])
        self.assertTrue(is_all_close)

        # (7) Test invalid input: list
        with self.assertRaisesRegex(TypeError, 'needs to be a single number'):
            prof.query_Vs_at_depth([1, 2])
        with self.assertRaisesRegex(TypeError, 'needs to be a single number'):
            prof.query_Vs_at_depth({1, 2})
        with self.assertRaisesRegex(TypeError, 'needs to be a single number'):
            prof.query_Vs_at_depth((1, 2))

        # (8) Test invalid input: negative values
        with self.assertRaisesRegex(ValueError, 'Please provide non-negative'):
            prof.query_Vs_at_depth(-2)
        with self.assertRaisesRegex(ValueError, 'Please provide non-negative'):
            prof.query_Vs_at_depth(np.array([-2, 1]))

        #--------------- Query Vs_Profile objects as results ------------------
        # (1) Test invalid input: non-increasing array
        with self.assertRaisesRegex(ValueError, 'needs to be monotonically increasing'):
            prof.query_Vs_at_depth(np.array([1, 2, 5, 4, 6]), as_profile=True)

        # (2) Test invalid input: repeated values
        with self.assertRaisesRegex(ValueError, 'should not contain duplicate values'):
            prof.query_Vs_at_depth(np.array([1, 2, 4, 4, 6]), as_profile=True)

        # (3) Test a scalar input
        result = prof.query_Vs_at_depth(1.0, as_profile=True)
        benchmark = Vs_Profile(np.array([[1, 0], [120, 120]]).T)
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (4) Test a general case
        result \
            = prof.query_Vs_at_depth(np.array([0, 1, 2, 3, 9]), as_profile=True)
        benchmark \
            = Vs_Profile(np.array([[1, 1, 1, 6, 0], [120, 120, 190, 190, 280]]).T)
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

    def test_query_Vs_given_thk(self):
        prof = Vs_Profile('./files/sample_profile.txt')

        #-------------- Using a scalar as `thk` -------------------------------
        # (1a) Test trivial case: top of layer
        result = prof.query_Vs_given_thk(9, n_layers=1, at_midpoint=False)
        is_all_close = np.allclose(result, [10])
        self.assertTrue(is_all_close)

        # (1b) Test trivial case: mid point of layer
        result = prof.query_Vs_given_thk(9, n_layers=1, at_midpoint=True)
        is_all_close = np.allclose(result, [50])
        self.assertTrue(is_all_close)

        # (2a) Test normal case: top of layer
        result = prof.query_Vs_given_thk(1, n_layers=5, at_midpoint=False)
        is_all_close = np.allclose(result, [10, 20, 30, 40, 50])
        self.assertTrue(is_all_close)

        # (2a) Test normal case: mid point of layer
        result = prof.query_Vs_given_thk(1, n_layers=14, at_midpoint=True)
        is_all_close = np.allclose(result, [10, 20, 30, 40, 50, 60, 70, 80,
                                            90, 100, 110, 120, 120, 120])
        self.assertTrue(is_all_close)

        # (3a) Test invalid input
        with self.assertRaisesRegex(ValueError, 'should be positive'):
            result = prof.query_Vs_given_thk(1, n_layers=0)

        # (3b) Test invalid input
        with self.assertRaisesRegex(TypeError, 'needs to be a scalar or a numpy'):
            result = prof.query_Vs_given_thk([1, 2], n_layers=0)

        # (4a) Test large thickness: top of layer
        result = prof.query_Vs_given_thk(100, n_layers=4, at_midpoint=False)
        is_all_close = np.allclose(result, [10, 120, 120, 120])
        self.assertTrue(is_all_close)

        # (4b) Test large thickness: mid point of layer (version 1)
        result = prof.query_Vs_given_thk(100, n_layers=4, at_midpoint=True)
        is_all_close = np.allclose(result, [120, 120, 120, 120])
        self.assertTrue(is_all_close)

        # (4c) Test large thickness: mid point of layer (version 2)
        result = prof.query_Vs_given_thk(17, n_layers=3, at_midpoint=True)
        is_all_close = np.allclose(result, [90, 120, 120])
        self.assertTrue(is_all_close)

        # (4d) Test large thickness: mid point of layer (version 3)
        result = prof.query_Vs_given_thk(18, n_layers=3, at_midpoint=True)
        is_all_close = np.allclose(result, [100, 120, 120])
        self.assertTrue(is_all_close)

        # (5a) Test returning Vs_Profile object: one layer, on top of layers
        result = prof.query_Vs_given_thk(9, n_layers=1, as_profile=True,
                                         at_midpoint=False, add_halfspace=True)
        benchmark = Vs_Profile(np.array([[9, 10], [0, 10]]))
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (5b) Test returning Vs_Profile object: one layer, mid point of layers
        result = prof.query_Vs_given_thk(9, n_layers=1, as_profile=True,
                                         at_midpoint=True, add_halfspace=True)
        benchmark = Vs_Profile(np.array([[9, 50], [0, 50]]))
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (5c) Test returning Vs_Profile object: one layer, mid point of layers
        result = prof.query_Vs_given_thk(9, n_layers=1, as_profile=True,
                                         at_midpoint=True, add_halfspace=False)
        benchmark = Vs_Profile(np.array([[9, 50]]))
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (6a) Test returning Vs_Profile object: multiple layers, top of layers
        result = prof.query_Vs_given_thk(3, n_layers=5, as_profile=True,
                                         at_midpoint=False, add_halfspace=True)
        benchmark = Vs_Profile(np.array([[3, 3, 3, 3, 3, 0],
                                         [10, 40, 70, 100, 120, 120]]).T)
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (6b) Test returning Vs_Profile object: multiple layers, mid of layers
        result = prof.query_Vs_given_thk(3, n_layers=5, as_profile=True,
                                         at_midpoint=True, add_halfspace=True)
        benchmark = Vs_Profile(np.array([[3, 3, 3, 3, 3, 0],
                                         [20, 50, 80, 110, 120, 120]]).T)
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Vs_Profile)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
