import unittest
import numpy as np
import matplotlib.pyplot as plt

from PySeismoSoil.class_Vs_profile import Vs_Profile

import PySeismoSoil.helper_generic as hlp

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_Vs_Profile(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        data, _ = hlp.read_two_column_stuff(_join(f_dir, 'two_column_data_example.txt'))
        data[:, 0] *= 10
        data[:, 1] *= 100
        prof = Vs_Profile(data)
        self.prof = prof
        super().__init__(methodName=methodName)

    def test_Vs_profile_format__case_1(self):
        # Test `None` as `data`
        data = None
        with self.assertRaisesRegex(TypeError, 'must be a file name or a numpy array'):
            Vs_Profile(data)

    def test_Vs_profile_format__case_2(self):
        # Test other type as `data`
        data = 3.6
        with self.assertRaisesRegex(TypeError, 'must be a file name or a numpy array'):
            Vs_Profile(data)

    def test_Vs_profile_format__case_3(self):
        # Test NaN values
        data = np.array([[10, 20, 30, 0], [100, 120, 160, 190]], dtype=float).T
        data[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'should contain no NaN values'):
            Vs_Profile(data)

    def test_Vs_profile_format__case_4(self):
        # Test non-positive values in thickness
        data = np.array([[10, 20, 30, 0], [100, 120, 160, 190]], dtype=float).T
        data[2, 0] = 0
        with self.assertRaisesRegex(ValueError, 'should be all positive, except'):
            Vs_Profile(data)

    def test_Vs_profile_format__case_5(self):
        # Test negative values in last layer thickness
        data = np.array([[10, 20, 30, 0], [100, 120, 160, 190]], dtype=float).T
        data[-1, 0] = -1
        with self.assertRaisesRegex(ValueError, 'last layer thickness should be'):
            Vs_Profile(data)

    def test_Vs_profile_format__case_6(self):
        # Test correct number of dimensions
        data = np.array([[[1, 2, 3, 0], [1, 2, 3, 4]]]).T  # one more dimension
        with self.assertRaisesRegex(ValueError, 'should be a 2D numpy array'):
            Vs_Profile(data)

    def test_Vs_profile_format__case_7(self):
        # Test negative values in Vs
        data = np.array([[10, 20, 30, 0], [100, 120, 160, 190]], dtype=float).T
        data[2, 1] = -1
        with self.assertRaisesRegex(ValueError, 'Vs column should be all positive.'):
            Vs_Profile(data)

    def test_Vs_profile_format__case_8(self):
        # Test non-positive values in damping and density
        data = np.array(
            [
                [10, 20, 30, 0],
                [100, 120, 160, 190],
                [0.01, 0.01, 0.01, 0.01],
                [1600, 1600, 1600, 1600],
                [1, 2, 3, 0],
            ],
            dtype=float,
        ).T
        data_ = data.copy()
        data_[2, 3] = 0
        with self.assertRaisesRegex(ValueError, 'damping and density columns'):
            Vs_Profile(data_)

    def test_Vs_profile_format__case_9(self):
        # Test "material number" column: all integers
        data = np.array(
            [
                [10, 20, 30, 0],
                [100, 120, 160, 190],
                [0.01, 0.01, 0.01, 0.01],
                [1600, 1600, 1600, 1600],
                [1, 2, 3, 0],
            ],
            dtype=float,
        ).T
        data_ = data.copy()
        data_[1, -1] = 2.2
        with self.assertRaisesRegex(ValueError, 'should be all integers'):
            Vs_Profile(data_)

    def test_Vs_profile_format__case_10(self):
        # Test "material number" column: all positive
        data = np.array(
            [
                [10, 20, 30, 0],
                [100, 120, 160, 190],
                [0.01, 0.01, 0.01, 0.01],
                [1600, 1600, 1600, 1600],
                [1, 2, 3, 0],
            ],
            dtype=float,
        ).T
        data_ = data.copy()
        data_[1, -1] = 0
        with self.assertRaisesRegex(ValueError, 'should be all positive'):
            Vs_Profile(data_)

    def test_Vs_profile_format__case_11(self):
        # Test "material number" column: last layer should >= 0
        data = np.array(
            [
                [10, 20, 30, 0],
                [100, 120, 160, 190],
                [0.01, 0.01, 0.01, 0.01],
                [1600, 1600, 1600, 1600],
                [1, 2, 3, 0],
            ],
            dtype=float,
        ).T
        data_ = data.copy()
        data_[-1, -1] = -1
        with self.assertRaisesRegex(ValueError, 'last layer should be non-negative'):
            Vs_Profile(data_)

    def test_Vs_profile_format__case_12(self):
        # Test correct number of columns
        data = np.array(
            [
                [10, 20, 30, 0],
                [100, 120, 160, 190],
                [0.01, 0.01, 0.01, 0.01],
                [1600, 1600, 1600, 1600],
                [1, 2, 3, 0],
            ],
            dtype=float,
        ).T
        data_ = data[:, 0:-1]  # one fewer column
        with self.assertRaisesRegex(ValueError, 'either 2 or 5 columns'):
            Vs_Profile(data_)

    def test_plot(self):
        self.prof.plot(c='r', ls='--')

        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes()
        self.prof.plot(fig=fig, ax=ax, label='profile')
        ax.legend(loc='best')

    def test_add_halfspace__case_1__already_a_half_space(self):
        data = np.genfromtxt(
            _join(f_dir, 'sample_profile.txt'),
        )  # already has halfspace
        prof_1 = Vs_Profile(data, add_halfspace=False)
        prof_2 = Vs_Profile(data, add_halfspace=True)

        self.assertEqual(0, prof_1._thk[-1])
        self.assertEqual(0, prof_2._thk[-1])
        self.assertNotEqual(0, prof_1._thk[-2])
        self.assertNotEqual(0, prof_2._thk[-2])  # assert only one "halfspace"
        self.assertEqual(12, prof_1.n_layer)
        self.assertEqual(prof_1.n_layer, prof_2.n_layer)

    def test_add_halfspace__case_1__no_half_space(self):
        data = np.genfromtxt(
            _join(f_dir, 'two_column_data_example.txt'),
        )  # no halfspace
        prof_1 = Vs_Profile(data, add_halfspace=False)
        prof_2 = Vs_Profile(data, add_halfspace=True)

        self.assertNotEqual(0, prof_1._thk[-1])
        self.assertEqual(0, prof_2._thk[-1])
        self.assertNotEqual(0, prof_2._thk[-2])  # assert only one "halfspace"
        self.assertNotEqual(0, prof_1._thk[-2])
        self.assertEqual(15, prof_1.n_layer)
        self.assertEqual(prof_1.n_layer, prof_2.n_layer)

    def test_vs30(self):
        self.assertAlmostEqual(self.prof.vs30, 276.9231, delta=1e-4)

    def test_get_amplif_function(self):
        profile_FKSH14 = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        af_RO = profile_FKSH14.get_ampl_function(freq_resolution=0.5, fmax=15)[0]
        af_benchmark = np.array(
            [
                [0.50, 1.20218233839345],  # from MATLAB
                [1, 2.40276180417506],
                [1.50, 3.35891308492276],
                [2, 1.52759821088595],
                [2.50, 1.23929961393844],
                [3, 1.44564547138629],
                [3.50, 2.43924932880498],
                [4, 4.01661301316906],
                [4.50, 2.32960159664501],
                [5, 1.79841404983353],
                [5.50, 1.96256021192571],
                [6, 3.12817017367637],
                [6.50, 3.92494425374814],
                [7, 2.10815322297781],
                [7.50, 1.66638537272089],
                [8, 1.95562752738785],
                [8.50, 3.37394970215842],
                [9, 2.59724801539598],
                [9.50, 1.57980154466212],
                [10, 1.42540110327715],
                [10.5, 1.82180321630950],
                [11, 3.04707962007382],
                [11.5, 2.60349869620899],
                [12, 1.84273534851058],
                [12.5, 1.79995928341286],
                [13, 2.35928076072069],
                [13.5, 3.59881564870728],
                [14, 3.31112261403936],
                [14.5, 2.61283127927210],
                [15, 2.69868407060282],
            ],
        )
        self.assertTrue(
            np.allclose(
                af_RO.spectrum_2col,
                af_benchmark,
                atol=1e-9,
                rtol=0.0,
            ),
        )

    def test_f0_BH(self):
        self.assertAlmostEqual(self.prof.get_f0_BH(), 1.05, delta=1e-2)

    def test_f0_RO(self):
        self.assertAlmostEqual(self.prof.get_f0_RO(), 1.10, delta=1e-2)

    def test_truncate__case_1(self):
        # Case 1: Truncation in the middle of a layer
        data = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]]).T
        prof = Vs_Profile(data)
        new_prof = prof.truncate(depth=8, Vs=2000)
        benchmark = np.array([[5, 3, 0], [200, 500, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

    def test_truncate__case_2(self):
        # Case 2: Truncation on the boundary of a layer
        data = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]]).T
        prof = Vs_Profile(data)
        new_prof = prof.truncate(depth=9, Vs=2000)
        benchmark = np.array([[5, 4, 0], [200, 500, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

    def test_truncate__case_3(self):
        # Case 3: Truncation beyond the total depth of original profile
        data = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]]).T
        prof = Vs_Profile(data)
        new_prof = prof.truncate(depth=30, Vs=2000)
        benchmark = np.array([[5, 4, 3, 2, 16, 0], [200, 500, 700, 1000, 1200, 2000]]).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

    def test_truncate__case_3b(self):
        # Case 3b: Truncation beyond the total depth of original profile
        data_ = np.array([[5, 4, 3, 2, 1, 0], [200, 500, 700, 1000, 1200, 1500]]).T
        prof = Vs_Profile(data_)
        new_prof = prof.truncate(depth=30, Vs=2000)
        benchmark = np.array(
            [[5, 4, 3, 2, 1, 15, 0], [200, 500, 700, 1000, 1200, 1500, 2000]],
        ).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

    def test_truncate__case_3c(self):
        # Case 3c: Truncation beyond the total depth of original profile
        data_ = np.array([[5, 4, 3, 2, 1, 0], [200, 500, 700, 1000, 1200, 1200]]).T
        prof = Vs_Profile(data_)
        new_prof = prof.truncate(depth=30, Vs=2000)
        benchmark = np.array(
            [[5, 4, 3, 2, 1, 15, 0], [200, 500, 700, 1000, 1200, 1200, 2000]],
        ).T
        self.assertTrue(np.allclose(new_prof.vs_profile[:, :2], benchmark))

    def test_query_Vs_at_depth__query_numpy_array(self):
        prof = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))

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
        result = prof.query_Vs_at_depth(np.array([7, 8, 9]), as_profile=False)
        is_all_close = np.allclose(result, [190, 280, 280])
        self.assertTrue(is_all_close)

        # (6) Test depth at ground sufrace and interface
        result = prof.query_Vs_at_depth(np.array([0, 1, 2, 3]), as_profile=False)
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

    def test_query_Vs_at_depth__query_Vs_Profile_objects(self):
        prof = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))

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
        result = prof.query_Vs_at_depth(np.array([0, 1, 2, 3, 9]), as_profile=True)
        benchmark = Vs_Profile(np.array([[1, 1, 1, 6, 0], [120, 120, 190, 190, 280]]).T)
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

    def test_query_Vs_given_thk__using_a_scalar_as_thk(self):
        prof = Vs_Profile(_join(f_dir, 'sample_profile.txt'))

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
        is_all_close = np.allclose(
            result,
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 120, 120],
        )
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
        result = prof.query_Vs_given_thk(
            9,
            n_layers=1,
            as_profile=True,
            at_midpoint=False,
            add_halfspace=True,
        )
        benchmark = Vs_Profile(np.array([[9, 10], [0, 10]]))
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (5b) Test returning Vs_Profile object: one layer, mid point of layers
        result = prof.query_Vs_given_thk(
            9,
            n_layers=1,
            as_profile=True,
            at_midpoint=True,
            add_halfspace=True,
        )
        benchmark = Vs_Profile(np.array([[9, 50], [0, 50]]))
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (5c) Test returning Vs_Profile object: one layer, mid point of layers
        result = prof.query_Vs_given_thk(
            9,
            n_layers=1,
            as_profile=True,
            at_midpoint=True,
            add_halfspace=False,
        )
        benchmark = Vs_Profile(np.array([[9, 50]]))
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (6a) Test returning Vs_Profile object: multiple layers, top of layers
        result = prof.query_Vs_given_thk(
            3,
            n_layers=5,
            as_profile=True,
            at_midpoint=False,
            add_halfspace=True,
            show_fig=True,
        )
        benchmark = Vs_Profile(
            np.array([[3, 3, 3, 3, 3, 0], [10, 40, 70, 100, 120, 120]]).T,
        )
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)

        # (6b) Test returning Vs_Profile object: multiple layers, mid of layers
        result = prof.query_Vs_given_thk(
            3,
            n_layers=5,
            as_profile=True,
            at_midpoint=True,
            add_halfspace=True,
            show_fig=True,
        )
        benchmark = Vs_Profile(
            np.array([[3, 3, 3, 3, 3, 0], [20, 50, 80, 110, 120, 120]]).T,
        )
        compare = np.allclose(result.vs_profile, benchmark.vs_profile)
        self.assertTrue(compare)


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Vs_Profile)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
