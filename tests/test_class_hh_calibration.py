import os
import unittest
from os.path import join as _join

import numpy as np

from PySeismoSoil.class_curves import Multiple_GGmax_Curves
from PySeismoSoil.class_hh_calibration import HH_Calibration
from PySeismoSoil.class_parameters import HH_Param_Multi_Layer
from PySeismoSoil.class_Vs_profile import Vs_Profile

f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_HH_Calibration(unittest.TestCase):
    def test_init__success_without_curve(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        hh_c = HH_Calibration(vs_profile)  # noqa: F841

    def test_init__wrong_vs_profile_type(self):
        with self.assertRaisesRegex(TypeError, 'must be of type Vs_Profile'):
            HH_Calibration(np.array([1, 2, 3, 4, 5]))

    def test_init__success_with_vs_profile_and_curve(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = Multiple_GGmax_Curves(_join(f_dir, 'curve_FKSH14.txt'))
        hh_c = HH_Calibration(vs_profile, GGmax_curves=curves)  # noqa: F841

    def test_init__incorrect_curves_type(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        msg = 'If `GGmax_curves` is not `None`, it must be of type Multiple_GGmax_Curves'
        with self.assertRaisesRegex(TypeError, msg):
            HH_Calibration(vs_profile, GGmax_curves=np.array([1, 2, 3]))

    def test_init__incorrect_length_of_curves(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = Multiple_GGmax_Curves(_join(f_dir, 'curve_FKSH14.txt'))
        del curves[-1]  # remove the last layer
        msg = 'The number of layers implied in `GGmax_curves` and `vs_profile` must be the same.'
        with self.assertRaisesRegex(ValueError, msg):
            HH_Calibration(vs_profile, GGmax_curves=curves)

    def test_init__incorrect_Tmax_type(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = Multiple_GGmax_Curves(_join(f_dir, 'curve_FKSH14.txt'))
        Tmax = [1, 2, 3, 4, 5]
        msg = '`Tmax_profile` must be a 1D numpy array.'
        with self.assertRaisesRegex(TypeError, msg):
            HH_Calibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)

    def test_init__incorrect_Tmax_length(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = Multiple_GGmax_Curves(_join(f_dir, 'curve_FKSH14.txt'))
        Tmax = np.array([1, 2, 3])
        msg = 'The length of `Tmax_profile` needs to equal'
        with self.assertRaisesRegex(ValueError, msg):
            HH_Calibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)

    def test_init__success_with_curves_and_Tmax(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = Multiple_GGmax_Curves(_join(f_dir, 'curve_FKSH14.txt'))
        Tmax = np.array([1, 2, 3, 4, 5])
        hh_c = HH_Calibration(
            vs_profile, GGmax_curves=curves, Tmax_profile=Tmax
        )
        self.assertTrue(isinstance(hh_c.vs_profile, Vs_Profile))
        self.assertTrue(isinstance(hh_c.GGmax_curves, Multiple_GGmax_Curves))
        self.assertTrue(isinstance(hh_c.Tmax_profile, np.ndarray))

    def test_fit__case_1_with_only_Vs_profile(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        hh_c = HH_Calibration(vs_profile)
        HH_G_param = hh_c.fit(verbose=False)
        HH_G_param_benchmark = HH_Param_Multi_Layer(
            _join(f_dir, 'HH_G_FKSH14.txt')
        )
        self.assertTrue(
            np.allclose(
                HH_G_param.serialize_to_2D_array(),
                HH_G_param_benchmark.serialize_to_2D_array(),
                rtol=1e-5,
                atol=0.0,
            ),
        )

    def test_fit__case_2_with_both_Vs_profile_and_GGmax_curves(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = Multiple_GGmax_Curves(_join(f_dir, 'curve_FKSH14.txt'))
        hh_c = HH_Calibration(vs_profile, GGmax_curves=curves)
        HH_G_param = hh_c.fit(verbose=False)
        HH_G_benchmark_data = np.array(
            [
                [
                    3.0000000e-04,
                    1.0000000e-04,
                    1.0000000e-04,
                    1.0000000e-05,
                    1.0000000e-04,
                ],
                [
                    1.0000000e02,
                    1.0000000e02,
                    1.0000000e02,
                    1.0000000e02,
                    1.0000000e02,
                ],
                [
                    2.8507234e-04,
                    5.1624545e-04,
                    9.4453471e-04,
                    1.2979909e-03,
                    1.4497853e-03,
                ],
                [
                    1.7522706e00,
                    1.7145687e00,
                    1.6405793e00,
                    1.5863492e00,
                    1.5645630e00,
                ],
                [
                    9.1900098e-01,
                    9.1900268e-01,
                    9.1899964e-01,
                    9.1900711e-01,
                    9.1899869e-01,
                ],
                [
                    2.1110400e07,
                    6.8590000e07,
                    1.4896000e08,
                    2.2544125e09,
                    3.2839763e09,
                ],
                [
                    2.3335679e-01,
                    1.9914917e-01,
                    2.5378449e-01,
                    4.1906791e-02,
                    6.5981061e-02,
                ],
                [
                    2.6500966e04,
                    6.4856617e04,
                    1.4880507e05,
                    8.0485491e05,
                    1.1078508e06,
                ],
                [
                    9.3773869e-01,
                    8.5090452e-01,
                    8.6175879e-01,
                    1.0300000e00,
                    6.8809045e-01,
                ]
            ],
        )
        HH_G_param_benchmark = HH_Param_Multi_Layer(HH_G_benchmark_data)
        self.assertTrue(
            np.allclose(
                HH_G_param.serialize_to_2D_array(),
                HH_G_param_benchmark.serialize_to_2D_array(),
                rtol=1e-2,
                atol=0.0,
            ),
        )


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(
        Test_Class_HH_Calibration
    )
    unittest.TextTestRunner(verbosity=2).run(SUITE)
