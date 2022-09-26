import unittest
import numpy as np

from PySeismoSoil.class_hh_calibration import HHCalibration
from PySeismoSoil.class_Vs_profile import VsProfile
from PySeismoSoil.class_curves import MultipleGGmaxCurves
from PySeismoSoil.class_parameters import MultiLayerParamHH

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_HH_Calibration(unittest.TestCase):
    def test_init__success_without_curve(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        hh_c = HHCalibration(vs_profile)  # noqa: F841

    def test_init__wrong_vs_profile_type(self):
        with self.assertRaisesRegex(TypeError, 'must be of type VsProfile'):
            HHCalibration(np.array([1, 2, 3, 4, 5]))

    def test_init__success_with_vs_profile_and_curve(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = MultipleGGmaxCurves(_join(f_dir, 'curve_FKSH14.txt'))
        hh_c = HHCalibration(vs_profile, GGmax_curves=curves)  # noqa: F841

    def test_init__incorrect_curves_type(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        msg = 'If `GGmax_curves` is not `None`, it must be of type MultipleGGmaxCurves'
        with self.assertRaisesRegex(TypeError, msg):
            HHCalibration(vs_profile, GGmax_curves=np.array([1, 2, 3]))

    def test_init__incorrect_length_of_curves(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = MultipleGGmaxCurves(_join(f_dir, 'curve_FKSH14.txt'))
        del curves[-1]  # remove the last layer
        msg = 'The number of layers implied in `GGmax_curves` and `vs_profile` must be the same.'
        with self.assertRaisesRegex(ValueError, msg):
            HHCalibration(vs_profile, GGmax_curves=curves)

    def test_init__incorrect_Tmax_type(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = MultipleGGmaxCurves(_join(f_dir, 'curve_FKSH14.txt'))
        Tmax = [1, 2, 3, 4, 5]
        msg = '`Tmax_profile` must be a 1D numpy array.'
        with self.assertRaisesRegex(TypeError, msg):
            HHCalibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)

    def test_init__incorrect_Tmax_length(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = MultipleGGmaxCurves(_join(f_dir, 'curve_FKSH14.txt'))
        Tmax = np.array([1, 2, 3])
        msg = 'The length of `Tmax_profile` needs to equal'
        with self.assertRaisesRegex(ValueError, msg):
            HHCalibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)

    def test_init__success_with_curves_and_Tmax(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = MultipleGGmaxCurves(_join(f_dir, 'curve_FKSH14.txt'))
        Tmax = np.array([1, 2, 3, 4, 5])
        hh_c = HHCalibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)
        self.assertTrue(isinstance(hh_c.vs_profile, VsProfile))
        self.assertTrue(isinstance(hh_c.GGmax_curves, MultipleGGmaxCurves))
        self.assertTrue(isinstance(hh_c.Tmax_profile, np.ndarray))

    def test_fit__case_1_with_only_Vs_profile(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        hh_c = HHCalibration(vs_profile)
        HH_G_param = hh_c.fit(verbose=False)
        HH_G_param_benchmark = MultiLayerParamHH(_join(f_dir, 'HH_G_FKSH14.txt'))
        self.assertTrue(np.allclose(
            HH_G_param.serialize_to_2D_array(),
            HH_G_param_benchmark.serialize_to_2D_array(),
            rtol=1e-5,
            atol=0.0,
        ))

    def test_fit__case_2_with_both_Vs_profile_and_GGmax_curves(self):
        vs_profile = VsProfile(_join(f_dir, 'profile_FKSH14.txt'))
        curves = MultipleGGmaxCurves(_join(f_dir, 'curve_FKSH14.txt'))
        hh_c = HHCalibration(vs_profile, GGmax_curves=curves)
        HH_G_param = hh_c.fit(verbose=False)
        HH_G_benchmark_data = np.array(
            [[0.0003, 0.0001, 0.0001, 0.0001, 0.0001],
             [100, 100, 100, 100, 100],
             [0.000285072, 0.000516205, 0.000944545, 0.00129825, 0.00144835],
             [1.75224, 1.71444, 1.64057, 1.58664, 1.56314],
             [0.918975, 0.919001, 0.918973, 0.919007, 0.918999],
             [2.11104e+07, 6.859e+07, 1.4896e+08, 2.25441e+09, 3.28398e+09],
             [0.233357, 0.199149, 0.253784, 1, 1],
             [26501, 64856.6, 148805, 804855, 1.10785e+06],
             [0.937739, 0.850905, 0.861759, 0.984774, 0.981156]]
        )
        HH_G_param_benchmark = MultiLayerParamHH(HH_G_benchmark_data)
        self.assertTrue(np.allclose(
            HH_G_param.serialize_to_2D_array(),
            HH_G_param_benchmark.serialize_to_2D_array(),
            rtol=1e-2,
            atol=0.0,
        ))


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_HH_Calibration)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
