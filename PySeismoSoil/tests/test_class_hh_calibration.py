# Author: Jian Shi

import unittest
import numpy as np

from PySeismoSoil.class_hh_calibration import HH_Calibration
from PySeismoSoil.class_Vs_profile import Vs_Profile
from PySeismoSoil.class_curves import Multiple_GGmax_Curves
from PySeismoSoil.class_parameters import HH_Param_Multi_Layer

class Test_Class_HH_Calibration(unittest.TestCase):
    def test_init(self):
        # This should pass
        vs_profile = Vs_Profile('./files/profile_FKSH14.txt')
        hh_c = HH_Calibration(vs_profile)

        # Type of vs_profile not correct
        with self.assertRaisesRegex(TypeError, 'must be of type Vs_Profile'):
            HH_Calibration(np.array([1, 2, 3, 4, 5]))

        # This should pass
        curves = Multiple_GGmax_Curves('./files/curve_FKSH14.txt')
        hh_c = HH_Calibration(vs_profile, GGmax_curves=curves)

        # Type of curves not correct
        with self.assertRaisesRegex(TypeError, 'If `GGmax_curves` is not `None`,'
                                    ' it must be of type Multiple_GGmax_Curves'):
            HH_Calibration(vs_profile, GGmax_curves=np.array([1, 2, 3]))

        # Length of curves not correct
        del curves[-1]  # remove the last layer
        with self.assertRaisesRegex(ValueError, 'The number of layers implied '
                                    'in `GGmax_curves` and `vs_profile` must be'
                                    ' the same.'):
            HH_Calibration(vs_profile, GGmax_curves=curves)

        # Type of Tmax not correct
        curves = Multiple_GGmax_Curves('./files/curve_FKSH14.txt')
        Tmax = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(TypeError, '`Tmax_profile` must be a 1D '
                                    'numpy array.'):
            HH_Calibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)

        # Length of Tmax not correct
        Tmax = np.array([1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'The length of `Tmax_profile` '
                                    'needs to equal'):
            HH_Calibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)

        # This should pass
        Tmax = np.array([1, 2, 3, 4, 5])
        hh_c = HH_Calibration(vs_profile, GGmax_curves=curves, Tmax_profile=Tmax)
        self.assertTrue(isinstance(hh_c.vs_profile, Vs_Profile))
        self.assertTrue(isinstance(hh_c.GGmax_curves, Multiple_GGmax_Curves))
        self.assertTrue(isinstance(hh_c.Tmax_profile, np.ndarray))

    def test_fit(self):
        # Case 1: users only have Vs profile
        vs_profile = Vs_Profile('./files/profile_FKSH14.txt')
        hh_c = HH_Calibration(vs_profile)
        HH_G_param = hh_c.fit(verbose=False)
        HH_G_param_benchmark = HH_Param_Multi_Layer('./files/HH_G_FKSH14.txt')
        self.assertTrue(np.allclose(HH_G_param.serialize_to_2D_array(),
                                    HH_G_param_benchmark.serialize_to_2D_array(),
                                    rtol=1e-5, atol=0.0))

        # Case 2: users have both Vs profile and G/Gmax curves
        curves = Multiple_GGmax_Curves('./files/curve_FKSH14.txt')
        hh_c = HH_Calibration(vs_profile, GGmax_curves=curves)
        HH_G_param = hh_c.fit(verbose=False)
        HH_G_benchmark_data \
            = np.array([[0.0003, 0.0001, 0.0001, 0.0001, 0.0001],
                        [100, 100, 100, 100, 100],
                        [0.000285072, 0.000516205, 0.000944545, 0.00129825, 0.00144835],
                        [1.75224, 1.71444, 1.64057, 1.58664, 1.56314],
                        [0.918975, 0.919001, 0.918973, 0.919007, 0.918999],
                        [2.11104e+07, 6.859e+07, 1.4896e+08, 2.25441e+09, 3.28398e+09],
                        [0.233357, 0.199149, 0.253784, 1, 1],
                        [26501, 64856.6, 148805, 804855, 1.10785e+06],
                        [0.937739, 0.850905, 0.861759, 0.984774, 0.981156]])
        HH_G_param_benchmark = HH_Param_Multi_Layer(HH_G_benchmark_data)
        self.assertTrue(np.allclose(HH_G_param.serialize_to_2D_array(),
                                    HH_G_param_benchmark.serialize_to_2D_array(),
                                    rtol=1e-2, atol=0.0))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_HH_Calibration)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
