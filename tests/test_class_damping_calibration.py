import unittest
import numpy as np

from PySeismoSoil.class_Vs_profile import Vs_Profile
from PySeismoSoil.class_damping_calibration import Damping_Calibration

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_Damping_Calibration(unittest.TestCase):
    def test_init__case_1_incorrect_input_type(self):
        vs_profile_array = np.genfromtxt(_join(f_dir, 'profile_FKSH14.txt'))
        with self.assertRaisesRegex(TypeError, 'must be of type Vs_Profile'):
            Damping_Calibration(vs_profile_array)

    def test_init__case_2_correct_input_type(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        Damping_Calibration(vs_profile)

    def test_get_damping_curves__check_n_layers_correct(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        d_cal = Damping_Calibration(vs_profile)
        strain_in_pct = np.logspace(-2, 1, num=23)
        mdc = d_cal.get_damping_curves(
            strain_in_pct=strain_in_pct,
            use_Darendeli_Dmin=True,
            show_fig=True,
        )
        self.assertEqual(mdc.n_layer, vs_profile.n_layer)

    def test_get_damping_curves__check_two_things(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        d_cal = Damping_Calibration(vs_profile)
        strain_in_pct = np.logspace(-2, 1, num=23)
        mdc = d_cal.get_damping_curves(
            strain_in_pct=strain_in_pct,
            use_Darendeli_Dmin=True,
            show_fig=True,
        )
        # Check two things:
        #  (1) The unit of strains are correct (i.e., %), and no interpolation happened
        #  (2) The unit of the damping ratios are correct (i.e., %)
        curve_matrix = mdc.get_curve_matrix()
        for i in range(mdc.n_layer):
            gamma = curve_matrix[:, i * 4 + 2]
            xi = curve_matrix[:, i * 4 + 3]
            self.assertTrue(np.allclose(gamma, strain_in_pct, atol=1e-5, rtol=0.0))
            self.assertGreaterEqual(xi[-1], 1.0)  # last element of each damping curve

    def test_get_damping_curves__check_use_Darendeli_Dmin_correct(self):
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        d_cal = Damping_Calibration(vs_profile)
        strain_in_pct = np.logspace(-2, 1, num=23)
        mdc = d_cal.get_damping_curves(
            strain_in_pct=strain_in_pct,
            use_Darendeli_Dmin=True,
            show_fig=True,
        )
        curve_matrix = mdc.get_curve_matrix()

        # Check `use_Darendeli_Dmin` operates as expected:
        mdc_ = d_cal.get_damping_curves(
            strain_in_pct=strain_in_pct,
            use_Darendeli_Dmin=False,
            show_fig=True,
        )
        curve_matrix_ = mdc_.get_curve_matrix()
        for i in range(mdc_.n_layer):
            xi = curve_matrix[:, i * 4 + 3]  # damping with Darendeli's D_min
            xi_ = curve_matrix_[:, i * 4 + 3]  # damping specified in `vs_profile`
            self.assertTrue(
                np.allclose(
                    xi_[0],
                    vs_profile.vs_profile[i, 2] * 100,
                    atol=1e-5,
                    rtol=0.0,
                ),
            )
            self.assertTrue(np.allclose(xi_ - xi_[0], xi - xi[0]))

    def test_get_HH_x_param(self):
        # Only test that `get_HH_x_param()` can run without bugs:
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        d_cal = Damping_Calibration(vs_profile)
        d_cal.get_HH_x_param(
            pop_size=1,
            n_gen=1,
            save_txt=False,
            use_scipy=True,
            show_fig=True,
        )

    def test_get_H4_x_param(self):
        # Only test that `get_H4_x_param()` can run without bugs:
        vs_profile = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        d_cal = Damping_Calibration(vs_profile)
        d_cal.get_H4_x_param(
            pop_size=1,
            n_gen=1,
            save_txt=False,
            use_scipy=True,
            show_fig=True,
        )


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Damping_Calibration)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
