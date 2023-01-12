import unittest
import numpy as np

import PySeismoSoil.helper_hh_calibration as hhc

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Helper_HH_Calibration(unittest.TestCase):
    def test_calc_rho(self):
        h = np.array([2, 3, 4, 5])
        Vs = np.array([200, 300, 400, 500])
        rho = hhc._calc_rho(h, Vs)
        expected_rho = np.array([1.6500, 2.1272, 2.1399, 2.1702]) * 1000
        self.assertTrue(np.allclose(rho, expected_rho, rtol=0.001, atol=0.0))

    def test_calc_Gmax(self):
        rho = np.array([1600, 1700, 1800])
        Vs = np.array([200, 300, 400])
        self.assertTrue(
            np.allclose(
                hhc._calc_Gmax(Vs, rho),
                [6.4e7, 1.53e8, 2.88e8],
                rtol=1e-2,
                atol=0.0,
            ),
        )

    def test_calc_vertical_stress(self):
        h = np.array([80, 90, 100])
        rho = np.array([1600, 1700, 1800])
        sigma = hhc._calc_vertical_stress(h, rho)
        sigma_ = [627840, 2006145, 3639510]  # from MATLAB
        self.assertTrue(np.allclose(sigma, sigma_, rtol=1e-3, atol=0.0))

    def test_calc_OCR__case_1_no_upper_limit(self):
        Vs = np.array([200, 300, 400])
        rho = np.array([1600, 1700, 1800])
        sigma_v0 = np.array([6e4, 8e4, 1e5])
        OCR = hhc._calc_OCR(Vs, rho, sigma_v0)
        OCR_bench = [4.26254237, 5.80208548, 7.08490535]
        self.assertTrue(np.allclose(OCR, OCR_bench))

    def test_calc_OCR__case_2_with_an_upper_limit_of_6(self):
        Vs = np.array([200, 300, 400])
        rho = np.array([1600, 1700, 1800])
        sigma_v0 = np.array([6e4, 8e4, 1e5])
        OCR = hhc._calc_OCR(Vs, rho, sigma_v0, OCR_upper_limit=6.0)
        OCR_bench = [4.26254237, 5.80208548, 6.0]
        self.assertTrue(np.allclose(OCR, OCR_bench))

    def test_calc_K0__case_1_phi_is_a_scalar(self):
        OCR = np.array([1, 2, 3, 4, 5])
        phi = 30
        K0 = hhc._calc_K0(OCR, phi=phi)
        self.assertTrue(
            np.allclose(
                K0,
                [0.5, 0.707, 0.866, 1.0, 1.118],
                atol=1e-3,
                rtol=0.0,
            ),
        )

    def test_calc_K0__case_1_phi_is_a_vector(self):
        OCR = np.array([1, 2, 3, 4, 5])
        phi = np.array([30, 40, 50, 60, 70])
        K0 = hhc._calc_K0(OCR, phi=phi)
        self.assertTrue(
            np.allclose(
                K0,
                [[0.5, 0.5577, 0.54279, 0.44506, 0.2736]],
                atol=1e-3,
                rtol=0.0,
            ),
        )

    def test_calc_PI(self):
        Vs = np.array([100, 300, 500])
        PI = hhc._calc_PI(Vs)
        self.assertTrue(np.allclose(PI, [10, 5, 0]))

    def test_calc_shear_strength(self):
        Vs = np.array([300, 600, 900])
        OCR = np.array([8, 6, 4])
        sigma_v0 = np.array([6e4, 8e4, 1e5])
        K0 = 0.4
        phi = 35.0
        Tmax = hhc._calc_shear_strength(Vs, OCR, sigma_v0, K0=K0, phi=phi)
        Tmax_bench = [106405.11792473, 112706.83771548, 44359.02160861]
        self.assertTrue(np.allclose(Tmax, Tmax_bench, rtol=1e-3, atol=0.0))

    def test_calc_mean_confining_stress(self):
        sigma_v0 = np.array([1e6, 1.5e6, 2.2e6])
        K0 = np.array([0.4, 0.55, 0.7])
        sigma_m0 = hhc._calc_mean_confining_stress(sigma_v0, K0)
        self.assertTrue(np.allclose(sigma_m0, [0.6e6, 1.05e6, 1.76e6]))

    def test_produce_Darendeli_curves__normal_case(self):
        # Case #1: normal case
        strain = np.array([0.001, 0.01])
        sigma_v0 = np.array([3000, 6000, 9000])
        PI = np.array([10, 10, 10])
        K0 = np.array([0.4, 0.4, 0.4])
        OCR = np.array([2, 2, 2])
        GGmax, D, gamma_ref = hhc.produce_Darendeli_curves(
            sigma_v0,
            PI=PI,
            OCR=OCR,
            K0=K0,
            strain_in_pct=strain * 100,
        )

        GGmax_bench = np.array(
            [
                [0.1223930, 0.1482878, 0.165438],  # from MATLAB
                [0.0165279, 0.0205492, 0.023331],
            ],
        )
        self.assertTrue(np.allclose(GGmax, GGmax_bench, atol=1e-5, rtol=0.0))

        D_bench = np.array([
            [0.2041934, 0.1906769, 0.1827475],
            [0.2305960, 0.2260726, 0.2236005],
        ])
        self.assertTrue(np.allclose(D, D_bench, atol=1e-5, rtol=0.0))

        gamma_ref_bench = np.array([0.1172329, 0.1492445, 0.1718822]) * 1e-3
        self.assertTrue(np.allclose(gamma_ref, gamma_ref_bench, atol=1e-5, rtol=0.0))

    def test_produce_Darendeli_curves__one_of_the_inputs_has_incorrect_length(self):
        # Case #2: one of the inputs has incorrect length
        strain = np.array([0.001, 0.01])
        sigma_v0 = np.array([3000, 6000, 9000])
        PI = np.array([10, 10, 10])
        K0 = np.array([0.4, 0.4, 0.4])
        OCR = np.array([2, 2, 2])
        with self.assertRaisesRegex(ValueError, '`PI` must have length 3, but not 6'):
            hhc.produce_Darendeli_curves(
                sigma_v0,
                PI=np.append(PI, PI),
                OCR=OCR,
                K0=K0,
                strain_in_pct=strain * 100,
            )

    def test_optimization_kernel__case_1(self):
        # Comparing results with MATLAB for different test cases
        x = np.geomspace(1e-6, 0.1, num=400)  # unit: 1
        x_ref = 0.000924894
        beta = 1.0
        s = 0.919
        Gmax = 5711500000.0
        Tmax = 1111840.0
        mu = 1.0
        a, x_t, d = hhc._optimization_kernel(x, x_ref, beta, s, Gmax, Tmax, mu)
        param_bench = [100.0, 0.000104122, 0.944975]  # results by MATLAB
        self.assertTrue(np.allclose([a, x_t, d], param_bench))

    def test_optimization_kernel__case_2(self):
        # Comparing results with MATLAB for different test cases
        x = np.geomspace(1e-6, 0.1, num=400)  # unit: 1
        x_ref = 0.000423305
        beta = 1.0
        s = 0.919
        Gmax = 238271000.0
        Tmax = 120482.0
        mu = 0.0605238
        a, x_t, d = hhc._optimization_kernel(x, x_ref, beta, s, Gmax, Tmax, mu)
        param_bench = [100.0, 0.000101161, 0.702563]
        self.assertTrue(np.allclose([a, x_t, d], param_bench))

    def test_optimization_kernel__case_3(self):
        # Comparing results with MATLAB for different test cases
        x = np.geomspace(1e-6, 0.1, num=400)  # unit: 1
        x_ref = 0.000600213
        beta = 1.0
        s = 0.919
        Gmax = 1112580000.0
        Tmax = 450959.0
        mu = 0.045854
        a, x_t, d = hhc._optimization_kernel(x, x_ref, beta, s, Gmax, Tmax, mu)
        param_bench = [100.0, 5.06128e-05, 0.686577]
        self.assertTrue(np.allclose([a, x_t, d], param_bench))

    def test_optimization_kernel__case_4(self):
        # Comparing results with MATLAB for different test cases
        x = np.geomspace(1e-6, 0.1, num=400)  # unit: 1
        x_ref = 0.000898872
        beta = 1.0
        s = 0.919
        Gmax = 5932650000.0
        Tmax = 1031310.0
        mu = 1.0
        a, x_t, d = hhc._optimization_kernel(x, x_ref, beta, s, Gmax, Tmax, mu)
        param_bench = [100.0, 0.000101161, 0.934121]
        self.assertTrue(np.allclose([a, x_t, d], param_bench))

    def test_hh_param_from_profile(self):
        vs_profile = np.genfromtxt(_join(f_dir, 'profile_FKSH14.txt'))
        HH_G_param = hhc.hh_param_from_profile(
            vs_profile, show_fig=False, verbose=False,
        )
        HH_G_benchmark = np.genfromtxt(_join(f_dir, 'HH_G_FKSH14.txt'))  # calculated by MATLAB
        # use low tolerance because the whole process is highly reproducible
        self.assertTrue(np.allclose(HH_G_param, HH_G_benchmark, rtol=1e-5, atol=0.0))

    def test_hh_param_from_curves__case_1(self):
        # Case 1: Fit G/Gmax curves generated using Darendeli (2001)
        vs_profile = np.genfromtxt(_join(f_dir, 'profile_FKSH14.txt'))
        curves = np.genfromtxt(_join(f_dir, 'curve_FKSH14.txt'))
        HH_G_param = hhc.hh_param_from_curves(
            vs_profile,
            curves,
            show_fig=False,
            verbose=False,
        )
        HH_G_benchmark = np.array(
            [
                [0.0003, 0.0001, 0.0001, 0.0001, 0.0001],
                [100, 100, 100, 100, 100],
                [0.000285072, 0.000516205, 0.000944545, 0.00129825, 0.00144835],
                [1.75224, 1.71444, 1.64057, 1.58664, 1.56314],
                [0.918975, 0.919001, 0.918973, 0.919007, 0.918999],
                [2.11104e07, 6.859e07, 1.4896e08, 2.25441e09, 3.28398e09],
                [0.233357, 0.199149, 0.253784, 1, 1],
                [26501, 64856.6, 148805, 804855, 1.10785e06],
                [0.937739, 0.850905, 0.861759, 0.984774, 0.981156],
            ],
        )
        # use higher tolerance because MKZ curve fitting has room for small errors
        self.assertTrue(np.allclose(HH_G_param, HH_G_benchmark, rtol=1e-2, atol=0.0))

    def test_hh_param_from_curves__case_2(self):
        # Case 2: Fit manually specified ("real-world") G/Gmax curves
        #         (Unable to benchmark because MKZ curve fitting can produce
        #          different parameters with similar curve-fitting error.)
        #         The user needs to do a visual inspection.
        vs_profile = np.genfromtxt(_join(f_dir, 'profile_P001.txt'))
        curves = np.genfromtxt(_join(f_dir, 'curve_P001.txt'))
        HH_G_param = hhc.hh_param_from_curves(  # noqa: F841
            vs_profile,
            curves,
            show_fig=True,
            verbose=False,
        )


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_HH_Calibration)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
