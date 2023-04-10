import unittest
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from PySeismoSoil.class_ground_motion import Ground_Motion as GM
from PySeismoSoil.class_Vs_profile import Vs_Profile
from PySeismoSoil.class_frequency_spectrum import Frequency_Spectrum

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_Ground_Motion(unittest.TestCase):
    def test_loading_data__two_columns_from_file(self):
        # Two columns from file
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='gal')

        PGA_benchmark = 294.30  # unit: cm/s/s
        PGV_benchmark = 31.46  # unit: cm/s
        PGD_benchmark = 38.77  # unit: cm
        tol = 1e-2

        self.assertAlmostEqual(gm.pga_in_gal, PGA_benchmark, delta=tol)
        self.assertAlmostEqual(gm.pgv_in_cm_s, PGV_benchmark, delta=tol)
        self.assertAlmostEqual(gm.pgd_in_cm, PGD_benchmark, delta=tol)
        self.assertAlmostEqual(gm.peak_Arias_Intensity, 1.524, delta=tol)
        self.assertAlmostEqual(gm.rms_accel, 0.4645, delta=tol)

    def test_loading_data__two_columns_from_numpy_array(self):
        # Two columns from numpy array
        gm = GM(np.array([[0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4]]).T, unit='m/s/s')
        self.assertAlmostEqual(gm.pga, 4)

    def test_loading_data__one_column_from_file(self):
        # One column from file
        gm = GM(_join(f_dir, 'one_column_data_example.txt'), unit='g', dt=0.2)
        self.assertAlmostEqual(gm.pga_in_g, 12.0)

    def test_loading_data__one_column_from_numpy_array(self):
        # One column from numpy array
        gm = GM(np.array([1, 2, 3, 4, 5]), unit='gal', dt=0.1)
        self.assertAlmostEqual(gm.pga_in_gal, 5.0)

    def test_loading_data__one_column_without_specifying_dt(self):
        # One column without specifying dt
        error_msg = 'is needed for one-column `data`.'
        with self.assertRaisesRegex(ValueError, error_msg):
            gm = GM(np.array([1, 2, 3, 4, 5]), unit='gal')  # noqa: F841

    def test_loading_data__test_invalid_unit_names(self):
        # Test invalid unit names
        with self.assertRaisesRegex(ValueError, 'Invalid `unit` name.'):
            GM(np.array([1, 2, 3, 4, 5]), unit='test', dt=0.1)

        with self.assertRaisesRegex(ValueError, r"use '/s/s' instead of 's\^2'"):
            GM(np.array([1, 2, 3, 4, 5]), unit='m/s^2', dt=0.1)

    def test_differentiation(self):
        veloc = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 3, 7, -1, -3, 5]]).T
        gm = GM(veloc, unit='m', motion_type='veloc')
        accel_benchmark = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 20, 40, -80, -20, 80]],
        ).T
        self.assertTrue(np.allclose(gm.accel, accel_benchmark))

    def test_integration__artificial_example(self):
        gm = GM(_join(f_dir, 'two_column_data_example.txt'), unit='m/s/s')
        v_bench = np.array(
            [
                [0.1000, 0.1000],  # from MATLAB
                [0.2000, 0.3000],
                [0.3000, 0.6000],
                [0.4000, 1.0000],
                [0.5000, 1.5000],
                [0.6000, 1.7000],
                [0.7000, 2.0000],
                [0.8000, 2.4000],
                [0.9000, 2.9000],
                [1.0000, 3.5000],
                [1.1000, 3.8000],
                [1.2000, 4.2000],
                [1.3000, 4.7000],
                [1.4000, 5.3000],
                [1.5000, 6.0000],
            ],
        )
        u_bench = np.array(
            [
                [0.1000, 0.0100],  # from MATLAB
                [0.2000, 0.0400],
                [0.3000, 0.1000],
                [0.4000, 0.2000],
                [0.5000, 0.3500],
                [0.6000, 0.5200],
                [0.7000, 0.7200],
                [0.8000, 0.9600],
                [0.9000, 1.2500],
                [1.0000, 1.6000],
                [1.1000, 1.9800],
                [1.2000, 2.4000],
                [1.3000, 2.8700],
                [1.4000, 3.4000],
                [1.5000, 4.0000],
            ],
        )
        self.assertTrue(np.allclose(gm.veloc, v_bench))
        self.assertTrue(np.allclose(gm.displ, u_bench))

    def test_integration__real_world_example(self):
        # Note: In this test, the result by cumulative trapezoidal numerical
        #       integration is used as the benchmark. Since it is infeasible to
        #       achieve perfect "alignment" between the two time histories,
        #       we check the correlation coefficient instead of element-wise
        #       check.
        veloc_ = np.genfromtxt(_join(f_dir, 'sample_accel.txt'))
        gm = GM(veloc_, unit='m/s', motion_type='veloc')
        displ = gm.displ[:, 1]
        displ_cumtrapz = np.append(0, sp.integrate.cumtrapz(veloc_[:, 1], dx=gm.dt))
        r = np.corrcoef(displ_cumtrapz, displ)[1, 1]  # cross-correlation
        self.assertTrue(r >= 0.999)

    def test_fourier_transform(self):
        gm = GM(_join(f_dir, 'two_column_data_example.txt'), unit='m/s/s')
        freq, spec = gm.get_Fourier_spectrum(real_val=False).raw_data.T

        freq_bench = [
            0.6667,
            1.3333,
            2.0000,
            2.6667,
            3.3333,
            4.0000,
            4.6667,
            5.3333,
        ]
        FS_bench = [
            60.0000 + 0.0000j,
            -1.5000 + 7.0569j,
            -1.5000 + 3.3691j,
            -7.5000 + 10.3229j,
            -1.5000 + 1.3506j,
            -1.5000 + 0.8660j,
            -7.5000 + 2.4369j,
            -1.5000 + 0.1577j,
        ]
        self.assertTrue(np.allclose(freq, freq_bench, atol=0.0001, rtol=0.0))
        self.assertTrue(np.allclose(spec, FS_bench, atol=0.0001, rtol=0.0))

    def test_baseline_correction(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='m/s/s')
        corrected = gm.baseline_correct(show_fig=True)
        self.assertTrue(isinstance(corrected, GM))

    def test_high_pass_filter(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='m')
        hp = gm.highpass(cutoff_freq=1.0, show_fig=True)
        self.assertTrue(isinstance(hp, GM))

    def test_low_pass_filter(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='m')
        lp = gm.lowpass(cutoff_freq=1.0, show_fig=True)
        self.assertTrue(isinstance(lp, GM))

    def test_band_pass_filter(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='m')
        bp = gm.bandpass(cutoff_freq=[0.5, 8], show_fig=True)
        self.assertTrue(isinstance(bp, GM))

    def test_band_stop_filter(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='m')
        bs = gm.bandstop(cutoff_freq=[0.5, 8], show_fig=True)
        self.assertTrue(isinstance(bs, GM))

    def test_amplify_via_profile(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='m')
        vs_prof = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        output_motion = gm.amplify(vs_prof, boundary='elastic')
        self.assertTrue(isinstance(output_motion, GM))

    def test_deconvolution(self):
        # Assert `deconvolve()` & `amplify()` are reverse operations to each other.
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='m')
        vs_prof = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))

        for boundary in ['elastic', 'rigid']:
            deconv_motion = gm.deconvolve(vs_prof, boundary=boundary)
            output_motion = deconv_motion.amplify(vs_prof, boundary=boundary)
            self.assertTrue(self.nearly_identical(gm.accel, output_motion.accel))

            amplified_motion = gm.amplify(vs_prof, boundary=boundary)
            output_motion = amplified_motion.deconvolve(vs_prof, boundary=boundary)
            self.assertTrue(self.nearly_identical(gm.accel, output_motion.accel))

    def test_plot(self):
        filename = _join(f_dir, 'sample_accel.txt')
        gm = GM(filename, unit='m')

        fig, axes = gm.plot()  # automatically generate fig/ax objects
        self.assertTrue(isinstance(axes, tuple))
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[0].title.get_text(), os.path.split(filename)[1])

        fig2 = plt.figure(figsize=(8, 8))
        fig2_, axes = gm.plot(fig=fig2)  # feed an external figure object
        self.assertTrue(np.allclose(fig2_.get_size_inches(), (8, 8)))

    def test_unit_convert(self):
        data = np.array([1, 3, 7, -2, -10, 0])
        gm = GM(data, unit='m', dt=0.1)
        accel = gm.accel[:, 1]
        accel_in_m = gm._unit_convert(unit='m/s/s')[:, 1]
        accel_in_gal = gm._unit_convert(unit='gal')[:, 1]
        accel_in_g = gm._unit_convert(unit='g')[:, 1]
        self.assertTrue(np.allclose(accel_in_m, accel))
        self.assertTrue(np.allclose(accel_in_gal, accel * 100))
        self.assertTrue(np.allclose(accel_in_g, accel / 9.81))

    def test_scale_motion(self):
        data = np.array([1, 3, 7, -2, -10, 0])
        gm = GM(data, unit='g', dt=0.1)
        gm_scaled_1 = gm.scale_motion(factor=2.0)  # scale by 2.0
        gm_scaled_2 = gm.scale_motion(target_PGA_in_g=5.0)  # scale by 0.5
        self.assertTrue(np.allclose(gm.accel[:, 1] * 2, gm_scaled_1.accel[:, 1]))
        self.assertTrue(np.allclose(gm.accel[:, 1] * 0.5, gm_scaled_2.accel[:, 1]))

    def test_amplify_by_tf__case_1_an_artificial_transfer_function(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='gal')
        ratio_benchmark = 2.76
        freq = np.arange(0.01, 50, step=0.01)
        tf = ratio_benchmark * np.ones_like(freq)
        transfer_function = Frequency_Spectrum(np.column_stack((freq, tf)))
        new_gm = gm.amplify_by_tf(transfer_function, show_fig=False)
        ratio = new_gm.accel[:, 1] / gm.accel[:, 1]
        self.assertTrue(np.allclose(ratio, ratio_benchmark))

    def test_amplify_by_tf__case_2_a_transfer_function_from_a_Vs_profile(self):
        gm = GM(_join(f_dir, 'sample_accel.txt'), unit='gal')
        vs_prof = Vs_Profile(_join(f_dir, 'profile_FKSH14.txt'))
        tf_RO, tf_BH, _ = vs_prof.get_transfer_function()
        gm_with_tf_RO = gm.amplify_by_tf(tf_RO)
        gm_with_tf_BH = gm.amplify_by_tf(tf_BH)

        gm_with_tf_RO_ = gm.amplify(vs_prof, boundary='elastic')
        gm_with_tf_BH_ = gm.amplify(vs_prof, boundary='rigid')

        # Assert that `amplify_by_tf()` and `amplify()` can generate
        # nearly identical results
        self.assertTrue(
            self.nearly_identical(gm_with_tf_RO.accel, gm_with_tf_RO_.accel),
        )
        self.assertTrue(
            self.nearly_identical(gm_with_tf_BH.accel, gm_with_tf_BH_.accel),
        )

    @staticmethod
    def nearly_identical(motion_1, motion_2, thres=0.99):
        """
        Assert that two ground motions are nearly identical, by checking the
        correlation coefficient between two time series.

        Parameters
        ----------
        motion_1 : numpy.ndarray
            Two-column array (time, acceleration).
        motion_2 : numpy.ndarray
            Two-column array (time, acceleration).
        thres : float
            The threshold that the correlation coefficient must be above (or
            equal to).

        Returns
        -------
        result : bool
            Whether the motions are nearly identical
        """
        if not np.allclose(motion_1[:, 0], motion_2[:, 0], rtol=0.001, atol=0.0):
            return False

        r = np.corrcoef(motion_1[:, 1], motion_2[:, 1])
        if r[1, 0] < thres:
            return False

        return True


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Ground_Motion)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
