import unittest
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

import PySeismoSoil.helper_generic as hlp
import PySeismoSoil.helper_hh_model as hh
import PySeismoSoil.helper_mkz_model as mkz
import PySeismoSoil.helper_site_response as sr

f_dir = Path(__file__).resolve().parent / 'files'


class Test_Helper_Site_Response(unittest.TestCase):
    def test_num_int(self) -> None:
        accel, _ = hlp.read_two_column_stuff(
            str(f_dir / 'two_column_data_example.txt'),
        )
        v, u = sr.num_int(accel)

        v_bench = np.array(
            [
                [0.1000, 0.1000],
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
                [0.1000, 0.0100],
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

        assert np.allclose(v, v_bench)
        assert np.allclose(u, u_bench)

    def test_num_diff(self) -> None:
        v_bench = np.array(
            [
                [0.1000, 0.1000],
                [0.2000, 0.3000],
                [0.3000, 0.6000],
                [0.4000, 1.0000],
                [0.5000, 1.5000],
                [0.6000, 1.7000],
                [0.7000, 2.0000],
                [0.8000, 2.4000],
                [0.9000, 2.9000],
                [1.0000, 3.5000],
            ],
        )

        # because the "initial offset" info is lost in num_diff
        v_bench[0, 1] = 0
        displac = np.array(
            [
                [0.1000, 0.0100],
                [0.2000, 0.0400],
                [0.3000, 0.1000],
                [0.4000, 0.2000],
                [0.5000, 0.3500],
                [0.6000, 0.5200],
                [0.7000, 0.7200],
                [0.8000, 0.9600],
                [0.9000, 1.2500],
                [1.0000, 1.6000],
            ],
        )
        veloc = sr.num_diff(displac)
        assert np.allclose(veloc, v_bench)

    def test_stratify(self) -> None:
        prof1 = np.array(
            [[3, 4, 5, 6, 0], [225, 225 * 2, 225 * 3, 225 * 2.4, 225 * 5]],
        ).T
        prof1_ = sr.stratify(prof1)

        prof1_benchmark = np.array(
            [
                [1, 1, 1, 2, 2, 2.5, 2.5, 2, 2, 2, 0],
                [225, 225, 225, 450, 450, 675, 675, 540, 540, 540, 1125],
            ],
        ).T

        assert np.allclose(prof1_, prof1_benchmark)

    def test_response_spectra(self) -> None:
        accel, _ = hlp.read_two_column_stuff(
            str(f_dir / 'two_column_data_example.txt'),
        )

        T_min = 0.01
        T_max = 10
        n_pts = 50
        Tn, SA = sr.response_spectra(
            accel,
            T_min=T_min,
            T_max=T_max,
            n_pts=n_pts,
            parallel=False,
        )[:2]

        Tn_bench = np.logspace(np.log10(T_min), np.log10(T_max), n_pts)
        SA_bench = np.array(
            [
                7.0000,
                7.0000,
                7.0000,
                7.0000,
                7.0001,
                7.0002,
                6.9995,
                7.0007,
                7.0024,
                6.9941,
                7.0176,
                6.9908,
                6.9930,
                6.9615,
                7.0031,
                7.1326,
                6.9622,
                7.0992,
                6.5499,
                7.3710,
                7.3458,
                6.8662,
                8.3708,
                8.5229,
                7.9719,
                7.5457,
                8.9573,
                10.6608,
                10.5915,
                9.4506,
                8.1594,
                6.9023,
                7.1242,
                6.5462,
                6.3940,
                6.3472,
                6.7302,
                7.0554,
                7.2901,
                7.6946,
                7.6408,
                7.1073,
                6.3034,
                5.3997,
                4.5102,
                3.6991,
                2.9946,
                2.4023,
                1.9156,
                1.5218,
            ],
        )

        assert np.allclose(Tn, Tn_bench)
        assert np.allclose(SA, SA_bench, rtol=0.0001, atol=0.0)

    def test_find_f0(self) -> None:
        data, _ = hlp.read_two_column_stuff(
            str(f_dir / 'two_column_data_example.txt')
        )
        f0 = sr.find_f0(data)
        f0_benchmark = 0.5
        assert f0 == pytest.approx(f0_benchmark, abs=1e-7)

        f0_incr = sr.find_f0(np.array([[0.1, 0.2, 0.3], [1, 2, 3]]).T)
        f0_decr = sr.find_f0(np.array([[0.1, 0.2, 0.3], [3, 2, 1]]).T)

        assert f0_incr == pytest.approx(
            0.3, abs=1e-7
        )  # monotonically increasing
        assert f0_decr == pytest.approx(
            0.1, abs=1e-7
        )  # monotonically decreasing

    def test_get_xi_rho(self) -> None:
        vs = np.array([100, 300, 500, 700, 900])
        xi, rho = sr.get_xi_rho(vs, formula_type=1)
        assert np.allclose(xi, [0.05, 0.02, 0.02, 0.02, 0.01])
        assert np.allclose(rho, [1600, 1800, 1800, 1800, 2000])

        assert np.allclose(
            sr.get_xi_rho(vs, formula_type=2)[0],
            [0.0484, 0.0295, 0.0167, 0.0108, 0.0077],
            atol=0.01,
            rtol=0.0,
        )
        assert np.allclose(
            sr.get_xi_rho(vs, formula_type=3)[0],
            [0.0833, 0.0278, 0.0167, 0.0119, 0.0093],
            atol=0.01,
            rtol=0.0,
        )

    def test_calc_Vs30_and_VsZ(self) -> None:
        vs_profile = np.array([[10, 10, 10, 10], [200, 300, 400, 500]]).T
        vs30 = sr.calc_Vs30(vs_profile)
        vs40 = sr.calc_VsZ(vs_profile, 40)
        vs30_benchmark = scipy.stats.hmean(vs_profile[:, 1][:3])
        vs40_benchmark = scipy.stats.hmean(vs_profile[:, 1])
        assert vs30 == pytest.approx(vs30_benchmark, abs=1e-7)
        assert vs40 == pytest.approx(vs40_benchmark, abs=1e-7)

        vs_profile = np.array([[10, 10], [200, 300]]).T
        vs30 = sr.calc_Vs30(
            vs_profile, option_for_profile_shallower_than_30m=1
        )
        vs30_benchmark = scipy.stats.hmean([200, 300, 300])
        assert vs30 == pytest.approx(vs30_benchmark, abs=1e-7)

        vs_profile = np.array([[10, 10], [200, 300]]).T
        vs30 = sr.calc_Vs30(
            vs_profile, option_for_profile_shallower_than_30m=2
        )
        vs30_benchmark = scipy.stats.hmean(vs_profile[:, 1])
        assert vs30 == pytest.approx(vs30_benchmark, abs=1e-7)

    def test_calc_z1__normal_case__Vs_reaches_1000_meters_per_sec(
            self,
    ) -> None:
        vs_prof_1 = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 1000, 1200]]).T
        assert sr.calc_z1(vs_prof_1) == pytest.approx(12, abs=1e-7)

    def test_calc_z1__abnormal_case__Vs_doesnt_reaches_1000_meters_per_sec(
            self,
    ) -> None:
        # Abnormal case: Vs does not reach 1000 m/s ---> use total depth
        vs_prof_2 = np.array([[5, 4, 3, 2, 1], [200, 500, 700, 800, 900]]).T
        assert sr.calc_z1(vs_prof_2) == pytest.approx(15, abs=1e-7)

    def test_thk2dep_and_dep2thk(self) -> None:
        thk = np.array([6, 5, 4, 3, 2, 0])
        dep_mid = np.array([3, 8.5, 13, 16.5, 19])
        dep_top = np.array([0, 6, 11, 15, 18, 20])

        assert np.allclose(sr.dep2thk(dep_top), thk)
        assert np.allclose(
            sr.dep2thk(dep_top, include_halfspace=False), thk[:-1]
        )
        assert np.allclose(sr.thk2dep(thk, midpoint=True), dep_mid)
        assert np.allclose(sr.thk2dep(thk), dep_top)

    def test_amplify_motion(self) -> None:
        time = np.linspace(0, np.pi * 4, num=1000)
        accel = np.sin(time) + np.cos(2 * time + 1) + np.sin(4 * time + 2)
        input_motion = np.column_stack((time, accel))

        # make a simple artificial transfer function
        freq = np.logspace(-4, 2, num=1000)
        a = 2
        b = 3
        amp = a * np.ones_like(freq)  # all frequencies are amplified a times
        phase = -b * np.ones_like(freq)  # all frequencies are delayed by b rad

        motion_out = sr.amplify_motion(
            input_motion,
            (freq, (amp, phase)),
            show_fig=True,
            taper=False,
        )[0]

        # you can calculate the response by hand:
        accel_out_bench = (
            a * np.sin(time - b)
            + a * np.cos(2 * time + 1 - b)
            + a * np.sin(4 * time + 2 - b)
        )
        motion_out_bench = np.column_stack((time, accel_out_bench))

        assert np.all(np.isreal(motion_out))
        assert np.allclose(motion_out, motion_out_bench, atol=0.02, rtol=0.0)

    def test_gen_profile_plot_array(self) -> None:
        thk = np.array([1, 2, 3, 4])
        vs = np.array([5, 6, 7, 8])
        zmax = 15

        x, y = sr._gen_profile_plot_array(thk, vs, zmax)

        x_benchmark = [5, 5, 6, 6, 7, 7, 8, 8]
        y_benchmark = [0, 1, 1, 3, 3, 6, 6, zmax]

        assert np.allclose(x, x_benchmark)
        assert np.allclose(y, y_benchmark)

    def test_calc_GGmax_from_stress_strain_curve__linear_case_GGmax_is_1(
            self,
    ) -> None:
        # Test linear stress strain: G/Gmax should be all 1's
        strain = np.array([0.1, 0.2, 0.3])
        stress = np.array([2, 4, 6])
        GGmax = sr.calc_GGmax_from_stress_strain(strain, stress)
        assert np.allclose(GGmax, [1, 1, 1])

    def test_calc_GGmax_from_stress_strain_curve__a_hand_calculated_case(
            self,
    ) -> None:
        strain = np.array([0.1, 0.2, 0.3])
        stress = np.array([4, 6, 7])
        GGmax = sr.calc_GGmax_from_stress_strain(strain, stress)
        assert np.allclose(GGmax, [1.0, 3.0 / 4, 7.0 / 3.0 / 4])

    def test_calc_damping_from_stress_strain__case_1(self) -> None:
        # Case 1: Test linear stress strain: damping should be 0
        strain = np.array([0.1, 0.2, 0.3])
        stress = np.array([2, 4, 6])
        Gmax = stress[0] / strain[0]
        damping = sr.calc_damping_from_stress_strain(strain, stress, Gmax)
        assert np.allclose(damping, [0, 0, 0])

    def test_calc_damping_from_stress_strain__case_2(self) -> None:
        # Case 2: Test elasto-perfectly-plastic: damping can be hand-calculated
        #
        #                 ^ stress
        #                _|___________
        #               / |    /     /
        #              /  |   /     /
        #             /   |  /     /
        #            /    | /     /
        #           /     |/     /
        #   -------/------+-----/-----------> strain
        #         /      O|    /
        #        /        |   /
        #       /         |  /
        #      /          | /
        #     /___________|/
        #                 |
        #
        strain = np.array([0.1, 0.2, 0.3, 0.4])
        stress = np.array([1, 2, 2, 2])
        Gmax = stress[0] / strain[0]
        damping = sr.calc_damping_from_stress_strain(strain, stress, Gmax)
        assert np.allclose(damping, [0, 0, 2.0 / 3.0 / np.pi, 1.0 / np.pi])

    def test_calc_damping_from_stress_strain__case_3(self) -> None:
        # Case 3: An edge case -- the initial damping is, in theory, almost 0
        strain_in_1 = np.array(
            [0.0001, 0.00011514, 0.000132571, 0.000152642, 0.000175751],
        )
        stress = np.array([274768, 304917, 336106, 369023, 403429])
        Gmax = 3102980000.0
        damping = sr.calc_damping_from_stress_strain(strain_in_1, stress, Gmax)
        assert damping[0] >= 0.0  # make sure it is >= 0

    def test_fit_all_damping_curves__success(self) -> None:
        data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
        curve = data[:, 2:4]
        res = sr.fit_all_damping_curves(
            [curve],
            hh.fit_HH_x_single_layer,
            hh.tau_HH,
            pop_size=1,
            n_gen=1,
        )
        assert isinstance(res, list)
        assert isinstance(res[0], dict)
        assert len(res[0]) == 9  # HH model: 9 parameters

    def test_fit_all_damping_curves__exception_when_no_func_serialize(
            self,
    ) -> None:
        data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
        curve = data[:, 2:4]
        with pytest.raises(
            ValueError, match='provide a function to serialize'
        ):
            sr.fit_all_damping_curves(
                [curve],
                hh.fit_HH_x_single_layer,
                hh.tau_HH,
                pop_size=1,
                n_gen=1,
                save_txt=True,
                func_serialize=None,
            )

    def test_fit_all_damping_curves__exception_with_incorrect_func_serialize(
            self,
    ) -> None:
        data = np.genfromtxt(f_dir / 'curve_FKSH14.txt')
        curve = data[:, 2:4]
        with pytest.raises(AssertionError, match=''):
            sr.fit_all_damping_curves(
                [curve],
                hh.fit_HH_x_single_layer,
                hh.tau_HH,
                pop_size=1,
                n_gen=1,
                save_txt=True,
                txt_filename='1.txt',  # no effect anyways
                func_serialize=mkz.serialize_params_to_array,
            )

    def test_plot_site_amp(self) -> None:
        # Test that `_plot_site_amp()` can plot figures without transfer
        # functions
        time = np.linspace(0, np.pi * 4, num=1000)
        accel_in = np.sin(time) + np.cos(2 * time + 1) + np.sin(4 * time + 2)
        accel_out = np.cos(time) + np.sin(2 * time + 1) + np.cos(4 * time + 2)
        input_motion = np.column_stack((time, accel_in))
        output_motion = np.column_stack((time, accel_out))
        sr._plot_site_amp(input_motion, output_motion, None, None)

    def test_align_two_time_arrays__normal_case__not_identical_arrays(
            self,
    ) -> None:
        t1 = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        t2 = np.array([0.1, 0.2, 0.3])
        t_ = sr._align_two_time_arrays(t1, t2)
        benchmark = np.linspace(0.01, 0.3, num=int(0.3 / 0.01))
        assert np.allclose(t_, benchmark)

    def test_align_two_time_arrays__normal_case__identical_arrays(
            self,
    ) -> None:
        t1 = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        t2 = t1.copy()
        t_ = sr._align_two_time_arrays(t1, t2)
        benchmark = t1.copy()
        assert np.allclose(t_, benchmark)

    def test_align_two_time_arrays__failure__length_less_than_2(self) -> None:
        t1 = np.array([3])
        t2 = np.array([1, 2, 3])
        with pytest.raises(ValueError, match='Both time arrays need to have'):
            sr._align_two_time_arrays(t1, t2)


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(
        Test_Helper_Site_Response
    )
    unittest.TextTestRunner(verbosity=2).run(SUITE)
