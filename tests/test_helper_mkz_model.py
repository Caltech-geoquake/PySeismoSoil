import numpy as np
import pytest
from scipy import stats

import PySeismoSoil.helper_mkz_model as mkz
import PySeismoSoil.helper_site_response as sr

STRAIN = np.logspace(-2, 1, num=12)
ATOL = 1e-4
PARAM = {'gamma_ref': 0.1, 's': 0.2, 'beta': 0.3, 'Gmax': 0.4}
ARRAY = np.array([1, 2, 3, 4]) / 10.0


def test_tau_MKZ() -> None:
    T = mkz.tau_MKZ(STRAIN, gamma_ref=1, beta=2, s=3, Gmax=4)

    # note: benchmark results come from comparable functions in MATLAB
    assert np.allclose(
        T,
        [
            0.0400,
            0.0750,
            0.1404,
            0.2630,
            0.4913,
            0.9018,
            1.4898,
            1.5694,
            0.7578,
            0.2413,
            0.0700,
            0.0200,
        ],
        atol=ATOL,
        rtol=0.0,
    )


def test_calc_damping_from_param() -> None:
    xi = sr.calc_damping_from_param(PARAM, STRAIN, mkz.tau_MKZ)
    assert np.allclose(
        xi,
        [
            0,
            0.0072,
            0.0101,
            0.0119,
            0.0133,
            0.0147,
            0.0163,
            0.0178,
            0.0195,
            0.0213,
            0.0232,
            0.0251,
        ],
        atol=ATOL,
        rtol=0.0,
    )


def test_serialize_params_to_array__success() -> None:
    array = mkz.serialize_params_to_array(PARAM)
    assert np.allclose(array, ARRAY)


def test_serialize_params_to_array__incorrect_number_of_dict_items() -> None:
    with pytest.raises(AssertionError, match=''):
        mkz.serialize_params_to_array({'test': 2})


def test_serialize_params_to_array__only_one_key_name_is_wrong() -> None:
    with pytest.raises(KeyError, match=''):
        mkz.serialize_params_to_array(
            # should be "Gmax"
            {'gamma_ref': 1, 's': 1, 'beta': 1, 'Gmax__': 1},
        )


def test_deserialize_array_to_params__success() -> None:
    param = mkz.deserialize_array_to_params(ARRAY)
    assert param == PARAM


def test_deserialize_array_to_params__incorrect_input_data_type() -> None:
    with pytest.raises(TypeError, match='must be a 1D numpy array'):
        mkz.deserialize_array_to_params([1, 2, 3, 4])


def test_deserialize_array_to_params__incorrect_number_of_parameters() -> None:
    with pytest.raises(AssertionError, match=''):
        # should be 4
        mkz.deserialize_array_to_params(np.array([1, 2, 3, 4, 5]))


def test_fit_MKZ() -> None:
    strain_in_1 = np.geomspace(1e-6, 0.1, num=50)  # unit: 1
    strain_in_pct = strain_in_1 * 100

    param_1 = {'gamma_ref': 0.0035, 'beta': 0.85, 's': 1.0, 'Gmax': 1e6}
    T_MKZ_1 = mkz.tau_MKZ(strain_in_1, **param_1)
    GGmax_1 = sr.calc_GGmax_from_stress_strain(strain_in_1, T_MKZ_1)

    param_2 = {'gamma_ref': 0.02, 'beta': 1.4, 's': 0.7, 'Gmax': 2e7}
    T_MKZ_2 = mkz.tau_MKZ(strain_in_1, **param_2)
    GGmax_2 = sr.calc_GGmax_from_stress_strain(strain_in_1, T_MKZ_2)

    damping = np.ones_like(strain_in_pct)  # dummy values
    curve_data = np.column_stack((
        strain_in_pct,
        GGmax_1,
        strain_in_pct,
        damping,
        strain_in_pct,
        GGmax_2,
        strain_in_pct,
        damping,
    ))
    _, fitted_curve = mkz.fit_MKZ(curve_data, show_fig=True)

    # Make sure that the R^2 score between data and fit >= 0.99
    GGmax_fitted_1 = np.interp(
        strain_in_pct,
        fitted_curve[:, 0],
        fitted_curve[:, 1],
    )
    GGmax_fitted_2 = np.interp(
        strain_in_pct,
        fitted_curve[:, 4],
        fitted_curve[:, 5],
    )
    r2_1 = stats.linregress(GGmax_1, GGmax_fitted_1)[2]
    r2_2 = stats.linregress(GGmax_2, GGmax_fitted_2)[2]
    assert r2_1 >= 0.99
    assert r2_2 >= 0.99
