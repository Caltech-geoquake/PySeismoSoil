import re
import warnings

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


@pytest.mark.parametrize(
    ('param', 'exception'),
    [
        pytest.param(
            {'test': 2},
            AssertionError,
            id='incorrect_number_of_dict_items',
        ),
        pytest.param(
            # should be "Gmax"
            {'gamma_ref': 1, 's': 1, 'beta': 1, 'Gmax__': 1},
            KeyError,
            id='only_one_key_name_is_wrong',
        ),
    ],
)
def test_serialize_params_to_array__failure(
        param: dict[str, float], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        mkz.serialize_params_to_array(param)


def test_deserialize_array_to_params__success() -> None:
    param = mkz.deserialize_array_to_params(ARRAY)
    assert param == PARAM


@pytest.mark.parametrize(
    ('array', 'exception', 'match'),
    [
        pytest.param(
            [1, 2, 3, 4],
            TypeError,
            'must be a 1D numpy array',
            id='incorrect_input_data_type',
        ),
        pytest.param(
            np.array([1, 2, 3, 4, 5]),  # should be 4
            ValueError,
            re.escape(
                'MKZ parameters must have 4 values (gamma_ref, s, beta, '
                'Gmax), but there are 5.'
            ),
            id='incorrect_number_of_parameters',
        ),
    ],
)
def test_deserialize_array_to_params__failure(
        array: list[float] | np.ndarray,
        exception: type[Exception],
        match: str | None,
) -> None:
    with pytest.raises(exception, match=match):
        mkz.deserialize_array_to_params(array)


def test_serialize_params_to_array__to_files() -> None:
    param = {**PARAM, 'Gmax': 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # Gmax = 1.0: nothing is lost
        array = mkz.serialize_params_to_array(param, to_files=True)

    assert np.allclose(array, [0.1, 0, 0.2, 0.3])  # gamma_ref, 0, s, beta


def test_serialize_params_to_array__to_files_warns_if_Gmax_is_not_1() -> None:
    with pytest.warns(UserWarning, match='has no Gmax'):
        array = mkz.serialize_params_to_array(PARAM, to_files=True)

    assert np.allclose(array, [0.1, 0, 0.2, 0.3])


@pytest.mark.parametrize('b', [0.0, 1e-12], ids=['b_is_0', 'b_is_close_to_0'])
def test_deserialize_array_to_params__from_files(b: float) -> None:
    param = mkz.deserialize_array_to_params(
        np.array([0.1, b, 0.2, 0.3]), from_files=True
    )
    assert param == {'gamma_ref': 0.1, 's': 0.2, 'beta': 0.3, 'Gmax': 1.0}


@pytest.mark.parametrize(
    ('array', 'match'),
    [
        pytest.param(
            np.array([0.1, 0.2, 0.3, 0.4]),  # gamma_ref, s, beta, Gmax
            'The 2nd value ("b") of H4 parameters (gamma_ref, b, s, beta) '
            'must be 0, but it is 0.2. H4 files written by PySeismoSoil 0.7.1 '
            'or earlier',
            id='old_layout',
        ),
        pytest.param(
            np.array([0.1, 0, np.nan, 0.3]),
            'H4 parameters must be finite numbers',
            id='NaN',
        ),
        pytest.param(
            np.array([0.1, 0, 0.2]),
            'H4 parameters must have 4 values (gamma_ref, b, s, beta), but '
            'there are 3.',
            id='incorrect_number_of_parameters',
        ),
    ],
)
def test_deserialize_array_to_params__from_files_failure(
        array: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(match)):
        mkz.deserialize_array_to_params(array, from_files=True)


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
