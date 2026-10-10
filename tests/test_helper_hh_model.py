import numpy as np
import pytest

import PySeismoSoil.helper_hh_model as hh
import PySeismoSoil.helper_site_response as sr

STRAIN = np.logspace(-2, 1, num=12)
ATOL = 1e-4
PARAM = {
    'gamma_t': 0.1,
    'a': 0.2,
    'gamma_ref': 0.3,
    'beta': 0.4,
    's': 0.5,
    'Gmax': 0.6,
    'mu': 0.7,
    'Tmax': 0.8,
    'd': 0.9,
}
ARRAY = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) / 10.0


def test_tau_FKZ() -> None:
    T = hh.tau_FKZ(STRAIN, Gmax=4, mu=3, d=2, Tmax=1)
    assert np.allclose(
        T,
        [
            0.0012,
            0.0042,
            0.0146,
            0.0494,
            0.1543,
            0.3904,
            0.6922,
            0.8876,
            0.9652,
            0.9898,
            0.9971,
            0.9992,
        ],
        atol=ATOL,
        rtol=0.0,
    )


def test_transition_function() -> None:
    w = hh.transition_function(STRAIN, a=3, gamma_t=0.05)
    assert np.allclose(
        w,
        [
            1.0000,
            1.0000,
            1.0000,
            0.9997,
            0.9980,
            0.9872,
            0.9216,
            0.6411,
            0.2136,
            0.0396,
            0.0062,
            0.0010,
        ],
        atol=ATOL,
        rtol=0.0,
    )


def test_tau_HH() -> None:
    T = hh.tau_HH(
        STRAIN,
        gamma_t=1,
        a=2,
        gamma_ref=3,
        beta=4,
        s=5,
        Gmax=6,
        mu=7,
        Tmax=8,
        d=9,
    )
    assert np.allclose(
        T,
        [
            0.0600,
            0.1124,
            0.2107,
            0.3948,
            0.7397,
            1.3861,
            2.5966,
            4.8387,
            8.0452,
            4.1873,
            0.4678,
            0.1269,
        ],
        atol=ATOL,
        rtol=0.0,
    )


def test_calc_damping_from_param() -> None:
    xi = sr.calc_damping_from_param(PARAM, STRAIN, hh.tau_HH)
    assert np.allclose(
        xi,
        [
            0.0000,
            0.0085,
            0.0139,
            0.0192,
            0.0256,
            0.0334,
            0.0430,
            0.0544,
            0.0675,
            0.0820,
            0.0973,
            0.1128,
        ],
        atol=ATOL,
        rtol=0.0,
    )


def test_serialize_params_to_array__success() -> None:
    array = hh.serialize_params_to_array(PARAM)
    assert np.allclose(array, ARRAY)


@pytest.mark.parametrize(
    ('param', 'exception'),
    [
        pytest.param(
            {'test': 2},
            AssertionError,
            id='incorrect_number_of_dict_items',
        ),
        # Only one key name is wrong
        pytest.param(
            {
                'gamma_t': 1,
                'a': 1,
                'gamma_ref': 1,
                'beta': 1,
                's': 1,
                'Gmax': 1,
                'mu': 1,
                'Tmax': 1,
                'd___': 1,  # should be "d"
            },
            KeyError,
            id='incorrect_dict_keys',
        ),
    ],
)
def test_serialize_params_to_array__failure(
        param: dict[str, float], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        hh.serialize_params_to_array(param)


def test_deserialize_array_to_params__success() -> None:
    param = hh.deserialize_array_to_params(ARRAY)
    assert param == PARAM


@pytest.mark.parametrize(
    ('array', 'exception', 'match'),
    [
        pytest.param(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            TypeError,
            'must be a 1D numpy array',
            id='incorrect_input_data_type',
        ),
        pytest.param(
            np.arange(8),  # should have 9 parameters
            AssertionError,
            None,
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
        hh.deserialize_array_to_params(array)
