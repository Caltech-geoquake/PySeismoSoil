import unittest

import numpy as np
import pytest

import PySeismoSoil.helper_hh_model as hh
import PySeismoSoil.helper_site_response as sr


class Test_Helper_HH_Model(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest') -> None:
        self.strain = np.logspace(-2, 1, num=12)
        self.atol = 1e-4
        self.param = {
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
        self.array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) / 10.0
        super().__init__(methodName=methodName)

    def test_tau_FKZ(self) -> None:
        T = hh.tau_FKZ(self.strain, Gmax=4, mu=3, d=2, Tmax=1)
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
            atol=self.atol,
            rtol=0.0,
        )

    def test_transition_function(self) -> None:
        w = hh.transition_function(self.strain, a=3, gamma_t=0.05)
        assert np.allclose(
            w,
            [
                1.0,
                1.0,
                1.0,
                0.9997,
                0.998,
                0.9872,
                0.9216,
                0.6411,
                0.2136,
                0.0396,
                0.0062,
                0.001,
            ],
            atol=self.atol,
            rtol=0.0,
        )

    def test_tau_HH(self) -> None:
        T = hh.tau_HH(
            self.strain,
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
                0.06,
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
            atol=self.atol,
            rtol=0.0,
        )

    def test_calc_damping_from_param(self) -> None:
        xi = sr.calc_damping_from_param(self.param, self.strain, hh.tau_HH)
        assert np.allclose(
            xi,
            [
                0.0,
                0.0085,
                0.0139,
                0.0192,
                0.0256,
                0.0334,
                0.043,
                0.0544,
                0.0675,
                0.082,
                0.0973,
                0.1128,
            ],
            atol=self.atol,
            rtol=0.0,
        )

    def test_serialize_params_to_array__success(self) -> None:
        array = hh.serialize_params_to_array(self.param)
        assert np.allclose(array, self.array)

    def test_serialize_params_to_array__incorrect_number_of_dict_items(
            self,
    ) -> None:
        with pytest.raises(AssertionError, match=''):
            hh.serialize_params_to_array({'test': 2})

    def test_serialize_params_to_array__incorrect_dict_keys(self) -> None:
        # Only one key name is wrong
        with pytest.raises(KeyError, match=''):
            hh.serialize_params_to_array(
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
            )

    def test_deserialize_array_to_params__success(self) -> None:
        param = hh.deserialize_array_to_params(self.array)
        assert param == self.param

    def test_deserialize_array_to_params__incorrect_input_data_type(
            self,
    ) -> None:
        with pytest.raises(TypeError, match='must be a 1D numpy array'):
            hh.deserialize_array_to_params([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_deserialize_array_to_params__incorrect_number_of_parameters(
            self,
    ) -> None:
        with pytest.raises(AssertionError, match=''):
            # should have 9 parameters
            hh.deserialize_array_to_params(np.arange(8))


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_HH_Model)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
