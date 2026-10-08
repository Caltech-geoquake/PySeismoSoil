import unittest

import numpy as np
import pytest

from PySeismoSoil.class_svm import SVM
from PySeismoSoil.class_Vs_profile import Vs_Profile


class Test_Class_SVM(unittest.TestCase):
    def test_init(self) -> None:
        Vs30 = 256
        z1 = 100
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        assert svm.Vs30 == Vs30
        assert svm.z1 == z1

    def test_Vs_cap_is_True(self) -> None:
        Vs30 = 256
        z1 = 10
        svm = SVM(Vs30, z1=z1, Vs_cap=True)
        assert svm.base_profile.vs_profile[-1, 0] == 0
        assert svm.base_profile.vs_profile[-1, 1] == 1000

    def test_Vs_cap_is_user_defined(self) -> None:
        Vs30 = 256
        z1 = 10
        Vs_cap = 1234.5
        svm = SVM(Vs30, z1=z1, Vs_cap=Vs_cap)
        assert svm.base_profile.vs_profile[-1, 0] == 0
        assert svm.base_profile.vs_profile[-1, 1] == Vs_cap

    def test_Vs_cap_is_False(self) -> None:
        pass  # this case is hard to test; skipped for now

    def test_base_profile(self) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        base_profile = svm.base_profile
        assert isinstance(base_profile, Vs_Profile)

    def test_get_discretized_profile__fixed_thk(self) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        discr_profile = svm.get_discretized_profile(
            fixed_thk=10, show_fig=False
        )
        assert isinstance(discr_profile, Vs_Profile)
        if svm.has_bedrock_Vs:  # bedrock Vs must match
            assert svm.bedrock_Vs == discr_profile.vs_profile[-1, 1]
            assert discr_profile.vs_profile[-1, 0] == 0

    def test_get_discretized_profile__valid_Vs_increment(self) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        discr_profile = svm.get_discretized_profile(
            Vs_increment=100, show_fig=False
        )
        assert isinstance(discr_profile, Vs_Profile)
        if svm.has_bedrock_Vs:  # bedrock Vs must match
            assert svm.bedrock_Vs == discr_profile.vs_profile[-1, 1]
            assert discr_profile.vs_profile[-1, 0] == 0

    def test_get_discretized_profile__invalid_Vs_increment(self) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        with pytest.raises(ValueError, match='max Vs of the smooth profile'):
            svm.get_discretized_profile(Vs_increment=5000)

    def test_get_discretized_profile__both_input_param_are_None(self) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        with pytest.raises(ValueError, match='You need to provide either'):
            svm.get_discretized_profile(Vs_increment=None, fixed_thk=None)

    def test_get_discretized_profile__both_input_param_are_provided(
            self,
    ) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        with pytest.raises(ValueError, match='do not provide both'):
            svm.get_discretized_profile(Vs_increment=1, fixed_thk=2)

    def test_get_randomized_profile(self) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)

        # A fixed seed, because the randomized profile does not end with the
        # bedrock if the randomized Vs of the last soil layer is >= 1000 m/s
        # (see #48)
        random_profile = svm.get_randomized_profile(seed=0, show_fig=False)

        assert isinstance(random_profile, Vs_Profile)

        if svm.has_bedrock_Vs:  # bedrock Vs must match
            assert svm.bedrock_Vs == random_profile.vs_profile[-1, 1]
            assert random_profile.vs_profile[-1, 0] == 0

        # Use iteration to pick compliant randomized Vs profile
        random_profile = svm.get_randomized_profile(
            show_fig=True,
            vs30_z1_compliance=True,
            verbose=True,
        )

    def test_get_randomized_profile__seed(self) -> None:
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)

        # The same seed gives the same profile
        profile_1 = svm.get_randomized_profile(seed=5).vs_profile
        profile_2 = svm.get_randomized_profile(seed=5).vs_profile
        np.testing.assert_array_equal(profile_1, profile_2)

        # With `vs30_z1_compliance=True`, the same seed gives the same
        # compliant profile. (The profile from seed 5 is not compliant, so the
        # search moves on to seed 6, 7, ...)
        profile_5 = svm.get_randomized_profile(
            seed=5,
            vs30_z1_compliance=True,
            verbose=False,
        ).vs_profile
        profile_6 = svm.get_randomized_profile(
            seed=5,
            vs30_z1_compliance=True,
            verbose=False,
        ).vs_profile
        np.testing.assert_array_equal(profile_5, profile_6)
        assert not np.array_equal(profile_1, profile_5)

        # Without a seed, every call gives a different profile. (The seed used
        # to come from the current time in seconds, so all the calls within
        # the same second gave the same profile.)
        profile_3 = svm.get_randomized_profile().vs_profile
        profile_4 = svm.get_randomized_profile().vs_profile
        assert not np.array_equal(profile_3, profile_4)

        # numpy's global random state is not changed. (This checks the legacy
        # global random state on purpose, hence the `noqa`.)
        np.random.seed(0)  # noqa: NPY002
        expected = np.random.random()  # noqa: NPY002
        np.random.seed(0)  # noqa: NPY002
        svm.get_randomized_profile(seed=5)
        svm.get_randomized_profile()
        assert expected == np.random.random()  # noqa: NPY002

    def test_index_closest(self) -> None:
        array = [0, 1, 2, 1.1, 0.4, -3.2]
        i, val = SVM._find_index_closest(array, 2.1)
        assert (i, val) == (2, 2)

        i, val = SVM._find_index_closest(array, -9)
        assert (i, val) == (5, -3.2)

        i, val = SVM._find_index_closest(array, -0.5)
        assert (i, val) == (0, 0)

        i, val = SVM._find_index_closest([1], 10000)
        assert (i, val) == (0, 1)

        with pytest.raises(ValueError, match='length of `array` needs to'):
            SVM._find_index_closest([], 2)

    def test_svm_profiles_plotting(self) -> None:
        vs30 = 256  # m/s
        z1 = 200  # m
        svm = SVM(vs30, z1=z1, show_fig=False)
        svm.get_discretized_profile(fixed_thk=20, show_fig=True)
        svm.get_discretized_profile(Vs_increment=1, show_fig=True)
        svm.get_discretized_profile(Vs_increment=100, show_fig=True)
        svm.get_randomized_profile(show_fig=True)


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_SVM)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
