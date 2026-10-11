import numpy as np
import pytest

from PySeismoSoil.class_svm import SVM
from PySeismoSoil.class_Vs_profile import Vs_Profile


def test_init() -> None:
    Vs30 = 256
    z1 = 100
    svm = SVM(target_Vs30=256, z1=100, show_fig=False)
    assert svm.Vs30 == Vs30
    assert svm.z1 == z1


@pytest.mark.parametrize(
    ('Vs_cap', 'bedrock_Vs'),
    [
        pytest.param(True, 1000, id='True'),
        pytest.param(1234.5, 1234.5, id='user_defined'),
    ],
)
def test_Vs_cap(*, Vs_cap: bool | float, bedrock_Vs: float) -> None:
    Vs30 = 256
    z1 = 10
    svm = SVM(Vs30, z1=z1, Vs_cap=Vs_cap)
    assert svm.base_profile.vs_profile[-1, 0] == 0
    assert svm.base_profile.vs_profile[-1, 1] == bedrock_Vs


def test_Vs_cap_is_False() -> None:
    pass  # this case is hard to test; skipped for now


def test_base_profile() -> None:
    svm = SVM(target_Vs30=256, z1=100, show_fig=False)
    base_profile = svm.base_profile
    assert isinstance(base_profile, Vs_Profile)


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({'fixed_thk': 10}, id='fixed_thk'),
        pytest.param({'Vs_increment': 100}, id='valid_Vs_increment'),
    ],
)
def test_get_discretized_profile(kwargs: dict[str, float]) -> None:
    svm = SVM(target_Vs30=256, z1=100, show_fig=False)
    discr_profile = svm.get_discretized_profile(**kwargs, show_fig=False)
    assert isinstance(discr_profile, Vs_Profile)
    if svm.has_bedrock_Vs:  # bedrock Vs must match
        assert svm.bedrock_Vs == discr_profile.vs_profile[-1, 1]
        assert discr_profile.vs_profile[-1, 0] == 0


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        pytest.param(
            {'Vs_increment': 5000},
            'max Vs of the smooth profile',
            id='invalid_Vs_increment',
        ),
        pytest.param(
            {'Vs_increment': None, 'fixed_thk': None},
            'You need to provide either',
            id='both_input_param_are_None',
        ),
        pytest.param(
            {'Vs_increment': 1, 'fixed_thk': 2},
            'do not provide both',
            id='both_input_param_are_provided',
        ),
    ],
)
def test_get_discretized_profile__failure(
        kwargs: dict[str, float | None], match: str
) -> None:
    svm = SVM(target_Vs30=256, z1=100, show_fig=False)
    with pytest.raises(ValueError, match=match):
        svm.get_discretized_profile(**kwargs)


def test_get_randomized_profile() -> None:
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


def test_get_randomized_profile__seed() -> None:
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


def test_index_closest() -> None:
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


def test_svm_profiles_plotting() -> None:
    vs30 = 256  # m/s
    z1 = 200  # m
    svm = SVM(vs30, z1=z1, show_fig=False)
    svm.get_discretized_profile(fixed_thk=20, show_fig=True)
    svm.get_discretized_profile(Vs_increment=1, show_fig=True)
    svm.get_discretized_profile(Vs_increment=100, show_fig=True)
    svm.get_randomized_profile(show_fig=True)
