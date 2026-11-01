import pathlib

import numpy as np
import pytest

from PySeismoSoil import helper_hh_model as hh
from PySeismoSoil import helper_simulations as sim
from PySeismoSoil import helper_site_response as sr
from PySeismoSoil import helper_time_domain as td

f_dir = pathlib.Path(__file__).parent / 'files'

PROFILE = np.genfromtxt(f_dir / 'profile_FKSH14.txt')
HH_G = np.genfromtxt(f_dir / 'HH_G_FKSH14.txt')
HH_X = np.genfromtxt(f_dir / 'HH_X_FKSH14.txt')
H4_G = np.genfromtxt(f_dir / 'H4_G_IWTH04.txt')[:, :5]
H4_X = np.genfromtxt(f_dir / 'H4_x_IWTH04.txt')[:, :5]


def _motion(n_pts: int, pga: float) -> np.ndarray:
    """Return an incident motion (m/s/s) scaled from the sample motion."""
    accel = np.genfromtxt(f_dir / 'sample_accel.txt')[:n_pts]
    accel[:, 0] -= accel[0, 0]
    accel[:, 1] *= pga / np.max(np.abs(accel[:, 1]))
    return accel


def _run(model: str, motion: np.ndarray, **kwargs: object) -> tuple:
    return td.time_domain_site_response(
        sr.stratify(PROFILE), motion, model=model, **kwargs
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_output_shapes_and_depths() -> None:
    motion = _motion(400, 0.5)
    new_profile = sr.stratify(PROFILE)
    n_layer = new_profile.shape[0] - 1
    layer_depth, node_depth, *outputs = _run('linear', motion)
    assert node_depth.shape == (n_layer + 1,)
    assert layer_depth.shape == (n_layer,)
    assert np.isclose(node_depth[-1], np.sum(new_profile[:-1, 0]))
    assert np.allclose(layer_depth, (node_depth[1:] + node_depth[:-1]) / 2)
    for out in outputs[:3]:  # accel, veloc, displ
        assert out.shape == (len(motion), n_layer + 1)

    for out in outputs[3:]:  # strain, stress
        assert out.shape == (len(motion), n_layer)

    # Velocity and displacement are relative to the bottom of the profile
    assert np.allclose(outputs[1][:, -1], 0)
    assert np.allclose(outputs[2][:, -1], 0)


def test_epp_with_large_strength_is_linear() -> None:
    motion = _motion(600, 2.0)
    linear = _run('linear', motion)
    n_mat = int(np.max(PROFILE[:-1, 4]))
    epp = _run('EPP', motion, tau_max=np.full(n_mat, 1e12))
    # Identical, except for round-off errors (summing the stresses of the
    # springs, instead of one multiplication)
    for a, b in zip(linear, epp, strict=True):
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9 * np.max(np.abs(a)))


def test_epp_stress_is_bounded_by_strength() -> None:
    motion = _motion(800, 4.0)
    n_mat = int(np.max(PROFILE[:-1, 4]))
    tau_max = np.full(n_mat, 5000.0)
    out_tau = _run('EPP', motion, tau_max=tau_max)[6]
    assert np.max(np.abs(out_tau)) <= 5000.0 * (1 + 1e-9)
    assert np.max(np.abs(out_tau)) > 4000.0  # it does yield


def test_non_masing_reduces_to_masing() -> None:
    # With the same parameters for G/Gmax and damping, the H4 model (non-
    # Masing rules, analytical backbone) is the same as the H2 model (Masing
    # rules, Iwan springs), up to the discretization of the springs.
    motion = _motion(1000, 3.0)
    h2 = _run('H2', motion, G_param=H4_G)
    h4 = _run('H4', motion, G_param=H4_G, xi_param=H4_G)
    assert np.max(np.abs(h2[5])) > 1e-4  # strongly nonlinear
    assert _corr(h2[2][:, 0], h4[2][:, 0]) > 0.999
    assert np.isclose(
        np.max(np.abs(h2[2][:, 0])), np.max(np.abs(h4[2][:, 0])), rtol=0.03
    )


@pytest.mark.parametrize('boundary', ['elastic', 'rigid', 'borehole'])
def test_nonlinear_models_at_small_strains_are_linear(boundary: str) -> None:
    motion = _motion(600, 0.002)
    linear = _run('linear', motion, boundary=boundary)[2][:, 0]
    for model, G, X in [('H2', H4_G, None), ('H4', H4_G, H4_X)]:
        surface = _run(model, motion, boundary=boundary, G_param=G, xi_param=X)
        assert _corr(linear, surface[2][:, 0]) > 0.99


@pytest.mark.parametrize('pga', [1e-6, 1e-3])
def test_hh_model_at_small_strains_is_linear(pga: float) -> None:
    # Regression test: the first reversals happen at extremely small strains
    # (before the waves arrive), and the Gmax of the unloading curve must
    # still be rescaled there (the HH_x parameters have a normalized Gmax)
    motion = _motion(800, pga)
    hh_surface = _run('HH', motion, G_param=HH_G, xi_param=HH_X)[2][:, 0]
    profile = sr.stratify(PROFILE)
    # Use the Gmax of the HH parameters in the linear model too
    n_mat = int(np.max(profile[:-1, 4]))
    rho = profile[:-1, 3]
    mat = profile[:-1, 4].astype(int) - 1
    profile[:-1, 1] = np.sqrt(HH_G[5, :n_mat][mat] / rho)
    linear = td.time_domain_site_response(profile, motion, model='linear')
    lin_surface = linear[2][:, 0]
    assert _corr(lin_surface, hh_surface) > 0.99
    assert np.isclose(
        np.max(np.abs(hh_surface)), np.max(np.abs(lin_surface)), rtol=0.05
    )


def test_hh_model_runs_for_strong_motion() -> None:
    motion = _motion(1000, 4.0)
    results = _run('HH', motion, G_param=HH_G, xi_param=HH_X)
    for out in results:
        assert np.all(np.isfinite(out))

    linear = _run('linear', motion)
    # Nonlinearity reduces the peak ground acceleration
    assert np.max(np.abs(results[2][:, 0])) < np.max(np.abs(linear[2][:, 0]))


def test_time_domain_linear_matches_frequency_domain() -> None:
    motion = _motion(2000, 0.1)
    td_surface = _run('linear', motion)[2][:, 0]
    outcrop = motion.copy()
    outcrop[:, 1] *= 2
    fd_surface = sim.linear(PROFILE, outcrop)[3][: len(motion), 1]
    # Compare the amplitude spectra between 0.5 Hz and 10 Hz
    freq = np.fft.rfftfreq(len(motion), motion[1, 0] - motion[0, 0])
    band = (freq > 0.5) & (freq < 10)
    amp_td = np.abs(np.fft.rfft(td_surface))[band]
    amp_fd = np.abs(np.fft.rfft(fd_surface))[band]
    assert np.isclose(np.sum(amp_td), np.sum(amp_fd), rtol=0.1)


@pytest.mark.parametrize('xx', [1e-6, 1e-4, 3e-3, 0.05])
def test_hh_backbone_matches_tau_hh(xx: float) -> None:
    p_g = np.tile(HH_G[:, 0], (1, 1))
    gmax2 = np.array([[HH_G[5, 0]], [HH_G[5, 0]]])
    param = hh.deserialize_array_to_params(HH_G[:, 0])
    expected = hh.tau_HH(np.array([xx]), **param)[0]
    stress = td._backbone(xx, 0, 0, td._MODEL_HH, p_g, p_g, gmax2)
    assert np.isclose(stress, expected, rtol=1e-10)
    # The backbone curve is an odd function
    stress_neg = td._backbone(-xx, 0, 0, td._MODEL_HH, p_g, p_g, gmax2)
    assert np.isclose(stress_neg, -stress)


def test_reversal_memory_grows(monkeypatch: pytest.MonkeyPatch) -> None:
    motion = _motion(800, 3.0)
    expected = _run('HH', motion, G_param=HH_G, xi_param=HH_X)
    monkeypatch.setattr(td, 'INITIAL_REVERSAL_CAPACITY', 4)
    results = _run('HH', motion, G_param=HH_G, xi_param=HH_X)
    for a, b in zip(expected, results, strict=True):
        assert np.allclose(a, b)


def test_progress_callback() -> None:
    fractions = []
    _run('linear', _motion(300, 0.1), progress_callback=fractions.append)
    assert len(fractions) > 1
    assert np.all(np.diff(fractions) > 0)
    assert np.isclose(fractions[-1], 1.0)


def test_progress_callback_can_abort() -> None:
    def abort(fraction: float) -> None:
        if fraction > 0.5:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run('linear', _motion(300, 0.1), progress_callback=abort)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'model': 'MKZ'}, '`model` must be one of'),
        ({'model': 'linear', 'boundary': 'free'}, '`boundary` must be one of'),
        ({'model': 'H2'}, '`G_param` is required'),
        ({'model': 'H4', 'G_param': H4_G}, '`xi_param` is required'),
        ({'model': 'HH', 'G_param': H4_G, 'xi_param': H4_X}, 'with 9 rows'),
        ({'model': 'H4', 'G_param': H4_G[:, :2], 'xi_param': H4_X}, 'fewer'),
        ({'model': 'EPP'}, '`tau_max` is required'),
        ({'model': 'EPP', 'tau_max': [1e5]}, 'fewer values'),
    ],
)
def test_invalid_inputs(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        td.time_domain_site_response(PROFILE, _motion(100, 0.1), **kwargs)


def test_invalid_profile_and_motion() -> None:
    with pytest.raises(ValueError, match='5 columns'):
        td.time_domain_site_response(
            PROFILE[:, :2], _motion(100, 0.1), model='linear'
        )

    with pytest.raises(ValueError, match='two columns'):
        td.time_domain_site_response(
            PROFILE, _motion(100, 0.1)[:, 1], model='linear'
        )

    bad_profile = PROFILE.copy()
    bad_profile[0, 1] = 0
    with pytest.raises(ValueError, match='must be positive'):
        td.time_domain_site_response(
            bad_profile, _motion(100, 0.1), model='linear'
        )
