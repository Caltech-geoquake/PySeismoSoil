import pathlib

import numpy as np
import pytest

from PySeismoSoil.gui import analyses as an
from PySeismoSoil.gui import data_io as io

f_dir = pathlib.Path(__file__).parent / 'files'
MOTION_NAME = 'Accel_02_weak_motion_(unit_gal).txt'


@pytest.mark.parametrize(
    ('bedrock', 'motion_type', 'time_domain', 'expected'),
    [
        ('elastic', 'incident', False, ('elastic', 1, 2, 2, 'ok')),
        ('elastic', 'outcrop', False, ('elastic', 2, 2, 2, 'caution')),
        ('elastic', 'borehole', False, ('elastic', 2, 2, 2, 'error')),
        ('rigid', 'incident', False, ('rigid', 1, 2, 2, 'ok')),
        ('rigid', 'borehole', False, ('rigid', 2, 2, 2, 'ok')),
        ('rigid', 'outcrop', False, ('rigid', 2, 2, 2, 'ok')),
        ('elastic', 'incident', True, ('elastic', 1, 2, 2, 'ok')),
        ('rigid', 'incident', True, ('rigid', 1, 2, 2, 'ok')),
        ('rigid', 'borehole', True, ('borehole', 1, 1, 1, 'ok')),
        ('rigid', 'outcrop', True, ('borehole', 1, 1, 1, 'ok')),
    ],
)
def test_resolve_boundary(
        *, bedrock: str, motion_type: str, time_domain: bool, expected: tuple
) -> None:
    setting = an.resolve_boundary(
        bedrock, motion_type, time_domain=time_domain
    )
    assert (
        setting.boundary,
        setting.division,
        setting.outcrop_factor,
        setting.incident_factor,
        setting.status,
    ) == expected


def test_resolve_boundary__invalid() -> None:
    with pytest.raises(ValueError, match='bedrock'):
        an.resolve_boundary('soft', 'incident', time_domain=False)

    with pytest.raises(ValueError, match='motion type'):
        an.resolve_boundary('rigid', 'surface', time_domain=False)


def _job(method: str, out_dir: pathlib.Path) -> an.AnalysisJob:
    profile = io.load_matrix(f_dir / 'profile_FKSH14.txt')
    kwargs = {}
    if method in {'eql', 'eql_fd', 'nl_h2', 'nl_h4', 'nl_hh'}:
        kwargs['curve'] = io.load_matrix(f_dir / 'curve_FKSH14.txt')

    if method == 'nl_hh':
        kwargs['G_param'] = io.load_matrix(f_dir / 'HH_G_FKSH14.txt')
        kwargs['xi_param'] = io.load_matrix(f_dir / 'HH_X_FKSH14.txt')
    elif method in {'nl_h2', 'nl_h4'}:
        kwargs['G_param'] = io.load_matrix(f_dir / 'H4_G_IWTH04.txt')[:, :5]
        kwargs['xi_param'] = io.load_matrix(f_dir / 'H4_x_IWTH04.txt')[:, :5]
    elif method == 'nl_epp':
        kwargs['tau_max'] = an.shear_strength_from_profile(profile)

    setting = an.resolve_boundary(
        'elastic', 'incident', time_domain=an.METHODS[method].time_domain
    )
    return an.AnalysisJob(
        method=method,
        vs_profile=profile,
        setting=setting,
        output_dir=out_dir,
        accel_factor=io.ACCEL_UNITS['gal'],
        **kwargs,
    )


@pytest.mark.parametrize('method', list(an.METHODS))
def test_run_motion(method: str, tmp_path: pathlib.Path) -> None:
    motion = io.load_matrix(f_dir / MOTION_NAME)[:1500]
    job = _job(method, tmp_path)
    figures = an.run_motion(job, motion, MOTION_NAME, verbose=False)
    assert len(figures) == 3
    stem = MOTION_NAME[:-4]
    out = tmp_path / stem
    tf_name = an.METHODS[method].tf_name
    expected = [
        'accel_on_surface.txt',
        're-discretized_profile.txt',
        'time_history_accel.txt',
        'time_history_veloc.txt',
        'time_history_displ.txt',
        'time_history_strain.txt',
        'time_history_stress.txt',
        'max_a_v_d.txt',
        'max_gamma_tau.txt',
        f'{tf_name}_TF.png',
        'accelerations.png',
        'max_a_v_d_gamma_tau.png',
    ]
    if an.METHODS[method].time_domain:
        expected.append(f'{tf_name}_TF_raw.txt')
        if method != 'linear_td':
            expected.append(f'{tf_name}_TF_smoothed.txt')
    else:
        expected.append(f'{tf_name}_TF.txt')

    for name in expected:
        assert (out / f'{stem}_{name}').exists(), name

    surface = np.genfromtxt(out / f'{stem}_accel_on_surface.txt')
    assert surface.shape[1] == 2
    assert np.all(np.isfinite(surface))
    # The motion is weak, so the peak surface acceleration is similar to
    # that of the linear (frequency-domain) analysis
    ref_dir = tmp_path / 'ref'
    an.run_motion(_job('linear_fd', ref_dir), motion, MOTION_NAME)
    ref = np.genfromtxt(ref_dir / stem / f'{stem}_accel_on_surface.txt')
    assert np.isclose(
        np.max(np.abs(surface[:, 1])), np.max(np.abs(ref[:, 1])), rtol=0.25
    )


def test_run_motion__units_and_motion_types(tmp_path: pathlib.Path) -> None:
    # The same motion, in m/s/s as an outcrop motion, or in g as an incident
    # motion (half the amplitude), gives the same results
    motion = io.load_matrix(f_dir / MOTION_NAME)[:1000]
    motion[:, 1] /= 100  # gal --> m/s/s
    job1 = _job('linear_fd', tmp_path / '1')
    job1.accel_factor = 1.0
    job1.setting = an.resolve_boundary('elastic', 'outcrop', time_domain=False)
    incident = motion.copy()
    incident[:, 1] /= 2 * 9.81
    job2 = _job('linear_fd', tmp_path / '2')
    job2.accel_factor = io.ACCEL_UNITS['g']
    an.run_motion(job1, motion, 'a.txt')
    an.run_motion(job2, incident, 'a.txt')
    out1 = np.genfromtxt(tmp_path / '1' / 'a' / 'a_accel_on_surface.txt')
    out2 = np.genfromtxt(tmp_path / '2' / 'a' / 'a_accel_on_surface.txt')
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-9)


def test_transfer_function_and_konno_ohmachi() -> None:
    t = np.arange(1000) * 0.01
    accel = np.sin(2 * np.pi * 2 * t) * np.exp(-t)
    freq, tf = an.transfer_function(accel, 3 * accel, dt=0.01)
    assert freq[0] > 0
    assert freq[-1] <= 30
    assert np.allclose(tf, 3)
    smoothed = an.konno_ohmachi(np.full(len(freq), 2.0), freq)
    assert np.allclose(smoothed, 2.0)


def test_masing_damping() -> None:
    strain = np.geomspace(1e-6, 1e-1, 50)
    assert np.allclose(an.masing_damping(strain, np.ones_like(strain)), 0)
    damping = an.masing_damping(strain, 1 / (1 + strain / 1e-3))
    assert np.all(np.diff(damping) > 0)
    assert 0 < damping[-1] < 2 / np.pi


def test_darendeli_curves_and_fitting() -> None:
    profile = io.load_matrix(f_dir / 'profile_FKSH14.txt')
    curve = an.darendeli_curves(profile, PI=20)
    io.check_data(curve, 'curve')
    assert curve.shape == (len(an.DARENDELI_STRAIN_PCT), 4 * 5)
    assert np.all(np.diff(curve[:, 1]) < 0)  # G/Gmax decreases
    below_1_pct = curve[:, 0] <= 1
    assert np.all(np.diff(curve[below_1_pct, 3]) > 0)  # damping increases
    H2, figures = an.fit_H2(curve[:, :8])
    io.check_data(H2, 'mkz_param')
    assert len(figures) == 2
    H4_G, _ = an.fit_H4_G(curve[:, :8])
    io.check_data(H4_G, 'mkz_param')
    H4_x, _ = an.fit_damping_params(curve[:, :4], 'H4', pop_size=20, n_gen=5)
    io.check_data(H4_x, 'mkz_param')
    HH_x, _ = an.fit_damping_params(curve[:, :4], 'HH', pop_size=20, n_gen=5)
    io.check_data(HH_x, 'hh_param')


def test_fit_HH_G(tmp_path: pathlib.Path) -> None:
    profile = io.load_matrix(f_dir / 'profile_FKSH14.txt')
    params = an.fit_HH_G(profile)
    io.check_data(params, 'hh_param')
    assert params.shape == (9, 5)
    curve = io.load_matrix(f_dir / 'curve_FKSH14.txt')
    params = an.fit_HH_G(
        profile,
        curve=curve,
        tau_max=an.shear_strength_from_profile(profile),
        fig_output_dir=str(tmp_path),
        profile_name='FKSH14',
    )
    assert params.shape == (9, 5)
    assert list(tmp_path.glob('*.png'))
    with pytest.raises(io.InputError, match='every soil layer'):
        an.fit_HH_G(profile, curve=curve[:, :8])


def test_shear_strength_from_profile() -> None:
    profile = io.load_matrix(f_dir / 'profile_FKSH14.txt')
    su = an.shear_strength_from_profile(profile)
    assert su.shape == (5,)
    assert np.all(np.diff(su) > 0)
    # Close to Tmax in the HH_G file (calculated the same way)
    tmax = io.load_matrix(f_dir / 'HH_G_FKSH14.txt')[7]
    assert np.allclose(su, tmax, rtol=0.1)
