from pathlib import Path
from sys import executable

import numpy as np
import pytest
from test_class_ground_motion import nearly_identical

from PySeismoSoil import class_simulation
from PySeismoSoil.class_curves import Multiple_GGmax_Damping_Curves
from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_parameters import (
    HH_Param_Multi_Layer,
    MKZ_Param_Multi_Layer,
)
from PySeismoSoil.class_simulation import (
    Equiv_Linear_Simulation,
    Linear_Simulation,
    Nonlinear_Simulation,
)
from PySeismoSoil.class_Vs_profile import Vs_Profile

f_dir = Path(__file__).resolve().parent / 'files'


def test_linear() -> None:
    input_motion = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='m')
    soil_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    ls = Linear_Simulation(soil_profile, input_motion)
    sim_result = ls.run(every_layer=True, show_fig=True)
    output = sim_result.accel_on_surface

    assert output.accel.shape == input_motion.accel.shape
    assert output.dt == input_motion.dt
    assert output.npts == input_motion.npts

    sim_result_ = ls.run(every_layer=False, show_fig=False)
    output_ = sim_result_.accel_on_surface

    # Check that two algorithms produce nearly identical results
    assert nearly_identical(output.accel, output_.accel, thres=0.99)


def test_equiv_linear() -> None:
    soil_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    input_motion = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='gal')
    curves = Multiple_GGmax_Damping_Curves(
        data=str(f_dir / 'curve_FKSH14.txt')
    )
    equiv_lin_sim = Equiv_Linear_Simulation(
        soil_profile,
        input_motion,
        curves,
        boundary='elastic',
    )
    output = equiv_lin_sim.run(show_fig=True)
    max_v = output.max_a_v_d[:, 2]

    # fmt: off
    max_v_benchmark = [
        0.404085, 0.403931, 0.402998, 0.399842, 0.390005,
        0.389336, 0.388176, 0.386563, 0.384449, 0.381578,
        0.377044, 0.371955, 0.366454, 0.36505, 0.363536,
        0.361912, 0.360214, 0.358405, 0.356464, 0.354368,
        0.352104, 0.349671, 0.347186, 0.344585, 0.342999,
        0.344762, 0.345623, 0.346219, 0.34607, 0.344739,
        0.342724, 0.339295, 0.334776, 0.32954, 0.324481,
        0.319054, 0.317311, 0.316842, 0.318159, 0.319668,
        0.321421, 0.323498, 0.325626, 0.326861, 0.328234,
        0.328466, 0.327704, 0.326466, 0.323216, 0.322324,
        0.320209, 0.316914, 0.312529, 0.308906, 0.304708,
        0.300202, 0.295505, 0.29226, 0.289536, 0.287653,
        0.287429, 0.290265, 0.292502,  # from MATLAB SeismoSoil
    ]
    # fmt: on

    # FFT of scipy and MATLAB are different, hence a lenient tolerance
    tol = 0.01
    assert np.allclose(max_v, max_v_benchmark, rtol=tol, atol=0.0)


# The error output of the nonlinear kernel (NLHH) when it cannot read a
# parameter file (for example, of MKZ parameters, which have 4 rows per layer,
# while NLHH reads 9 HH parameters per layer)
KERNEL_ERROR = (
    'forrtl: severe (24): end-of-file during read, unit 27, file HH_G.dat'
)


def fail_next_kernel_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make the next run of the nonlinear kernel fail (and only the next one).

    Instead of the kernel, a Python script runs with the same arguments of
    ``subprocess.run()``, so that its output is shown or captured in the same
    way. Like NLHH, it shows its progress on stdout and its error on stderr
    (``KERNEL_ERROR``), and exits with status 24.
    """
    run = class_simulation.subprocess.run
    script = (
        'import sys; '
        "print('|---------------- Calculating --------------------|'); "
        f'sys.stderr.write({KERNEL_ERROR!r}); '
        'sys.exit(24)'
    )

    def run_failing_kernel(_args: str, **kwargs: object) -> object:
        monkeypatch.setattr(class_simulation.subprocess, 'run', run)
        return run([executable, '-c', script], **kwargs)

    monkeypatch.setattr(class_simulation.subprocess, 'run', run_failing_kernel)


@pytest.mark.parametrize(
    ('MKZ_param_names', 'name_in_error'),
    [
        pytest.param(['G_param', 'xi_param'], 'G_param', id='both'),
        pytest.param(['G_param'], 'G_param', id='G_param'),
        pytest.param(['xi_param'], 'xi_param', id='xi_param'),
    ],
)
def test_nonlinear_init__MKZ_params_raise_TypeError(
        MKZ_param_names: list[str],
        name_in_error: str,
) -> None:
    input_motion = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='m')
    soil_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))  # 5 materials
    params = {
        'G_param': HH_Param_Multi_Layer(str(f_dir / 'HH_G_FKSH14.txt')),
        'xi_param': HH_Param_Multi_Layer(str(f_dir / 'HH_X_FKSH14.txt')),
    }
    for name in MKZ_param_names:
        params[name] = MKZ_Param_Multi_Layer([
            {'gamma_ref': 0.0005 * (j + 1), 's': 0.9, 'beta': 1.2, 'Gmax': 1.0}
            for j in range(5)
        ])

    # The nonlinear kernel would fail to read MKZ parameters in `run()`
    expected = (
        f'`{name_in_error}` must be an `HH_Param_Multi_Layer` object, not'
        ' `MKZ_Param_Multi_Layer`: the nonlinear kernel supports only HH'
        ' parameters.'
    )
    with pytest.raises(TypeError, match=expected):
        Nonlinear_Simulation(soil_profile, input_motion, **params)


@pytest.mark.parametrize('verbose', [True, False])
def test_nonlinear_run__kernel_failure_raises_ValueError(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
        *,
        verbose: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    accel_data = np.genfromtxt(f_dir / 'sample_accel.txt')
    input_motion = Ground_Motion(accel_data[::50], unit='gal')
    soil_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    sim = Nonlinear_Simulation(
        soil_profile,
        input_motion,
        G_param=HH_Param_Multi_Layer(str(f_dir / 'HH_G_FKSH14.txt')),
        xi_param=HH_Param_Multi_Layer(str(f_dir / 'HH_X_FKSH14.txt')),
    )
    fail_next_kernel_run(monkeypatch)
    with pytest.raises(ValueError, match='failed with exit status 24') as e:
        sim.run(sim_dir='nonlinear_sim', verbose=verbose)

    # The error has the error output of the kernel (instead of an unrelated
    # error from reading its empty output files)
    assert str(e.value).endswith(
        f'Error output of the kernel:\n{KERNEL_ERROR}'
    )

    # When `verbose`, the progress of the kernel is still shown as it runs
    if verbose:
        assert 'Calculating' in capfd.readouterr().out

    # The working directory is restored, and the files of the simulation (such
    # as the input files of the kernel) are kept
    assert Path.cwd() == tmp_path
    assert (tmp_path / 'nonlinear_sim' / 'HH_G.dat').is_file()


def test_nonlinear_init() -> None:
    input_motion = Ground_Motion(str(f_dir / 'sample_accel.txt'), unit='m')
    soil_profile = Vs_Profile(str(f_dir / 'profile_FKSH14.txt'))
    HH_G = HH_Param_Multi_Layer(str(f_dir / 'HH_G_FKSH14.txt'))
    HH_x = HH_Param_Multi_Layer(str(f_dir / 'HH_X_FKSH14.txt'))

    # this should succeed
    Nonlinear_Simulation(
        soil_profile,
        input_motion,
        G_param=HH_G,
        xi_param=HH_x,
        boundary='elastic',
    )

    # this should fail with ValueError
    HH_G_data = HH_G.param_list
    HH_x_data = HH_x.param_list
    HH_G_ = HH_Param_Multi_Layer(HH_G_data[:-1])  # exclude one layer
    HH_x_ = HH_Param_Multi_Layer(HH_x_data[:-1])  # exclude one layer
    with pytest.raises(ValueError, match='Not enough sets of parameters'):
        Nonlinear_Simulation(
            soil_profile,
            input_motion,
            G_param=HH_G_,
            xi_param=HH_x,
        )

    with pytest.raises(ValueError, match='Not enough sets of parameters'):
        Nonlinear_Simulation(
            soil_profile,
            input_motion,
            G_param=HH_G,
            xi_param=HH_x_,
        )
