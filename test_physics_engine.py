import numpy as np
import pytest

from engine_router import EngineRouter, SafeExpression
from physics_engine import (
    AdvancedCalculusEngine,
    CalculusEngine,
    DifferentialEquationsEngine,
    ElectrostaticsEngine,
    ElectrodynamicsEngine,
    EngineCatalog,
    FluidsEngine,
    GeometryEngine,
    KinematicsEngine,
    LinearAlgebraEngine,
    MagneticsEngine,
    MechanicsEngine,
    RotationalMechanicsEngine,
    UniversalEngine,
    VectorCalculusEngine,
    WavesEngine,
    harmonic_oscillator_derivative,
    integrate_rk4,
    pendulum_derivative,
    simulate_system,
)


def test_rk4_tracks_harmonic_oscillator_and_exact_end_time():
    omega = 2.0

    def derivative(_time, state):
        position, velocity = state
        return np.array([velocity, -(omega**2) * position])

    times, states = integrate_rk4(derivative, [1.0, 0.0], 1.03, 0.01)
    expected = np.cos(omega * times[-1])

    assert times[-1] == pytest.approx(1.03)
    assert states[-1, 0] == pytest.approx(expected, abs=1e-8)


def test_nonlinear_pendulum_conserves_energy_at_large_angle():
    length = 2.0
    gravity = 9.81

    def derivative(_time, state):
        angle, angular_velocity = state
        return np.array([
            angular_velocity,
            -(gravity / length) * np.sin(angle),
        ])

    _times, states = integrate_rk4(derivative, [1.2, 0.0], 20.0, 1 / 120)
    energy = (0.5 * length**2 * states[:, 1]**2
              + gravity * length * (1 - np.cos(states[:, 0])))

    assert np.ptp(energy) < 1e-6


def test_harmonic_oscillator_factory_matches_true_solution():
    times, states = simulate_system(
        harmonic_oscillator_derivative(omega=2.0),
        initial_state=[1.0, 0.0],
        duration=1.03,
        dt=0.01,
    )
    expected = np.cos(2.0 * times[-1])

    assert times[-1] == pytest.approx(1.03)
    assert states[-1, 0] == pytest.approx(expected, abs=1e-8)


def test_pendulum_factory_tracks_large_angle_energy():
    length = 2.0
    gravity = 9.81

    times, states = simulate_system(
        pendulum_derivative(length=length, gravity=gravity),
        initial_state=[1.2, 0.0],
        duration=20.0,
        dt=1 / 120,
    )
    energy = (0.5 * length**2 * states[:, 1]**2
              + gravity * length * (1 - np.cos(states[:, 0])))

    assert np.ptp(energy) < 1e-6
    assert times[-1] == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("duration", "dt"),
    [(-1, 0.1), (1, 0), (float("inf"), 0.1)],
)
def test_rk4_rejects_invalid_time_steps(duration, dt):
    with pytest.raises(ValueError):
        integrate_rk4(lambda _time, state: state, [1.0], duration, dt)


def test_rk4_validates_each_derivative_result_before_next_stage():
    with pytest.raises(ValueError, match="matching state"):
        integrate_rk4(lambda _time, _state: np.array([1.0, 2.0]), [1.0], 0.1, 0.1)
    with pytest.raises(FloatingPointError, match="derivative"):
        integrate_rk4(lambda _time, _state: np.array([np.nan]), [1.0], 0.1, 0.1)


def test_calculus_engine_derivative_and_integral_are_consistent():
    values = np.linspace(0.0, 2.0, 101)
    derivative = CalculusEngine.derivative(np.sin, values)
    integral = CalculusEngine.integrate(np.sin, 0.0, 2.0, 1000)

    assert np.all(np.isfinite(derivative))
    assert integral == pytest.approx(1.0 - np.cos(2.0), rel=1e-3)


def test_geometry_and_linear_algebra_engines_work_on_vectors():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert GeometryEngine.distance(a, b) == pytest.approx(5.0)
    assert LinearAlgebraEngine.dot(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(11.0)
    assert LinearAlgebraEngine.matrix_vector_product(
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.array([3.0, 4.0]),
    ).tolist() == pytest.approx([11.0, 4.0])


def test_mechanics_and_kinematics_engines_model_motion():
    x0, v0 = 0.0, 10.0
    x_t = KinematicsEngine.constant_acceleration_position(x0, v0, 0.0, 2.0)
    assert x_t == pytest.approx(20.0)

    trajectory = MechanicsEngine.projectile_trajectory(10.0, 45.0, 0.1)
    assert len(trajectory) > 0
    assert trajectory[0][1] == pytest.approx(0.0)
    assert trajectory[-1][1] <= 0.0
    assert trajectory[-1][1] == pytest.approx(0.0)
    assert trajectory[-1][0] == pytest.approx(100.0 / 9.81, rel=1e-3)
    assert MechanicsEngine.projectile_trajectory(
        10.0, 45.0, 0.1, max_time=0.0,
    ) == [(0.0, 0.0)]


def test_zero_charge_at_observation_point_does_not_create_singularity():
    field = ElectrostaticsEngine.electric_field(
        positions=[[0.0, 0.0, 0.0]],
        charges=[0.0],
        point=[0.0, 0.0, 0.0],
    )
    assert field == pytest.approx([0.0, 0.0, 0.0])


def test_electrostatics_magnetics_waves_and_fluids_engines_produce_fields():
    E = ElectrostaticsEngine.electric_field(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        charges=np.array([1.0, -1.0]),
        point=np.array([0.5, 0.0, 0.0]),
    )
    assert E.shape == (3,)
    assert np.all(np.isfinite(E))

    B = MagneticsEngine.magnetic_field(
        wire_point=np.array([0.0, 0.0, 0.0]),
        current=1.0,
        observation=np.array([0.0, 1.0, 0.0]),
    )
    assert B.shape == (3,)
    assert np.all(np.isfinite(B))

    wave = WavesEngine.traveling_wave(amplitude=1.0, wave_number=2.0, angular_frequency=3.0, phase=0.0, x=0.5, t=0.25)
    assert wave == pytest.approx(np.sin(2.0 * 0.5 - 3.0 * 0.25))

    flow = FluidsEngine.velocity_field(x=1.0, y=2.0, t=0.5)
    assert flow.shape == (2,)
    assert np.all(np.isfinite(flow))
    with pytest.raises(ValueError, match="singular"):
        ElectrostaticsEngine.electric_field([[0.0, 0.0, 0.0]], [1.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="singular"):
        MagneticsEngine.magnetic_field([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0])


def test_rotational_mechanics_obeys_torque_and_energy_relations():
    inertia = RotationalMechanicsEngine.moment_of_inertia([2.0, 1.0], [1.0, 2.0])
    acceleration = RotationalMechanicsEngine.angular_acceleration(10.0, inertia)
    assert inertia == pytest.approx(6.0)
    assert acceleration == pytest.approx(10 / 6)
    assert RotationalMechanicsEngine.kinetic_energy(inertia, 2.0) == pytest.approx(12.0)

    times, states = RotationalMechanicsEngine.simulate_constant_torque(
        10.0, inertia, 0.0, 0.0, 1.0, 0.1,
    )
    assert times[-1] == pytest.approx(1.0)
    assert states[-1] == pytest.approx([5 / 6, 10 / 6])


def test_kinematics_and_universal_engine_reject_nonfinite_values():
    with pytest.raises(ValueError):
        KinematicsEngine.constant_acceleration_position(0.0, 0.0, float("nan"), 1.0)
    with pytest.raises(ValueError):
        UniversalEngine(default_dt=0.0)
    with pytest.raises(ValueError):
        UniversalEngine.execute(
            "kinematics",
            "constant_acceleration_position",
            {"x0": 0.0, "v0": 0.0, "a": float("nan"), "t": 1.0},
        )


def test_force_driven_mechanics_tracks_time_sampled_particle_states():
    times, states = MechanicsEngine.simulate_force_field(
        lambda _time, _position, _velocity: np.array([0.0, -2.0]),
        mass=2.0,
        initial_position=[0.0, 0.0],
        initial_velocity=[3.0, 4.0],
        duration=1.0,
        dt=0.01,
    )

    assert times[-1] == pytest.approx(1.0)
    assert states[-1, :2] == pytest.approx([3.0, 3.5], abs=1e-10)
    assert states[-1, 2:] == pytest.approx([3.0, 3.0], abs=1e-10)


def test_variable_torque_simulation_matches_constant_torque_solution():
    times, states = RotationalMechanicsEngine.simulate_torque(
        lambda _time, _angle, _omega: 4.0,
        moment_of_inertia=2.0,
        initial_angle=0.2,
        initial_angular_velocity=1.0,
        duration=2.0,
        dt=0.02,
    )

    assert times[-1] == pytest.approx(2.0)
    assert states[-1] == pytest.approx([6.2, 5.0], abs=1e-10)


def test_electrodynamics_lorentz_force_and_poynting_vector():
    force = ElectrodynamicsEngine.lorentz_force(
        2.0,
        [1.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 2.0],
    )
    assert force == pytest.approx([14.0, 0.0, 0.0])
    assert ElectrodynamicsEngine.poynting_vector(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        mu0=2.0,
    ) == pytest.approx([0.0, 0.0, 0.5])


def test_plane_wave_fields_and_charged_particle_are_frame_sampled():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]])
    times = np.array([0.0, 0.25])
    electric, magnetic = ElectrodynamicsEngine.plane_wave_fields(
        positions,
        times,
        electric_amplitude=[1.0, 0.0, 0.0],
        propagation_direction=[0.0, 0.0, 1.0],
        frequency=1.0,
        speed_of_light=1.0,
    )
    particle_times, particle_states = ElectrodynamicsEngine.simulate_charged_particle(
        charge=1.0,
        mass=2.0,
        electric_field_at=lambda _time, _position: np.array([2.0, 0.0, 0.0]),
        magnetic_field_at=lambda _time, _position: np.zeros(3),
        initial_position=[0.0, 0.0, 0.0],
        initial_velocity=[0.0, 0.0, 0.0],
        duration=1.0,
        dt=0.01,
    )

    assert electric.shape == magnetic.shape == (2, 2, 3)
    assert np.allclose(np.sum(electric * magnetic, axis=-1), 0.0)
    assert ElectrodynamicsEngine.poynting_vector(electric[1, 0], magnetic[1, 0])[2] > 0
    assert particle_times[-1] == pytest.approx(1.0)
    assert particle_states[-1, 0] == pytest.approx(0.5, abs=1e-10)
    assert particle_states[-1, 3] == pytest.approx(1.0, abs=1e-10)


def test_fluid_source_and_vortex_are_divergence_free_away_from_singularities():
    point = np.array([1.2, 0.8])
    divergence_vortex = VectorCalculusEngine.divergence(
        lambda p: FluidsEngine.vortex_flow(p[0], p[1], 2.0),
        point,
    )
    divergence_source = VectorCalculusEngine.divergence(
        lambda p: FluidsEngine.source_flow(p[0], p[1], 2.0),
        point,
    )
    assert divergence_vortex == pytest.approx(0.0, abs=1e-8)
    assert divergence_source == pytest.approx(0.0, abs=1e-8)
    assert FluidsEngine.continuity_residual(-2.0, 2.0) == pytest.approx(0.0)
    point = np.array([0.3, 1.1])
    flow = lambda p: FluidsEngine.velocity_field(p[0], p[1], 0.4)
    assert FluidsEngine.continuity_error(flow, point) == pytest.approx(0.0, abs=1e-8)


def test_wave_engine_superposition_and_phase_speed():
    x = np.linspace(-1.0, 1.0, 5)
    summed = WavesEngine.superpose(
        [(1.0, 2.0, 4.0, 0.0), (0.5, 2.0, 4.0, np.pi / 2)],
        x,
        0.25,
    )
    expected = np.sin(2 * x - 1) + 0.5 * np.sin(2 * x - 1 + np.pi / 2)
    assert summed == pytest.approx(expected)
    assert WavesEngine.phase_velocity(2.0, 6.0) == pytest.approx(3.0)
    assert WavesEngine.sample_wave(1.0, 2.0, 3.0, 0.0, [-1.0, 1.0], 11).shape == (11, 2)


def test_wave_space_time_samples_match_traveling_wave():
    x, times, field = WavesEngine.sample_spacetime(
        amplitude=2.0,
        wave_number=np.pi,
        angular_frequency=np.pi,
        phase=0.0,
        x_range=[0.0, 2.0],
        time_range=[0.0, 1.0],
        samples_x=9,
        samples_t=5,
    )

    assert field.shape == (5, 9)
    assert x[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    assert field[0, 2] == pytest.approx(2.0)
    assert field[4, 2] == pytest.approx(-2.0)


def test_fluid_streamlines_and_pathlines_track_distinct_flow_models():
    streamline = FluidsEngine.trace_streamline(
        lambda point: FluidsEngine.vortex_flow(point[0], point[1], 1.0),
        seed=[1.0, 0.0],
        step_size=0.01,
        steps=200,
    )
    times, pathline = FluidsEngine.trace_pathline(
        lambda _time, _point: np.array([2.0, -1.0]),
        initial_position=[0.0, 0.0],
        duration=1.0,
        dt=0.05,
    )

    assert streamline.shape[1] == 2
    assert np.linalg.norm(streamline[-1]) == pytest.approx(1.0, abs=1e-8)
    assert times[-1] == pytest.approx(1.0)
    assert pathline[-1] == pytest.approx([2.0, -1.0])


def test_domain_registry_exposes_every_engine_and_explicit_operations():
    names = EngineCatalog.available()
    required = {
        "calculus", "geometry", "linear_algebra", "kinematics", "mechanics",
        "electrostatics", "magnetics", "fluids", "waves",
    }
    assert required.issubset(set(names))
    assert EngineCatalog.resolve("mechanics") is MechanicsEngine
    assert EngineCatalog.resolve("waves") is WavesEngine
    assert EngineCatalog.resolve("calculus") is CalculusEngine
    assert "derivative" in UniversalEngine.operations()["calculus"]


def test_explicit_engine_dispatch_matches_direct_calculus_and_matrix_operations():
    derivative = UniversalEngine.execute(
        "calculus",
        "derivative",
        {"func": {"expression": "x**2"}, "x": 3.0},
    )
    eigenvalues = UniversalEngine.execute(
        "linear_algebra",
        "eigenvalues",
        {"matrix": [[2.0, 0.0], [0.0, 3.0]]},
    )

    assert derivative == pytest.approx(6.0)
    assert eigenvalues == pytest.approx([3.0, 2.0])


def test_engine_router_rejects_unknown_operations_and_unsafe_expressions():
    with pytest.raises(ValueError, match="unsupported operation"):
        EngineRouter.execute("calculus", "build_episode", {})
    for expression in ("__import__('os')", "x.real", "x + y", "x[0]"):
        with pytest.raises(ValueError):
            SafeExpression(expression)


def test_vector_calculus_and_differential_equation_engines_remain_available():
    grad = VectorCalculusEngine.gradient(
        lambda xy: xy[0] ** 2 + xy[1] ** 2,
        np.array([1.0, 2.0]),
    )
    decay = DifferentialEquationsEngine.exponential_decay(k=3.0, t=1.0)
    field = VectorCalculusEngine.divergence(
        lambda xy: np.array([xy[0], xy[1]]),
        np.array([1.0, 2.0]),
    )

    assert grad.tolist() == pytest.approx([2.0, 4.0])
    assert decay == pytest.approx(np.exp(-3.0))
    assert field == pytest.approx(2.0)
