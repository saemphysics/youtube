"""Deterministic numerical tools for physics-driven animations.

The engine exposes a compact but production-friendly API for driving cinematic
simulations: a fixed-step RK4 integrator, harmonic-oscillator dynamics, and a
nonlinear pendulum model. These are intentionally generic and deterministic so
that the same state equations can be used both in numerical validation and in
rendered scenes. Mathematics domains live in ``math_engine`` and are re-exported
here to preserve the original public imports.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from math_engine import (
    AdvancedCalculusEngine,
    CalculusEngine,
    DifferentialEquationsEngine,
    GeometryEngine,
    LinearAlgebraEngine,
    VectorCalculusEngine,
)

__all__ = [
    "integrate_rk4",
    "harmonic_oscillator_derivative",
    "pendulum_derivative",
    "simulate_system",
    "AdvancedCalculusEngine",
    "CalculusEngine",
    "GeometryEngine",
    "LinearAlgebraEngine",
    "KinematicsEngine",
    "KineticsEngine",
    "MechanicsEngine",
    "RotationalMechanicsEngine",
    "ElectrostaticsEngine",
    "MagneticsEngine",
    "ElectrodynamicsEngine",
    "FluidsEngine",
    "WavesEngine",
    "VectorCalculusEngine",
    "DifferentialEquationsEngine",
    "EngineCatalog",
    "UniversalEngine",
]


def _validate_state_vector(initial_state: np.ndarray | list[float]) -> np.ndarray:
    """Ensure the state is a finite, non-empty 1D vector."""
    state = np.asarray(initial_state, dtype=np.float64)
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be a finite, non-empty vector")
    return state


def integrate_rk4(
    derivative: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray | list[float],
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a first-order state equation with fixed-step RK4.

    The final step is shortened when needed so the returned trajectory ends
    exactly at ``duration``. State vectors and derivatives use float64.
    """
    duration = float(duration)
    dt = float(dt)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and non-negative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    state = _validate_state_vector(initial_state)

    times = [0.0]
    states = [state.copy()]
    time = 0.0

    def evaluate_derivative(at_time: float, at_state: np.ndarray) -> np.ndarray:
        value = np.asarray(derivative(at_time, at_state), dtype=np.float64)
        if value.shape != at_state.shape:
            raise ValueError("derivative must return a vector matching state")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("derivative produced a non-finite value")
        return value

    while time < duration:
        step = min(dt, duration - time)
        k1 = evaluate_derivative(time, state)
        k2 = evaluate_derivative(time + step / 2, state + step * k1 / 2)
        k3 = evaluate_derivative(time + step / 2, state + step * k2 / 2)
        k4 = evaluate_derivative(time + step, state + step * k3)

        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("integration produced a non-finite state")

        time += step
        times.append(time)
        states.append(state.copy())

    return np.asarray(times), np.asarray(states)


def harmonic_oscillator_derivative(
    omega: float = 1.0,
    damping: float = 0.0,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return a harmonic oscillator derivative for [position, velocity]."""
    omega = float(omega)
    damping = float(damping)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be a finite positive number")
    if not np.isfinite(damping):
        raise ValueError("damping must be finite")

    def derivative(_time: float, state: np.ndarray | list[float]) -> np.ndarray:
        vec = _validate_state_vector(state)
        if vec.size != 2:
            raise ValueError("harmonic oscillator state must be length 2")
        position, velocity = vec
        return np.array([
            velocity,
            -(omega ** 2) * position - damping * velocity,
        ], dtype=np.float64)

    return derivative


def pendulum_derivative(
    length: float = 1.0,
    gravity: float = 9.81,
    damping: float = 0.0,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return a pendulum derivative for [angle, angular_velocity]."""
    length = float(length)
    gravity = float(gravity)
    damping = float(damping)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be a finite positive number")
    if not np.isfinite(gravity):
        raise ValueError("gravity must be finite")
    if not np.isfinite(damping):
        raise ValueError("damping must be finite")

    def derivative(_time: float, state: np.ndarray | list[float]) -> np.ndarray:
        vec = _validate_state_vector(state)
        if vec.size != 2:
            raise ValueError("pendulum state must be length 2")
        angle, angular_velocity = vec
        return np.array([
            angular_velocity,
            -(gravity / length) * np.sin(angle) - damping * angular_velocity,
        ], dtype=np.float64)

    return derivative


def simulate_system(
    derivative: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray | list[float],
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around ``integrate_rk4`` for scene-driven animation."""
    return integrate_rk4(derivative, initial_state, duration, dt)


class KinematicsEngine:
    """1D/2D kinematics for cinematic motion studies."""

    @staticmethod
    def constant_acceleration_position(x0: float, v0: float, a: float, t: float) -> float:
        x0, v0, a, t = map(float, (x0, v0, a, t))
        if not all(np.isfinite(value) for value in (x0, v0, a, t)):
            raise ValueError("kinematics parameters must be finite")
        return float(x0 + v0 * t + 0.5 * a * t ** 2)

    @staticmethod
    def constant_acceleration_velocity(v0: float, a: float, t: float) -> float:
        v0, a, t = map(float, (v0, a, t))
        if not all(np.isfinite(value) for value in (v0, a, t)):
            raise ValueError("kinematics parameters must be finite")
        return float(v0 + a * t)


KineticsEngine = KinematicsEngine


class MechanicsEngine:
    """Classical mechanics models with analytic, physically consistent states."""

    @staticmethod
    def projectile_trajectory(speed: float, angle_deg: float, dt: float, gravity: float = 9.81, max_time: float = 10.0):
        speed, angle_deg, dt, gravity, max_time = map(
            float, (speed, angle_deg, dt, gravity, max_time)
        )
        if not all(np.isfinite(value) for value in (speed, angle_deg, dt, gravity, max_time)):
            raise ValueError("projectile parameters must be finite")
        if speed < 0 or gravity <= 0 or dt <= 0 or max_time < 0:
            raise ValueError(
                "speed must be non-negative; gravity and dt must be positive; "
                "max_time must be non-negative"
            )
        theta = np.deg2rad(angle_deg)
        vx, vy = speed * np.cos(theta), speed * np.sin(theta)
        flight_time = max(0.0, 2.0 * vy / gravity)
        end_time = min(flight_time, max_time)
        times = np.arange(0.0, end_time, dt)
        times = np.append(times, end_time)
        trajectory = [
            (float(vx * t), float(max(0.0, vy * t - 0.5 * gravity * t**2)))
            for t in times
        ]
        return trajectory

    @staticmethod
    def spring_force(k: float, x: float) -> float:
        k, x = float(k), float(x)
        if not np.isfinite(k) or k < 0 or not np.isfinite(x):
            raise ValueError("spring constant must be finite and non-negative and displacement finite")
        return float(-k * x)

    @staticmethod
    def kinetic_energy(mass: float, velocity) -> float:
        mass = float(mass)
        v = np.asarray(velocity, dtype=np.float64)
        if not np.isfinite(mass) or mass <= 0 or v.size == 0 or not np.all(np.isfinite(v)):
            raise ValueError("mass must be positive and velocity finite")
        return float(0.5 * mass * np.dot(v.ravel(), v.ravel()))

    @staticmethod
    def momentum(mass: float, velocity) -> np.ndarray:
        mass = float(mass)
        v = np.asarray(velocity, dtype=np.float64)
        if not np.isfinite(mass) or mass <= 0 or v.ndim != 1 or not np.all(np.isfinite(v)):
            raise ValueError("mass must be positive and velocity a finite vector")
        return mass * v

    @staticmethod
    def newtonian_acceleration(force, mass: float) -> np.ndarray:
        force_array = np.asarray(force, dtype=np.float64)
        mass = float(mass)
        if force_array.size == 0 or not np.all(np.isfinite(force_array)):
            raise ValueError("force must contain finite values")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be finite and positive")
        return force_array / mass

    @staticmethod
    def simulate_force_field(
        force_at: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        mass: float,
        initial_position,
        initial_velocity,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate a particle under F(time, position, velocity), returning [x, v]."""
        mass = float(mass)
        position = _validate_state_vector(initial_position)
        velocity = _validate_state_vector(initial_velocity)
        if position.shape != velocity.shape:
            raise ValueError("initial position and velocity must have matching dimensions")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be finite and positive")

        dimension = position.size

        def derivative(time: float, state: np.ndarray) -> np.ndarray:
            current_position = state[:dimension]
            current_velocity = state[dimension:]
            force = np.asarray(
                force_at(time, current_position.copy(), current_velocity.copy()),
                dtype=np.float64,
            )
            if force.shape != (dimension,) or not np.all(np.isfinite(force)):
                raise ValueError("force must return a finite vector matching position")
            return np.concatenate((current_velocity, force / mass))

        times, states = integrate_rk4(
            derivative,
            np.concatenate((position, velocity)),
            duration,
            dt,
        )
        return times, states


class RotationalMechanicsEngine:
    """Rigid-body rotational dynamics about a fixed axis."""

    @staticmethod
    def moment_of_inertia(masses, radii) -> float:
        mass_values = np.asarray(masses, dtype=np.float64)
        radius_values = np.asarray(radii, dtype=np.float64)
        if mass_values.ndim != 1 or radius_values.shape != mass_values.shape:
            raise ValueError("masses and radii must be matching one-dimensional arrays")
        if (
            mass_values.size == 0
            or not np.all(np.isfinite(mass_values))
            or not np.all(np.isfinite(radius_values))
            or np.any(mass_values < 0)
        ):
            raise ValueError("masses and radii must be finite; masses cannot be negative")
        return float(np.dot(mass_values, radius_values**2))

    @staticmethod
    def angular_acceleration(torque: float, moment_of_inertia: float) -> float:
        torque, moment_of_inertia = float(torque), float(moment_of_inertia)
        if not np.isfinite(torque) or not np.isfinite(moment_of_inertia) or moment_of_inertia <= 0:
            raise ValueError("torque must be finite and moment of inertia positive")
        return torque / moment_of_inertia

    @staticmethod
    def kinetic_energy(moment_of_inertia: float, angular_velocity: float) -> float:
        inertia, omega = float(moment_of_inertia), float(angular_velocity)
        if not np.isfinite(inertia) or inertia < 0 or not np.isfinite(omega):
            raise ValueError("moment of inertia and angular velocity must be finite")
        return 0.5 * inertia * omega**2

    @staticmethod
    def simulate_constant_torque(
        torque: float,
        moment_of_inertia: float,
        initial_angle: float,
        initial_angular_velocity: float,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        torque, inertia, angle, omega, duration, dt = map(
            float, (torque, moment_of_inertia, initial_angle, initial_angular_velocity, duration, dt)
        )
        if not all(np.isfinite(value) for value in (torque, inertia, angle, omega, duration, dt)):
            raise ValueError("rotational simulation parameters must be finite")
        if inertia <= 0 or duration < 0 or dt <= 0:
            raise ValueError("moment of inertia and dt must be positive; duration cannot be negative")
        times = np.arange(0.0, duration, dt)
        times = np.append(times, duration)
        alpha = torque / inertia
        states = np.column_stack((angle + omega * times + 0.5 * alpha * times**2, omega + alpha * times))
        return times, states

    @staticmethod
    def simulate_torque(
        torque_at: Callable[[float, float, float], float],
        moment_of_inertia: float,
        initial_angle: float,
        initial_angular_velocity: float,
        duration: float,
        dt: float,
        *,
        damping: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate angle and angular velocity under a time-varying torque."""
        inertia, angle, omega, damping, duration, dt = map(
            float,
            (moment_of_inertia, initial_angle, initial_angular_velocity, damping, duration, dt),
        )
        if not all(np.isfinite(value) for value in (inertia, angle, omega, damping)):
            raise ValueError("rotational parameters must be finite")
        if inertia <= 0 or damping < 0:
            raise ValueError("moment of inertia must be positive and damping non-negative")

        def derivative(time: float, state: np.ndarray) -> np.ndarray:
            torque = float(torque_at(time, state[0], state[1]))
            if not np.isfinite(torque):
                raise ValueError("torque function must return a finite value")
            return np.array([state[1], (torque - damping * state[1]) / inertia])

        return integrate_rk4(derivative, [angle, omega], duration, dt)


class ElectrostaticsEngine:
    """Coulomb-law electric field computations for charge diagrams."""

    @staticmethod
    def electric_field(positions, charges, point, k: float = 8.9875517923e9):
        p = np.asarray(point, dtype=np.float64)
        pos = np.asarray(positions, dtype=np.float64)
        q = np.asarray(charges, dtype=np.float64)
        k = float(k)
        if pos.ndim != 2 or pos.shape[1:] != (3,) or p.shape != (3,):
            raise ValueError("positions and point must be 3D vectors")
        if q.ndim != 1 or pos.shape[0] != q.shape[0]:
            raise ValueError("charges and positions must have matching lengths")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            raise ValueError("positions, charges, and point must be finite")
        if not np.isfinite(k) or k <= 0:
            raise ValueError("Coulomb constant must be finite and positive")

        field = np.zeros(3, dtype=np.float64)
        for charge_position, charge in zip(pos, q):
            if charge == 0.0:
                continue
            delta = p - charge_position
            radius_sq = np.dot(delta, delta)
            if radius_sq < 1e-12:
                raise ValueError("electric field is singular at a point charge")
            field += k * charge * delta / (radius_sq ** 1.5)
        return field


class MagneticsEngine:
    """Simple magnetostatics approximations for wire and loop visualizations."""

    @staticmethod
    def magnetic_field(wire_point, current: float, observation, mu0: float = 4e-7 * np.pi):
        p = np.asarray(wire_point, dtype=np.float64)
        o = np.asarray(observation, dtype=np.float64)
        current, mu0 = float(current), float(mu0)
        if p.shape != (3,) or o.shape != (3,):
            raise ValueError("wire_point and observation must be 3D vectors")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(o)):
            raise ValueError("wire_point and observation must be finite")
        if not np.isfinite(current) or not np.isfinite(mu0) or mu0 <= 0:
            raise ValueError("current must be finite and permeability positive")
        delta = o - p
        radius_sq = np.dot(delta, delta)
        if radius_sq < 1e-12:
            raise ValueError("magnetic field is singular on the ideal wire")
        direction = np.array([0.0, 0.0, 1.0])
        field = mu0 * float(current) * np.cross(direction, delta) / (2.0 * np.pi * radius_sq)
        return field


class ElectrodynamicsEngine:
    """Electric, magnetic, and Lorentz-force calculations for EM scenes."""

    @staticmethod
    def lorentz_force(charge: float, electric_field, velocity, magnetic_field) -> np.ndarray:
        charge = float(charge)
        electric = np.asarray(electric_field, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        magnetic = np.asarray(magnetic_field, dtype=np.float64)
        if any(vector.shape != (3,) for vector in (electric, velocity, magnetic)):
            raise ValueError("electric field, velocity, and magnetic field must be 3D vectors")
        if not np.isfinite(charge) or any(
            not np.all(np.isfinite(vector)) for vector in (electric, velocity, magnetic)
        ):
            raise ValueError("charge and field vectors must be finite")
        return charge * (electric + np.cross(velocity, magnetic))

    @staticmethod
    def poynting_vector(electric_field, magnetic_field, *, mu0: float = 4e-7 * np.pi) -> np.ndarray:
        electric = np.asarray(electric_field, dtype=np.float64)
        magnetic = np.asarray(magnetic_field, dtype=np.float64)
        mu0 = float(mu0)
        if electric.shape != (3,) or magnetic.shape != (3,):
            raise ValueError("electric and magnetic fields must be 3D vectors")
        if not np.all(np.isfinite(electric)) or not np.all(np.isfinite(magnetic)):
            raise ValueError("field vectors must be finite")
        if not np.isfinite(mu0) or mu0 <= 0:
            raise ValueError("mu0 must be finite and positive")
        return np.cross(electric, magnetic) / mu0

    @staticmethod
    def plane_wave_fields(
        positions,
        times,
        electric_amplitude,
        propagation_direction,
        frequency: float,
        *,
        phase: float = 0.0,
        speed_of_light: float = 299_792_458.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample transverse E and B fields for a monochromatic plane wave.

        Returned arrays have shape (time, position, 3); the magnetic field
        obeys B = k_hat x E / c, so the fields and energy flux share direction.
        """
        points = np.asarray(positions, dtype=np.float64)
        time_values = np.asarray(times, dtype=np.float64)
        amplitude = np.asarray(electric_amplitude, dtype=np.float64)
        direction = np.asarray(propagation_direction, dtype=np.float64)
        frequency, phase, speed_of_light = map(float, (frequency, phase, speed_of_light))
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("positions must be a non-empty array of 3D points")
        if time_values.ndim != 1 or time_values.size == 0:
            raise ValueError("times must be a non-empty vector")
        if amplitude.shape != (3,) or direction.shape != (3,):
            raise ValueError("electric amplitude and propagation direction must be 3D vectors")
        if not all(np.all(np.isfinite(value)) for value in (points, time_values, amplitude, direction)):
            raise ValueError("wave positions, times, and vectors must be finite")
        if not np.isfinite(frequency) or frequency < 0 or not np.isfinite(phase):
            raise ValueError("frequency must be finite and non-negative and phase finite")
        if not np.isfinite(speed_of_light) or speed_of_light <= 0:
            raise ValueError("speed of light must be finite and positive")
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0.0:
            raise ValueError("propagation direction must be non-zero")
        unit_direction = direction / direction_norm
        if abs(float(np.dot(amplitude, unit_direction))) > 1e-10 * max(1.0, np.linalg.norm(amplitude)):
            raise ValueError("electric amplitude must be perpendicular to propagation direction")

        angular_frequency = 2.0 * np.pi * frequency
        wave_number = angular_frequency / speed_of_light
        argument = (
            angular_frequency * time_values[:, None]
            - wave_number * (points @ unit_direction)[None, :]
            + phase
        )
        electric = np.sin(argument)[:, :, None] * amplitude[None, None, :]
        magnetic = np.cross(unit_direction, electric) / speed_of_light
        return electric, magnetic

    @staticmethod
    def simulate_charged_particle(
        charge: float,
        mass: float,
        electric_field_at: Callable[[float, np.ndarray], np.ndarray],
        magnetic_field_at: Callable[[float, np.ndarray], np.ndarray],
        initial_position,
        initial_velocity,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        charge, mass = float(charge), float(mass)
        if not np.isfinite(charge) or not np.isfinite(mass) or mass <= 0:
            raise ValueError("charge must be finite and mass finite and positive")

        def force_at(time: float, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
            electric = np.asarray(electric_field_at(time, position), dtype=np.float64)
            magnetic = np.asarray(magnetic_field_at(time, position), dtype=np.float64)
            if electric.shape != position.shape or magnetic.shape != position.shape:
                raise ValueError("electric and magnetic fields must match particle dimensions")
            if not np.all(np.isfinite(electric)) or not np.all(np.isfinite(magnetic)):
                raise ValueError("electric and magnetic fields must be finite")
            if position.size != 3:
                raise ValueError("charged-particle electrodynamics requires 3D states")
            return ElectrodynamicsEngine.lorentz_force(charge, electric, velocity, magnetic)

        return MechanicsEngine.simulate_force_field(
            force_at, mass, initial_position, initial_velocity, duration, dt,
        )


class FluidsEngine:
    """Analytic 2D incompressible flow fields suitable for vector-field scenes."""

    @staticmethod
    def _validate_point(x, y, core_radius=1e-6) -> tuple[float, float, float]:
        x, y, core_radius = float(x), float(y), float(core_radius)
        if not all(np.isfinite(value) for value in (x, y, core_radius)) or core_radius <= 0:
            raise ValueError("coordinates must be finite and core_radius positive")
        return x, y, core_radius

    @staticmethod
    def uniform_flow(x, y, speed: float = 1.0, angle: float = 0.0) -> np.ndarray:
        x, y = float(x), float(y)
        speed, angle = float(speed), float(angle)
        if not all(np.isfinite(value) for value in (x, y, speed, angle)):
            raise ValueError("flow parameters must be finite")
        return speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

    @classmethod
    def source_flow(cls, x, y, strength: float = 1.0, *, core_radius: float = 1e-6) -> np.ndarray:
        x, y, core = cls._validate_point(x, y, core_radius)
        strength = float(strength)
        if not np.isfinite(strength):
            raise ValueError("source strength must be finite")
        radius_sq = x * x + y * y + core * core
        return strength / (2.0 * np.pi * radius_sq) * np.array([x, y])

    @classmethod
    def vortex_flow(cls, x, y, circulation: float = 1.0, *, core_radius: float = 1e-6) -> np.ndarray:
        x, y, core = cls._validate_point(x, y, core_radius)
        circulation = float(circulation)
        if not np.isfinite(circulation):
            raise ValueError("circulation must be finite")
        radius_sq = x * x + y * y + core * core
        return circulation / (2.0 * np.pi * radius_sq) * np.array([-y, x])

    @staticmethod
    def velocity_field(x, y, t=0.0):
        x_val, y_val, t_val = float(x), float(y), float(t)
        if not all(np.isfinite(value) for value in (x_val, y_val, t_val)):
            raise ValueError("flow coordinates and time must be finite")
        phase_x = x_val + t_val
        phase_y = y_val - t_val
        return np.array([
            np.sin(phase_x) * np.cos(phase_y),
            -np.cos(phase_x) * np.sin(phase_y),
        ], dtype=np.float64)

    @staticmethod
    def continuity_error(vector_field, point, density_rate: float = 0.0, *, h: float = 1e-6) -> float:
        residual = FluidsEngine.continuity_residual(
            VectorCalculusEngine.divergence(vector_field, point, h=h),
            density_rate,
        )
        return abs(residual)

    @staticmethod
    def continuity_residual(divergence: float, density_rate: float = 0.0) -> float:
        divergence, density_rate = float(divergence), float(density_rate)
        if not np.isfinite(divergence) or not np.isfinite(density_rate):
            raise ValueError("continuity terms must be finite")
        return divergence + density_rate

    @staticmethod
    def trace_streamline(
        vector_field: Callable[[np.ndarray], np.ndarray],
        seed,
        step_size: float = 0.03,
        steps: int = 400,
        *,
        bounds=None,
    ) -> np.ndarray:
        """Trace a steady 2D streamline using normalized-field RK4 steps."""
        point = _validate_state_vector(seed)
        step_size, steps = float(step_size), int(steps)
        if point.shape != (2,) or not np.isfinite(step_size) or step_size <= 0 or steps < 1:
            raise ValueError("seed must be 2D, step_size positive, and steps at least 1")
        limits = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        if limits is not None and (
            limits.shape != (2, 2) or not np.all(np.isfinite(limits))
            or np.any(limits[:, 1] <= limits[:, 0])
        ):
            raise ValueError("bounds must be [[xmin, xmax], [ymin, ymax]]")

        def tangent(position: np.ndarray) -> np.ndarray:
            velocity = np.asarray(vector_field(position.copy()), dtype=np.float64)
            if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
                raise ValueError("vector field must return a finite 2D vector")
            magnitude = np.linalg.norm(velocity)
            return np.zeros(2) if magnitude <= 1e-12 else velocity / magnitude

        points = [point.copy()]
        for _ in range(steps):
            k1 = tangent(point)
            if np.linalg.norm(k1) == 0.0:
                break
            k2 = tangent(point + step_size * k1 / 2)
            k3 = tangent(point + step_size * k2 / 2)
            k4 = tangent(point + step_size * k3)
            next_point = point + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if limits is not None and (
                np.any(next_point < limits[:, 0]) or np.any(next_point > limits[:, 1])
            ):
                break
            points.append(next_point)
            point = next_point
        return np.asarray(points)

    @staticmethod
    def trace_pathline(
        velocity_at: Callable[[float, np.ndarray], np.ndarray],
        initial_position,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Follow a fluid particle through a potentially time-dependent 2D flow."""
        position = _validate_state_vector(initial_position)
        if position.shape != (2,):
            raise ValueError("initial position must be a 2D vector")

        def derivative(time: float, state: np.ndarray) -> np.ndarray:
            velocity = np.asarray(velocity_at(time, state.copy()), dtype=np.float64)
            if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
                raise ValueError("velocity field must return a finite 2D vector")
            return velocity

        return integrate_rk4(derivative, position, duration, dt)


class WavesEngine:
    """Wave equations for traveling, standing, and interference patterns."""

    @staticmethod
    def traveling_wave(amplitude: float, wave_number: float, angular_frequency: float, phase: float, x: float, t: float) -> float:
        amplitude, wave_number, angular_frequency, phase, x, t = map(
            float,
            (amplitude, wave_number, angular_frequency, phase, x, t),
        )
        if not all(np.isfinite(value) for value in (amplitude, wave_number, angular_frequency, phase, x, t)):
            raise ValueError("wave parameters and coordinates must be finite")
        return float(amplitude * np.sin(wave_number * x - angular_frequency * t + phase))

    @staticmethod
    def standing_wave(amplitude: float, wave_number: float, angular_frequency: float, phase: float, x: float, t: float) -> float:
        amplitude, wave_number, angular_frequency, phase, x, t = map(
            float,
            (amplitude, wave_number, angular_frequency, phase, x, t),
        )
        if not all(np.isfinite(value) for value in (amplitude, wave_number, angular_frequency, phase, x, t)):
            raise ValueError("wave parameters and coordinates must be finite")
        return float(2.0 * amplitude * np.sin(wave_number * x + phase) * np.cos(angular_frequency * t))

    @staticmethod
    def phase_velocity(wave_number: float, angular_frequency: float) -> float:
        wave_number, angular_frequency = float(wave_number), float(angular_frequency)
        if not np.isfinite(wave_number) or wave_number == 0 or not np.isfinite(angular_frequency):
            raise ValueError("wave number must be finite and nonzero; angular frequency finite")
        return angular_frequency / wave_number

    @staticmethod
    def superpose(waves, x, t: float = 0.0) -> np.ndarray:
        x_values = np.asarray(x, dtype=np.float64)
        t = float(t)
        if x_values.ndim != 1 or not np.all(np.isfinite(x_values)) or not np.isfinite(t):
            raise ValueError("x must be a finite vector and t finite")
        total = np.zeros_like(x_values)
        for wave in waves:
            if len(wave) != 4:
                raise ValueError("each wave must be (amplitude, wave_number, angular_frequency, phase)")
            amplitude, wave_number, angular_frequency, phase = map(float, wave)
            if not all(np.isfinite(v) for v in (amplitude, wave_number, angular_frequency, phase)):
                raise ValueError("wave parameters must be finite")
            total += amplitude * np.sin(wave_number * x_values - angular_frequency * t + phase)
        return total

    @staticmethod
    def sample_wave(
        amplitude: float,
        wave_number: float,
        angular_frequency: float,
        phase: float,
        x_range,
        samples: int = 200,
        t: float = 0.0,
    ) -> np.ndarray:
        bounds = np.asarray(x_range, dtype=np.float64)
        samples, t = int(samples), float(t)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0]:
            raise ValueError("x_range must contain finite increasing bounds")
        if samples < 2 or not np.isfinite(t):
            raise ValueError("samples must be at least 2 and t finite")
        x = np.linspace(bounds[0], bounds[1], samples)
        y = WavesEngine.superpose([(amplitude, wave_number, angular_frequency, phase)], x, t)
        return np.column_stack((x, y))

    @staticmethod
    def sample_spacetime(
        amplitude: float,
        wave_number: float,
        angular_frequency: float,
        phase: float,
        x_range,
        time_range,
        *,
        samples_x: int = 200,
        samples_t: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return x, time, and a displacement grid ready for frame animation."""
        x_bounds = np.asarray(x_range, dtype=np.float64)
        t_bounds = np.asarray(time_range, dtype=np.float64)
        samples_x, samples_t = int(samples_x), int(samples_t)
        values = tuple(map(float, (amplitude, wave_number, angular_frequency, phase)))
        if x_bounds.shape != (2,) or t_bounds.shape != (2,):
            raise ValueError("x_range and time_range must each contain two bounds")
        if not np.all(np.isfinite(x_bounds)) or x_bounds[1] <= x_bounds[0]:
            raise ValueError("x_range must be finite and increasing")
        if not np.all(np.isfinite(t_bounds)) or t_bounds[1] < t_bounds[0]:
            raise ValueError("time_range must be finite and non-decreasing")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("wave parameters must be finite")
        if samples_x < 2 or samples_t < 1:
            raise ValueError("samples_x must be at least 2 and samples_t at least 1")
        x = np.linspace(*x_bounds, samples_x)
        times = np.linspace(*t_bounds, samples_t)
        displacement = values[0] * np.sin(
            values[1] * x[None, :] - values[2] * times[:, None] + values[3]
        )
        return x, times, displacement


# Backwards-compatible alias for the original motion-engine naming.
Mechanics = MechanicsEngine
Electrostatics = ElectrostaticsEngine
Magnetics = MagneticsEngine
Fluids = FluidsEngine
Waves = WavesEngine
Kinematics = KinematicsEngine


class EngineCatalog:
    """Registry and resolution utilities for the domain engine library."""

    _registry = {
        "advanced_calculus": AdvancedCalculusEngine,
        "calculus": CalculusEngine,
        "geometry": GeometryEngine,
        "linear_algebra": LinearAlgebraEngine,
        "kinematics": KinematicsEngine,
        "mechanics": MechanicsEngine,
        "rotational_mechanics": RotationalMechanicsEngine,
        "electrostatics": ElectrostaticsEngine,
        "magnetics": MagneticsEngine,
        "electrodynamics": ElectrodynamicsEngine,
        "fluids": FluidsEngine,
        "waves": WavesEngine,
        "vector_calculus": VectorCalculusEngine,
        "differential_equations": DifferentialEquationsEngine,
    }

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def resolve(cls, name: str):
        key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
        alias_map = {
            "linearalgebra": "linear_algebra",
            "lin_alg": "linear_algebra",
            "advancedcalculus": "advanced_calculus",
            "multivariable_calculus": "advanced_calculus",
            "rotational": "rotational_mechanics",
            "rotation": "rotational_mechanics",
            "rotational_mechanics": "rotational_mechanics",
            "vectorcalculus": "vector_calculus",
            "vectorcalc": "vector_calculus",
            "vector_calculus": "vector_calculus",
            "diff_eq": "differential_equations",
            "diff_eqs": "differential_equations",
            "ode": "differential_equations",
            "odes": "differential_equations",
            "differential_equation": "differential_equations",
            "differential_equations": "differential_equations",
            "differentialequations": "differential_equations",
            "electromagnetism": "electrodynamics",
            "electromagnetics": "electrodynamics",
            "magnetostatics": "magnetics",
            "kinetics": "kinematics",
            "kinematic": "kinematics",
            "dynamics": "mechanics",
        }
        key = alias_map.get(key, key)
        if key not in cls._registry:
            raise KeyError(f"unsupported engine: {name!r}")
        return cls._registry[key]

    @classmethod
    def instantiate(cls, name: str, **kwargs):
        engine = cls.resolve(name)
        if isinstance(engine, type):
            return engine(**kwargs)
        return engine

    @classmethod
    def describe(cls) -> dict[str, str]:
        descriptions = {
            "advanced_calculus": "Partial derivatives, Jacobians, Hessians, directional derivatives, and line integrals.",
            "calculus": "Derivatives, integrals, and analytic approximation tools.",
            "geometry": "Euclidean measurements, polygon centroids, and affine transformations.",
            "linear_algebra": "Vector, matrix, least-squares, and sampled transformation operations.",
            "kinematics": "Position and velocity evolution under simple motion models.",
            "mechanics": "Projectile motion and RK4 particle dynamics under force fields.",
            "rotational_mechanics": "Moment of inertia, variable-torque dynamics, and rotational energy.",
            "electrostatics": "Coulomb field generation and charge interactions.",
            "magnetics": "Magnetic field approximations and current-centric geometry.",
            "electrodynamics": "Lorentz dynamics, sampled plane waves, and electromagnetic energy flux.",
            "fluids": "RK4 pathlines, steady streamlines, and incompressible flow diagnostics.",
            "waves": "Traveling, standing, and sampled space-time wave fields.",
            "vector_calculus": "Gradient, divergence, and flow intuition in multiple dimensions.",
            "differential_equations": "Decay, resonance, and dynamic evolution at the heart of physics.",
        }
        return {key: descriptions.get(key, "") for key in cls.available()}


class UniversalEngine:
    """Validated entry point for explicitly selected registered calculations."""

    def __init__(self, default_dt: float = 1.0 / 120.0):
        default_dt = float(default_dt)
        if not np.isfinite(default_dt) or default_dt <= 0:
            raise ValueError("default_dt must be finite and positive")
        self.default_dt = default_dt

    @staticmethod
    def available_domains() -> list[str]:
        return EngineCatalog.available()

    @staticmethod
    def resolve_domain(domain: str):
        return EngineCatalog.resolve(domain)

    @staticmethod
    def operations() -> dict[str, dict[str, str]]:
        from engine_router import EngineRouter

        return EngineRouter.operations()

    @staticmethod
    def execute(domain: str, operation: str, parameters: dict):
        from engine_router import EngineRouter

        return EngineRouter.execute(domain, operation, parameters)
