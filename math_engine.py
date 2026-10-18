"""Numerical mathematics primitives for topic-specific learning engines."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = [
    "AdvancedCalculusEngine",
    "CalculusEngine",
    "GeometryEngine",
    "LinearAlgebraEngine",
    "VectorCalculusEngine",
    "DifferentialEquationsEngine",
]


class CalculusEngine:
    """Core numerical calculus primitives for animation math."""

    @staticmethod
    def derivative(
        func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray | float,
        *,
        h: float = 1e-6,
    ):
        h = float(h)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")

        x_arr = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x must contain only finite values")
        if x_arr.ndim == 0:
            value = (func(x_arr + h) - func(x_arr - h)) / (2.0 * h)
            result = float(value)
            if not np.isfinite(result):
                raise FloatingPointError("derivative produced a non-finite value")
            return result
        if x_arr.ndim != 1 or x_arr.size < 3:
            raise ValueError("x must be a scalar or a 1D array with at least 3 values")
        if np.any(np.diff(x_arr) <= 0):
            raise ValueError("x values must be strictly increasing")

        values = np.asarray(func(x_arr), dtype=np.float64)
        if values.shape != x_arr.shape:
            raise ValueError("function output must match the input shape")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("function produced non-finite values")
        return np.gradient(values, x_arr, edge_order=2)

    @staticmethod
    def integrate(
        func: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n: int = 1000,
    ) -> float:
        n = int(n)
        a = float(a)
        b = float(b)
        if n < 2:
            raise ValueError("n must be at least 2")
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("integration bounds must be finite")
        xs = np.linspace(a, b, n + 1)
        ys = np.asarray(func(xs), dtype=np.float64)
        if ys.shape != xs.shape:
            raise ValueError("function output must match the sample shape")
        if not np.all(np.isfinite(ys)):
            raise FloatingPointError("function produced non-finite values")
        return float(np.trapezoid(ys, xs))


class AdvancedCalculusEngine:
    """Finite-difference multivariable calculus and sampled visual data."""

    @staticmethod
    def partial_derivative(scalar_field, point, axis: int, *, h: float = 1e-5) -> float:
        p = np.asarray(point, dtype=np.float64)
        axis = int(axis)
        h = float(h)
        if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
            raise ValueError("point must be a finite, non-empty vector")
        if axis < 0 or axis >= p.size:
            raise ValueError("axis is outside the point dimension")
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        offset = np.zeros_like(p)
        offset[axis] = h
        value = (scalar_field(p + offset) - scalar_field(p - offset)) / (2 * h)
        if not np.isfinite(value):
            raise FloatingPointError("partial derivative is non-finite")
        return float(value)

    @classmethod
    def jacobian(cls, vector_field, point, *, h: float = 1e-5) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
            raise ValueError("point must be a finite, non-empty vector")
        baseline = np.asarray(vector_field(p), dtype=np.float64)
        if baseline.ndim != 1 or baseline.size == 0 or not np.all(np.isfinite(baseline)):
            raise ValueError("vector field must return a finite, non-empty vector")
        result = np.empty((baseline.size, p.size), dtype=np.float64)
        h = float(h)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        for axis in range(p.size):
            offset = np.zeros_like(p)
            offset[axis] = h
            plus = np.asarray(vector_field(p + offset), dtype=np.float64)
            minus = np.asarray(vector_field(p - offset), dtype=np.float64)
            if plus.shape != baseline.shape or minus.shape != baseline.shape:
                raise ValueError("vector field output shape must remain constant")
            result[:, axis] = (plus - minus) / (2 * h)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Jacobian contains non-finite values")
        return result

    @classmethod
    def hessian(cls, scalar_field, point, *, h: float = 1e-4) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        h = float(h)
        if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
            raise ValueError("point must be a finite, non-empty vector")
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        center = float(scalar_field(p))
        if not np.isfinite(center):
            raise FloatingPointError("scalar field returned a non-finite value")
        result = np.empty((p.size, p.size), dtype=np.float64)
        offsets = np.eye(p.size) * h
        for i in range(p.size):
            result[i, i] = (
                scalar_field(p + offsets[i]) - 2 * center + scalar_field(p - offsets[i])
            ) / h**2
            for j in range(i):
                value = (
                    scalar_field(p + offsets[i] + offsets[j])
                    - scalar_field(p + offsets[i] - offsets[j])
                    - scalar_field(p - offsets[i] + offsets[j])
                    + scalar_field(p - offsets[i] - offsets[j])
                ) / (4 * h**2)
                result[i, j] = result[j, i] = value
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Hessian contains non-finite values")
        return result

    @staticmethod
    def sample_curve(func, start: float, stop: float, samples: int = 200) -> np.ndarray:
        start = float(start)
        stop = float(stop)
        samples = int(samples)
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise ValueError("curve bounds must be finite and increasing")
        if samples < 2:
            raise ValueError("samples must be at least 2")
        x = np.linspace(start, stop, samples)
        y = np.asarray(func(x), dtype=np.float64)
        if y.shape != x.shape or not np.all(np.isfinite(y)):
            raise ValueError("curve function must return finite values matching the samples")
        return np.column_stack((x, y))

    @staticmethod
    def directional_derivative(scalar_field, point, direction, *, h: float = 1e-5) -> float:
        p = np.asarray(point, dtype=np.float64)
        vector = np.asarray(direction, dtype=np.float64)
        h = float(h)
        if p.ndim != 1 or p.size == 0 or vector.shape != p.shape:
            raise ValueError("point and direction must be matching non-empty vectors")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(vector)):
            raise ValueError("point and direction must be finite")
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("direction must be non-zero")
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        unit = vector / norm
        value = (scalar_field(p + h * unit) - scalar_field(p - h * unit)) / (2 * h)
        if not np.isfinite(value):
            raise FloatingPointError("directional derivative is non-finite")
        return float(value)

    @staticmethod
    def arc_length(parametric_curve, start: float, stop: float, samples: int = 1000) -> float:
        start, stop, samples = float(start), float(stop), int(samples)
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise ValueError("curve bounds must be finite and increasing")
        if samples < 2:
            raise ValueError("samples must be at least 2")
        parameters = np.linspace(start, stop, samples)
        points = np.asarray(parametric_curve(parameters), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] != samples or points.shape[1] == 0:
            raise ValueError("parametric curve must return one non-empty point per sample")
        if not np.all(np.isfinite(points)):
            raise FloatingPointError("parametric curve returned non-finite points")
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    @staticmethod
    def line_integral(vector_field, parametric_curve, start: float, stop: float, samples: int = 1000) -> float:
        start, stop, samples = float(start), float(stop), int(samples)
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise ValueError("curve bounds must be finite and increasing")
        if samples < 3:
            raise ValueError("samples must be at least 3")
        parameters = np.linspace(start, stop, samples)
        points = np.asarray(parametric_curve(parameters), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] != samples or points.shape[1] == 0:
            raise ValueError("parametric curve must return one non-empty point per sample")
        if not np.all(np.isfinite(points)):
            raise FloatingPointError("parametric curve returned non-finite points")
        tangents = np.gradient(points, parameters, axis=0, edge_order=2)
        values = np.asarray([vector_field(point) for point in points], dtype=np.float64)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise ValueError("vector field must return finite vectors matching curve dimension")
        return float(np.trapezoid(np.einsum("ij,ij->i", values, tangents), parameters))


class GeometryEngine:
    """Euclidean geometry primitives used in visual proofs and diagrams."""

    @staticmethod
    def distance(point_a, point_b) -> float:
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        if a.shape != b.shape or a.ndim != 1 or a.size == 0:
            raise ValueError("points must be non-empty vectors with matching shapes")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("points must be finite")
        return float(np.linalg.norm(a - b))

    @staticmethod
    def angle_between(vector_a, vector_b) -> float:
        a = np.asarray(vector_a, dtype=np.float64)
        b = np.asarray(vector_b, dtype=np.float64)
        if a.shape != b.shape or a.ndim != 1 or a.size == 0:
            raise ValueError("vectors must be non-empty vectors with matching shapes")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("vectors must be finite")
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0.0:
            raise ValueError("angle between zero vectors is undefined")
        cosine = np.clip(np.dot(a, b) / denominator, -1.0, 1.0)
        return float(np.arccos(cosine))

    @staticmethod
    def line_equation(point_a, point_b):
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        if a.shape != (2,) or b.shape != (2,):
            raise ValueError("line endpoints must be 2D vectors")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("line endpoints must be finite")
        delta = b - a
        if np.allclose(delta, 0.0):
            raise ValueError("line endpoints must be distinct")
        if abs(delta[0]) <= 1e-12:
            return float("inf"), float(a[0])
        slope = delta[1] / delta[0]
        intercept = a[1] - slope * a[0]
        return float(slope), float(intercept)

    @staticmethod
    def polygon_area(vertices) -> float:
        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError("polygon requires at least three 2D vertices")
        if not np.all(np.isfinite(points)):
            raise ValueError("polygon vertices must be finite")
        x, y = points.T
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def polygon_centroid(vertices) -> np.ndarray:
        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError("polygon requires at least three 2D vertices")
        if not np.all(np.isfinite(points)):
            raise ValueError("polygon vertices must be finite")
        following = np.roll(points, -1, axis=0)
        cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
        signed_area_twice = np.sum(cross)
        if abs(signed_area_twice) <= 1e-12:
            raise ValueError("polygon centroid is undefined for zero-area vertices")
        weighted_sum = np.sum((points + following) * cross[:, None], axis=0)
        return weighted_sum / (3.0 * signed_area_twice)

    @staticmethod
    def affine_transform(points, matrix, translation=None) -> np.ndarray:
        coordinates = np.asarray(points, dtype=np.float64)
        transform = np.asarray(matrix, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] == 0:
            raise ValueError("points must be a non-empty matrix of coordinates")
        if transform.shape != (coordinates.shape[1], coordinates.shape[1]):
            raise ValueError("matrix dimensions must match point dimensions")
        offset = (np.zeros(coordinates.shape[1], dtype=np.float64)
                  if translation is None else np.asarray(translation, dtype=np.float64))
        if offset.shape != (coordinates.shape[1],):
            raise ValueError("translation must match point dimensions")
        if not all(np.all(np.isfinite(value)) for value in (coordinates, transform, offset)):
            raise ValueError("points, matrix, and translation must be finite")
        return coordinates @ transform.T + offset

    @staticmethod
    def rotate(point, angle_radians: float, *, origin=(0.0, 0.0)) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        center = np.asarray(origin, dtype=np.float64)
        angle = float(angle_radians)
        if p.shape != (2,) or center.shape != (2,) or not np.isfinite(angle):
            raise ValueError("point and origin must be 2D, with a finite angle")
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        return center + rotation @ (p - center)


class LinearAlgebraEngine:
    """Lightweight linear algebra utilities for vector and matrix problems."""

    @staticmethod
    def dot(vector_a, vector_b):
        a = np.asarray(vector_a, dtype=np.float64)
        b = np.asarray(vector_b, dtype=np.float64)
        if a.ndim != 1 or a.shape != b.shape:
            raise ValueError("dot-product operands must be vectors with matching shapes")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("dot-product operands must be finite")
        return float(np.dot(a, b))

    @staticmethod
    def matrix_vector_product(matrix, vector):
        m = np.asarray(matrix, dtype=np.float64)
        v = np.asarray(vector, dtype=np.float64)
        if m.ndim != 2 or v.ndim != 1 or m.shape[1] != v.size:
            raise ValueError("matrix and vector dimensions are incompatible")
        if not np.all(np.isfinite(m)) or not np.all(np.isfinite(v)):
            raise ValueError("matrix and vector must be finite")
        return np.asarray(m @ v, dtype=np.float64)

    @staticmethod
    def normalize(vector):
        v = np.asarray(vector, dtype=np.float64)
        if v.ndim != 1 or v.size == 0:
            raise ValueError("vector must be non-empty and one-dimensional")
        if not np.all(np.isfinite(v)):
            raise ValueError("vector must be finite")
        norm = np.linalg.norm(v)
        if norm == 0.0:
            raise ValueError("cannot normalize a zero vector")
        return v / norm

    @staticmethod
    def solve(matrix, vector) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64)
        v = np.asarray(vector, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or v.shape != (m.shape[0],):
            raise ValueError("solve requires a square matrix and matching vector")
        if not np.all(np.isfinite(m)) or not np.all(np.isfinite(v)):
            raise ValueError("matrix and vector must be finite")
        try:
            return np.linalg.solve(m, v)
        except np.linalg.LinAlgError as error:
            raise ValueError("matrix is singular") from error

    @staticmethod
    def eigenvalues(matrix) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.all(np.isfinite(m)):
            raise ValueError("eigenvalues requires a finite square matrix")
        values = np.linalg.eigvals(m)
        return np.asarray(np.sort(values)[::-1], dtype=np.float64)

    @staticmethod
    def eigendecomposition(matrix) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.all(np.isfinite(m)):
            raise ValueError("eigendecomposition requires a finite square matrix")
        values, vectors = np.linalg.eig(m)
        order = np.argsort(np.real(values))[::-1]
        return values[order], vectors[:, order]

    @staticmethod
    def singular_value_decomposition(matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2 or min(m.shape, default=0) == 0 or not np.all(np.isfinite(m)):
            raise ValueError("SVD requires a finite non-empty matrix")
        return np.linalg.svd(m, full_matrices=False)

    @staticmethod
    def least_squares(matrix, target) -> np.ndarray:
        m = np.asarray(matrix, dtype=np.float64)
        b = np.asarray(target, dtype=np.float64)
        if m.ndim != 2 or min(m.shape, default=0) == 0:
            raise ValueError("least-squares matrix must be non-empty and two-dimensional")
        if b.ndim not in (1, 2) or b.shape[0] != m.shape[0]:
            raise ValueError("target must have a matching leading dimension")
        if not np.all(np.isfinite(m)) or not np.all(np.isfinite(b)):
            raise ValueError("matrix and target must be finite")
        solution, *_ = np.linalg.lstsq(m, b, rcond=None)
        return solution

    @staticmethod
    def sample_matrix_interpolation(matrix_start, matrix_end, samples: int = 30) -> np.ndarray:
        start = np.asarray(matrix_start, dtype=np.float64)
        end = np.asarray(matrix_end, dtype=np.float64)
        samples = int(samples)
        if start.ndim != 2 or start.shape != end.shape or min(start.shape, default=0) == 0:
            raise ValueError("endpoint matrices must be non-empty and have matching shapes")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError("endpoint matrices must be finite")
        if samples < 2:
            raise ValueError("samples must be at least 2")
        fractions = np.linspace(0.0, 1.0, samples)
        return (1.0 - fractions[:, None, None]) * start + fractions[:, None, None] * end


class VectorCalculusEngine:
    """Numerical gradient, divergence, and curl for low-dimensional fields."""

    @staticmethod
    def gradient(scalar_field, point, *, h: float = 1e-6):
        p = np.asarray(point, dtype=np.float64)
        h = float(h)
        if p.ndim != 1 or p.size not in (2, 3):
            raise ValueError("point must be a 2D or 3D vector")
        if not np.all(np.isfinite(p)) or not np.isfinite(h) or h <= 0:
            raise ValueError("point and h must be finite, with h positive")
        gradient = np.empty_like(p)
        for axis in range(p.size):
            offset = np.zeros_like(p)
            offset[axis] = h
            gradient[axis] = (
                scalar_field(p + offset) - scalar_field(p - offset)
            ) / (2.0 * h)
        return gradient

    @staticmethod
    def divergence(vector_field, point, *, h: float = 1e-6) -> float:
        p = np.asarray(point, dtype=np.float64)
        h = float(h)
        if p.ndim != 1 or p.size not in (2, 3):
            raise ValueError("point must be a 2D or 3D vector")
        if not np.all(np.isfinite(p)) or not np.isfinite(h) or h <= 0:
            raise ValueError("point and h must be finite, with h positive")
        divergence = 0.0
        for axis in range(p.size):
            offset = np.zeros_like(p)
            offset[axis] = h
            forward = np.asarray(vector_field(p + offset), dtype=np.float64)
            backward = np.asarray(vector_field(p - offset), dtype=np.float64)
            if forward.shape != p.shape or backward.shape != p.shape:
                raise ValueError("vector field output must match the point dimension")
            divergence += (forward[axis] - backward[axis]) / (2.0 * h)
        return float(divergence)

    @staticmethod
    def curl(vector_field, point, *, h: float = 1e-6) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        h = float(h)
        if p.shape != (3,):
            raise ValueError("curl is implemented for 3D fields only")
        if not np.all(np.isfinite(p)) or not np.isfinite(h) or h <= 0:
            raise ValueError("point and h must be finite, with h positive")
        jacobian = np.empty((3, 3), dtype=np.float64)
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = h
            forward = np.asarray(vector_field(p + offset), dtype=np.float64)
            backward = np.asarray(vector_field(p - offset), dtype=np.float64)
            if forward.shape != (3,) or backward.shape != (3,):
                raise ValueError("vector field output must be a 3D vector")
            jacobian[:, axis] = (forward - backward) / (2.0 * h)
        return np.array([
            jacobian[2, 1] - jacobian[1, 2],
            jacobian[0, 2] - jacobian[2, 0],
            jacobian[1, 0] - jacobian[0, 1],
        ])


class DifferentialEquationsEngine:
    """Closed-form differential equation models for intuitive explanation."""

    @staticmethod
    def exponential_decay(
        k: float = 1.0,
        t: float = 0.0,
        initial_value: float = 1.0,
    ) -> float:
        return float(initial_value * np.exp(-float(k) * float(t)))

    @staticmethod
    def harmonic_response(
        omega: float = 1.0,
        t: float = 0.0,
        amplitude: float = 1.0,
    ) -> float:
        return float(amplitude * np.sin(float(omega) * float(t)))
