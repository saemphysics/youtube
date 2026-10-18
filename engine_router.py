"""Safe dispatch from declarative model operations to registered math engines."""
from __future__ import annotations

import ast
import inspect
import math
import operator
from typing import Any

import numpy as np

from physics_engine import EngineCatalog

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": np.abs,
    "cos": np.cos,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tan": np.tan,
}


def _expression_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        value = float(node.value)
        if not math.isfinite(value):
            raise ValueError("expression constants must be finite")
        return value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _expression_node(node.left, variables),
            _expression_node(node.right, variables),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_expression_node(node.operand, variables))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](_expression_node(node.args[0], variables))
    raise ValueError(f"unsupported expression syntax: {type(node).__name__}")


class SafeExpression:
    """A small, non-evaluating arithmetic expression callable for engine inputs."""

    def __init__(self, expression: str):
        self.expression = str(expression).strip()
        if not self.expression:
            raise ValueError("expression cannot be empty")
        try:
            self.tree = ast.parse(self.expression, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"invalid expression: {exc.msg}") from exc
        names = {
            node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)
        }
        variables = names - set(_FUNCTIONS)
        if not variables or variables - {"x", "t"}:
            raise ValueError("expression must use x or t and supported math functions")
        self.variables = variables
        self._validate(self.tree)

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Expression, ast.Load, ast.Name, ast.Constant)):
            if isinstance(node, ast.Name) and node.id not in self.variables | set(_FUNCTIONS):
                raise ValueError(f"unsupported expression name: {node.id}")
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            self._validate(node.left)
            self._validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            self._validate(node.operand)
            return
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCTIONS
                or len(node.args) != 1
                or node.keywords
            ):
                raise ValueError("only single-argument standard math functions are supported")
            self._validate(node.args[0])
            return
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)):
            return
        raise ValueError(f"unsupported expression syntax: {type(node).__name__}")

    def __call__(self, value: Any) -> Any:
        result = _expression_node(self.tree, {name: value for name in self.variables})
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("expression produced a non-finite result")
        return result


def _prepare(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"expression"}:
            return SafeExpression(value["expression"])
        return {key: _prepare(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_prepare(item) for item in value]
    return value


def to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_json_value(value.tolist())
    if isinstance(value, np.generic):
        return to_json_value(value.item())
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("engine result contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"engine result is not JSON-compatible: {type(value).__name__}")


class EngineRouter:
    """Expose registered numerical methods as validated data-in/data-out calls."""

    @staticmethod
    def operations() -> dict[str, dict[str, str]]:
        catalog = {}
        for domain in EngineCatalog.available():
            engine = EngineCatalog.resolve(domain)
            catalog[domain] = {
                name: str(inspect.signature(method))
                for name, method in inspect.getmembers(engine, callable)
                if not name.startswith("_")
                and not name.startswith("from_")
                and not name.startswith("to_")
                and _is_public_operation(method)
            }
        return catalog

    @staticmethod
    def execute(domain: str, operation: str, parameters: dict[str, Any]) -> Any:
        if not isinstance(parameters, dict):
            raise TypeError("engine parameters must be a JSON object")
        engine = EngineCatalog.resolve(domain)
        method = getattr(engine, operation, None)
        if operation.startswith("_") or not callable(method) or not _is_public_operation(method):
            raise ValueError(f"unsupported operation {operation!r} for engine {domain!r}")

        signature = inspect.signature(method)
        positional = [
            parameter for parameter in signature.parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if any(
            parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            for parameter in signature.parameters.values()
        ):
            raise ValueError(f"operation {domain}.{operation} has a dynamic signature")

        unknown = set(parameters) - {parameter.name for parameter in positional}
        missing = {
            parameter.name
            for parameter in positional
            if parameter.default is inspect.Parameter.empty
            and parameter.name not in parameters
        }
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing {', '.join(sorted(missing))}")
            if unknown:
                details.append(f"unknown {', '.join(sorted(unknown))}")
            raise ValueError(f"invalid parameters for {domain}.{operation}: {'; '.join(details)}")

        prepared = {name: _prepare(value) for name, value in parameters.items()}
        return to_json_value(method(**prepared))


def _is_public_operation(method: Any) -> bool:
    if not callable(method):
        return False
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return all(
        parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        for parameter in signature.parameters.values()
    )
