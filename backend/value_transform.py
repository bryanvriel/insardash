from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np


MAX_TRANSFORM_LENGTH = 256
IDENTITY_TRANSFORM = ""

ALLOWED_NP_NAMES: dict[str, Any] = {
    "abs": np.abs,
    "absolute": np.absolute,
    "angle": np.angle,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "ceil": np.ceil,
    "clip": np.clip,
    "cos": np.cos,
    "deg2rad": np.deg2rad,
    "degrees": np.degrees,
    "exp": np.exp,
    "expm1": np.expm1,
    "floor": np.floor,
    "hypot": np.hypot,
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    "log10": np.log10,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "mod": np.mod,
    "nan_to_num": np.nan_to_num,
    "pi": np.pi,
    "rad2deg": np.rad2deg,
    "radians": np.radians,
    "round": np.round,
    "sign": np.sign,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "square": np.square,
    "tan": np.tan,
    "where": np.where,
}

ALLOWED_BINARY_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)
ALLOWED_COMPARE_OPS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


class TransformError(ValueError):
    """Raised when a value transform is invalid or cannot be applied."""


@dataclass(frozen=True)
class ValueTransform:
    expression: str
    code: Any | None = None

    @property
    def is_identity(self) -> bool:
        return self.expression == IDENTITY_TRANSFORM

    def apply(self, values: Any) -> np.ndarray:
        array = np.asarray(values)
        if self.is_identity:
            return array

        namespace = {"x": array, "np": _NumpyNamespace()}
        try:
            with np.errstate(all="ignore"):
                result = eval(self.code, {"__builtins__": {}}, namespace)
        except Exception as exc:  # pragma: no cover - exact numpy exception varies by expression
            raise TransformError(f"Unable to apply transform {self.expression!r}: {exc}") from exc

        return _coerce_transform_result(result, array.shape, self.expression)

    def apply_scalar(self, value: float) -> float:
        result = self.apply(np.asarray(value, dtype=np.float64))
        return float(np.asarray(result).reshape(()))


class _NumpyNamespace:
    def __getattr__(self, name: str) -> Any:
        try:
            return ALLOWED_NP_NAMES[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _TransformValidator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, ALLOWED_BINARY_OPS):
            raise TransformError("Only +, -, *, /, //, %, and ** operators are allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, ALLOWED_UNARY_OPS):
            raise TransformError("Only unary + and - operators are allowed")
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> None:
        for operator in node.ops:
            if not isinstance(operator, ALLOWED_COMPARE_OPS):
                raise TransformError("Only numeric comparison operators are allowed")
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Call(self, node: ast.Call) -> None:
        if node.keywords:
            raise TransformError("Transform functions do not accept keyword arguments")
        if not isinstance(node.func, ast.Attribute):
            raise TransformError("Transform functions must be called as np.<name>(...)")
        self._validate_np_attribute(node.func, allow_callable=True)
        for argument in node.args:
            self.visit(argument)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._validate_np_attribute(node, allow_callable=False)

    def _validate_np_attribute(self, node: ast.Attribute, allow_callable: bool) -> None:
        if node.attr.startswith("_"):
            raise TransformError("Private NumPy attributes are not allowed")
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise TransformError("Only np.<name> attributes are allowed")
        if node.attr not in ALLOWED_NP_NAMES:
            raise TransformError(f"np.{node.attr} is not an allowed transform function or constant")
        value = ALLOWED_NP_NAMES[node.attr]
        if callable(value) and not allow_callable:
            raise TransformError(f"np.{node.attr} must be called as a function")
        if not callable(value) and allow_callable:
            raise TransformError(f"np.{node.attr} is not callable")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id != "x":
            raise TransformError(f"Unknown transform name: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TransformError("Only numeric constants are allowed")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        raise TransformError("Indexing is not allowed in map transforms")

    def visit_List(self, node: ast.List) -> None:
        raise TransformError("Lists are not allowed in map transforms")

    def visit_Tuple(self, node: ast.Tuple) -> None:
        raise TransformError("Tuples are not allowed in map transforms")

    def visit_Dict(self, node: ast.Dict) -> None:
        raise TransformError("Dictionaries are not allowed in map transforms")

    def generic_visit(self, node: ast.AST) -> None:
        raise TransformError(f"Unsupported transform syntax: {type(node).__name__}")


def compile_transform(expression: str | None) -> ValueTransform:
    normalized = normalize_transform(expression)
    if normalized == IDENTITY_TRANSFORM:
        return ValueTransform(expression=IDENTITY_TRANSFORM)

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise TransformError(f"Invalid transform syntax: {exc.msg}") from exc

    _TransformValidator().visit(tree)
    code = compile(tree, "<map-transform>", "eval")
    return ValueTransform(expression=normalized, code=code)


def normalize_transform(expression: str | None) -> str:
    if expression is None:
        return IDENTITY_TRANSFORM
    normalized = expression.strip()
    if not normalized:
        return IDENTITY_TRANSFORM
    if len(normalized) > MAX_TRANSFORM_LENGTH:
        raise TransformError(f"Transform must be {MAX_TRANSFORM_LENGTH} characters or fewer")
    return normalized


def _coerce_transform_result(result: Any, input_shape: tuple[int, ...], expression: str) -> np.ndarray:
    array = np.asarray(result)
    if array.shape == input_shape:
        return array
    if array.shape == ():
        return np.full(input_shape, array.item())
    raise TransformError(
        f"Transform {expression!r} returned shape {array.shape}, expected scalar or {input_shape}"
    )
