from __future__ import annotations

import numpy as np
import pytest

from backend.value_transform import TransformError, ValueTransform, compile_transform


def test_allowed_transforms_match_numpy() -> None:
    x = np.asarray([-np.pi, -1.0, 0.25, 1.0, 10.0], dtype=np.float64)
    positive = np.asarray([0.1, 1.0, 10.0], dtype=np.float64)

    cases = [
        (None, x, x),
        ("", x, x),
        ("np.abs(x)", x, np.abs(x)),
        ("x % (2*np.pi)", x, x % (2 * np.pi)),
        ("np.cos(x)", x, np.cos(x)),
        ("np.sin(x)", x, np.sin(x)),
        ("np.log10(x)", positive, np.log10(positive)),
    ]

    for expression, values, expected in cases:
        result = compile_transform(expression).apply(values)
        np.testing.assert_allclose(result, expected)


def test_scalar_constants_are_broadcast() -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    result = compile_transform("np.pi").apply(values)

    np.testing.assert_allclose(result, np.full(values.shape, np.pi))


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo nope')",
        "open('/tmp/nope')",
        "np.__dict__",
        "np.abs",
        "np.pi(x)",
        "np.asarray(x)",
        "x[0]",
        "[x]",
        "(x, x)",
        "x and x",
    ],
)
def test_unsafe_or_unsupported_transforms_are_rejected(expression: str) -> None:
    with pytest.raises(TransformError):
        compile_transform(expression)


def test_incompatible_result_shape_is_rejected() -> None:
    transform = ValueTransform("bad shape", compile("x[:, 0]", "<test-transform>", "eval"))

    with pytest.raises(TransformError):
        transform.apply(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
