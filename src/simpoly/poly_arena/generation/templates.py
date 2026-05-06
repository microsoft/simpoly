# SPDX-License-Identifier: MIT

import re
import typing as ty


def variable_to_placeholder(v: str, symbol: str) -> str:
    return symbol + v + symbol


def config_to_placeholder_dict(d: dict[str, ty.Any], symbol: str = "_") -> dict[str, ty.Any]:
    return {variable_to_placeholder(k, symbol=symbol): v for k, v in d.items()}


def render_template(template: str, d: dict[str, ty.Any]) -> str:
    for k, v in d.items():
        new_s = template.replace(k, str(v))
        if new_s == template:
            raise RuntimeError(f"Replacement of '{k}' had no effect: {template}")
        template = new_s

    assert_no_blanks_left(template)
    return template


def assert_no_blanks_left(s: str) -> None:
    pattern = re.compile(r"\s_(\w+)_\s")
    match = pattern.search(s)
    if match is not None:
        raise AssertionError(f"Blank remaining in template: {match.group(0)}")
