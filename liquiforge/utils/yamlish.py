# -*- coding: utf-8 -*-
"""A minimal YAML subset parser (maps, lists, scalars) — zero dependencies.

Supports the small surface area liquiforge recipes need:
- ``key: value`` scalars (str / int / float / bool / null)
- nested mappings by indentation
- block lists (``key:`` followed by ``- item`` lines at deeper indent)
- inline ``[a, b]`` and ``{k: v}`` flows
No anchors, tags, multi-line strings, or other full-YAML features.
"""
from __future__ import annotations


def _scalar(text: str):
    t = text.strip()
    if t == "" or t in ("null", "~"):
        return None
    if t == "true":
        return True
    if t == "false":
        return False
    low = t.lower()
    if t.startswith('"') and t.endswith('"'):
        return t[1:-1]
    if t.startswith("'") and t.endswith("'"):
        return t[1:-1]
    if low in ("inf", "nan"):
        return float(low)
    try:
        if any(c in t for c in ".eE") or low.startswith(("-", "+")):
            return float(t)
        return int(t)
    except ValueError:
        return t


def _parse_flow(t: str):
    t = t.strip()
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        return [_scalar(x) for x in inner.split(",")]
    if t.startswith("{") and t.endswith("}"):
        inner = t[1:-1].strip()
        if not inner:
            return {}
        out = {}
        for pair in inner.split(","):
            k, _, v = pair.partition(":")
            out[_scalar(k)] = _scalar(v)
        return out
    return _scalar(t)


def yaml_loads(text: str) -> dict:
    root: dict = {}
    # stack of (indent, container, parent, key, pending_list)
    stack: list = [(-1, root, None, None, False)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        # pop deeper entries
        while stack and indent <= stack[-1][0]:
            stack.pop()

        indent_cur, container, parent, key, _pending = stack[-1]

        if line.startswith("- "):
            item = _parse_flow(line[2:].strip())
            # attach list to the parent key at parent's indentation
            if parent is not None and not isinstance(container, list):
                lst: list = []
                parent[key] = lst
                # replace the parent mapping frame with a list frame
                # find parent frame to patch
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][2] is parent:
                        stack[i] = (stack[i][0], parent, stack[i][2], stack[i][3], True)
                # push a list frame
                stack.append((indent, lst, parent, key, False))
                container = lst
            if isinstance(container, list):
                container.append(item)
            else:
                container[item] = None
            continue

        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip().strip("\"'")
            v = v.strip()
            if v == "":
                # could be a nested mapping or a block list
                child: dict = {}
                container[k] = child
                stack.append((indent, child, container, k, False))
            else:
                container[k] = _parse_flow(v)
        else:
            # bare list item without dash (lenient)
            container[line] = None

    return root
