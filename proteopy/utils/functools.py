import re
import inspect
from functools import partial, update_wrapper
from textwrap import dedent
from typing import Dict, Set, Optional

__all__ = ["partial_with_docsig", "fix"]

# -----------------------
# Docstring utilities
# -----------------------

_NUMPY_START = re.compile(r"^\s*Parameters\s*\Z", re.IGNORECASE)
_GOOGLE_START = re.compile(r"^\s*Args:\s*\Z")
_REST_PARAM  = lambda name: re.compile(rf"^\s*:param\s+{re.escape(name)}\s*:")
_REST_TYPE   = lambda name: re.compile(rf"^\s*:type\s+{re.escape(name)}\s*:")

def _numpy_param_re(name):   # e.g. "greeting : str" or just "greeting"
    return re.compile(
        rf"^(\s*){re.escape(name)}\s*(?::|\Z)",
        re.UNICODE,
    )

def _google_param_re(name):  # e.g. "greeting (str):" or "greeting:"
    return re.compile(rf"^(\s*){re.escape(name)}\s*(\([^)]+\))?\s*:\s*\Z")

def _strip_block_at(lines, start_idx, base_indent_str):
    """
    Remove a parameter block starting at start_idx.
    A block is the start line plus following lines that are more indented (or blank).
    """
    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(' '))

    base = indent_of(lines[start_idx])
    i = start_idx + 1
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.strip() == "":
            i += 1
            continue
        if indent_of(ln) <= base:
            break
        i += 1
    del lines[start_idx:i]
    return start_idx

def _remove_numpy_google_param_blocks(lines, name):
    i = 0
    n = len(lines)
    in_numpy = False
    in_google = False

    while i < n:
        line = lines[i]

        # Section starts
        if _NUMPY_START.match(line):
            in_numpy, in_google = True, False
            i += 1; continue
        if _GOOGLE_START.match(line):
            in_numpy, in_google = False, True
            i += 1; continue

        # Section termination heuristics
        if in_numpy and re.match(r"^\s*\S.*\Z", line) and line.strip().endswith(":") and line.strip().lower() not in {"parameters:"}:
            in_numpy = False
        if in_google and (line.strip() and not line.startswith(" ")):
            in_google = False

        if in_numpy:
            m = _numpy_param_re(name).match(line)
            if m:
                i = _strip_block_at(lines, i, m.group(1))
                n = len(lines)
                continue

        if in_google:
            m = _google_param_re(name).match(line)
            if m:
                i = _strip_block_at(lines, i, m.group(1))
                n = len(lines)
                continue

        i += 1

def _remove_rest_param_lines(lines, name):
    i = 0
    n = len(lines)
    param_re = _REST_PARAM(name)
    type_re  = _REST_TYPE(name)

    def remove_line_and_continuation(start_idx):
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip(' '))
        j = start_idx + 1
        while j < len(lines):
            ln = lines[j]
            if not ln.strip():
                j += 1
                continue
            indent = len(ln) - len(ln.lstrip(' '))
            if indent <= base_indent:
                break
            j += 1
        del lines[start_idx:j]
        return start_idx

    while i < n:
        line = lines[i]
        if param_re.match(line) or type_re.match(line):
            i = remove_line_and_continuation(i)
            n = len(lines)
            continue
        i += 1

def _format_fixed_note(fixed_map: Dict[str, object], func_name: str) -> str:
    # short, universal note appended to the docstring, now mentioning the original function
    kv = ", ".join(f"{k}={v!r}" for k, v in fixed_map.items())
    return (
        "Note:\n"
        f"    This function is a partial of `{func_name}`, with the following arguments fixed: {kv}."
    )

def _replace_doc_header(lines, new_header: str):
    header_text = dedent(new_header).strip()
    header_lines = header_text.splitlines() if header_text else []

    if not lines:
        return header_lines

    param_idx = None
    for idx, line in enumerate(lines):
        if _NUMPY_START.match(line):
            param_idx = idx
            break

    if param_idx is None:
        return header_lines

    tail = lines[param_idx:]
    while tail and tail[0].strip() == "":
        tail = tail[1:]

    if header_lines:
        return header_lines + [""] + tail
    return tail

def _prune_docstring(doc: str,
                     fixed_names: Set[str],
                     add_note: bool,
                     fixed_map: Dict[str, object],
                     func_name: str,
                     docstr_header: Optional[str]) -> str:
    if doc:
        lines = dedent(doc).splitlines()
    else:
        lines = []

    # Remove parameter entries for fixed names
    if lines:
        for name in fixed_names:
            _remove_numpy_google_param_blocks(lines, name)
            _remove_rest_param_lines(lines, name)

    if docstr_header is not None:
        lines = _replace_doc_header(lines, docstr_header)

    if not lines and docstr_header:
        lines = dedent(docstr_header).strip().splitlines()

    # Normalize blank lines
    pruned = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                pruned.append("")
        else:
            blank_run = 0
            pruned.append(ln)

    doc_out = "\n".join(pruned).strip()

    if add_note and fixed_map:
        note = _format_fixed_note(fixed_map, func_name)
        doc_out = (doc_out + ("\n\n" if doc_out else "") + note).strip()

    return doc_out

# -----------------------
# Signature + metadata
# -----------------------

def _bound_fixed_map(func, args, kwargs) -> Dict[str, object]:
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    return dict(bound.arguments)

def _prune_signature(sig: inspect.Signature, fixed_names: Set[str]) -> inspect.Signature:
    new_params = [p for name, p in sig.parameters.items() if name not in fixed_names]
    return sig.replace(parameters=new_params)

def _prune_annotations(ann: Optional[dict], fixed_names: Set[str]) -> Optional[dict]:
    if not ann:
        return None
    return {k: v for k, v in ann.items() if k not in fixed_names}

# -----------------------
# Public helper
# -----------------------

def partial_with_docsig(
    func,
    /,
    *args,
    add_fixed_note: bool = True,
    docstr_header: Optional[str] = None,
    **kwargs,
):
    """
    Create a functools.partial that:
      - inherits metadata (__name__, __module__, __qualname__, __wrapped__, etc.)
      - removes fixed parameters from the displayed signature
      - removes fixed parameters' entries from the docstring (NumPy/Google/reST styles)
      - prunes __annotations__ for fixed parameters
      - optionally appends a note mentioning the original function and fixed args

    Parameters
    ----------
    func : callable
        The target callable.
    *args
        Positional args to fix.
    add_fixed_note : bool, optional
        Whether to append a note about fixed arguments. Default True.
    docstr_header : str, optional
        When provided, replace the original docstring header (up to the
        Parameters section) with this text.
    **kwargs
        Keyword args to fix.

    Returns
    -------
    callable
        A partial-like callable with updated docstring and signature.
    """
    p = partial(func, *args, **kwargs)
    update_wrapper(p, func)

    fixed_map   = _bound_fixed_map(func, args, kwargs)
    fixed_names = set(fixed_map)

    # Signature: remove fixed params
    p.__signature__ = _prune_signature(inspect.signature(func), fixed_names)

    # Annotations: drop fixed params (optional but tidy)
    pruned_ann = _prune_annotations(getattr(func, "__annotations__", None), fixed_names)
    if pruned_ann is not None:
        p.__annotations__ = pruned_ann

    # Docstring: remove fixed params docs + append note with original function name
    original_doc = inspect.getdoc(func) or func.__doc__ or ""
    p.__doc__ = _prune_docstring(
        original_doc,
        fixed_names,
        add_note=add_fixed_note,
        fixed_map=fixed_map,
        func_name=getattr(func, "__name__", "<callable>"),
        docstr_header=docstr_header,
    )

    return p
