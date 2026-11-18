"""
Spec-guided TDD support for pytest (soft workflow).

Scope: only test modules marked with ``pytestmark = pytest.mark.spec_guided``
are affected. All other tests in the suite run exactly as without this file,
in both modes, so the support conftest can live next to an existing suite.

Naming is the only link between tests and the (unpushed) specs:

* Every test is named ``test_<ID>_<short_name>``, where ``<ID>`` is the test ID
  from the test spec (e.g. T10, P1). No spec files or requirement markers are
  referenced from code.
* Holdout tests carry the clean default names and live in ``Test<X>``, first
  in the module. Each has an implementer counterpart with the same name plus
  the suffix ``_IMP`` in the class ``Test<X>IMP`` after it: the mirror with
  different inputs.
* Randomized tests (those using the ``rng`` fixture, i.e. property and
  metamorphic tests) have no ``_IMP`` counterpart: fresh random inputs cannot
  be memorized. They live in ``Test<X>`` and run in both modes.

Behavior in marked modules:

* ``pytest``                human mode (default), behaves like any pytest run:
                            checks that holdout and ``_IMP`` tests pair up,
                            then runs the holdout, property and metamorphic
                            tests with full pytest detail.
* ``pytest --implementer``  implementer mode: runs the ``_IMP``, property and
                            metamorphic tests with obscured errors: test ID,
                            kind of failure, and for randomized tests the
                            failing input registered via ``report_input``. No
                            tracebacks, source, captured output or warning
                            locations are shown; ``--pdb`` / ``--trace`` are
                            refused.
* ``--seed N``              reproduces the random inputs (seed in the header).

Soft workflow: nothing here prevents an agent from reading the tests.
"""

from __future__ import annotations

import random
import re

import pytest
from _pytest.outcomes import Failed

_MARKER = "spec_guided"
_SEED_KEY = pytest.StashKey[int]()
_SUFFIX = "_IMP"
_CLASS_SUFFIX = "IMP"
_NAME = re.compile(rf"^test_([A-Z]+\d+)_\w+?({_SUFFIX})?$")
_INPUT_ATTR = "_sgtdd_failing_input"


# --------------------------------------------------------------------------
# Options, configuration and header
# --------------------------------------------------------------------------


def pytest_addoption(parser):
    group = parser.getgroup("spec-guided-tdd")
    group.addoption(
        "--implementer",
        action="store_true",
        default=False,
        help=(
            "Implementer mode: run _IMP and randomized tests with "
            "obscured errors (no holdout tests)."
        ),
    )
    group.addoption(
        "--seed",
        type=int,
        default=None,
        help="Session seed for the rng fixture (default: random).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{_MARKER}: module follows the spec-guided TDD layout "
        "(holdout/_IMP pairs).",
    )
    seed = config.getoption("--seed")
    config.stash[_SEED_KEY] = (
        seed if seed is not None else random.SystemRandom().randrange(2**32)
    )

    if config.getoption("--implementer"):
        if config.getoption("usepdb", False) or config.getoption(
            "trace", False
        ):
            raise pytest.UsageError(
                "--pdb and --trace are not available with --implementer"
            )
        # The warnings summary prints the test source line that triggered a
        # warning, which would reveal test inputs. pytest.warns still works.
        config.addinivalue_line("filterwarnings", "ignore")


def pytest_report_header(config):
    mode = "implementer" if config.getoption("--implementer") else "human"
    return f"spec-guided-tdd: mode={mode} seed={config.stash[_SEED_KEY]}"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _base_name(item) -> str:
    return getattr(item, "originalname", item.name)


def _test_id(item) -> str:
    m = _NAME.match(_base_name(item))
    return m.group(1) if m else _base_name(item)


def _is_spec_guided(item) -> bool:
    return item.get_closest_marker(_MARKER) is not None


def _is_imp(item) -> bool:
    return _base_name(item).endswith(_SUFFIX)


def _looks_like_imp(item) -> bool:
    cls = getattr(item, "cls", None)
    return _is_imp(item) or (
        cls is not None and cls.__name__.endswith(_CLASS_SUFFIX)
    )


def _is_randomized(item) -> bool:
    return "rng" in getattr(item, "fixturenames", ())


# --------------------------------------------------------------------------
# Collection: scope, naming check, pairing check, selection
# --------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    implementer = config.getoption("--implementer")
    guided = [it for it in items if _is_spec_guided(it)]
    _check_unmarked(items)
    _check_names(guided)
    if not implementer:
        _check_pairs(guided)

    def wanted(it) -> bool:
        if implementer:
            return _is_imp(it) or _is_randomized(it)
        return not _is_imp(it)

    keep = [it for it in items if not _is_spec_guided(it) or wanted(it)]
    drop = [it for it in items if _is_spec_guided(it) and not wanted(it)]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def _check_unmarked(items) -> None:
    # A forgotten marker would run holdout tests unobscured in implementer
    # mode; the _IMP counterparts in the same module give it away.
    errors = [
        f"{it.nodeid}: {_SUFFIX}-style test outside a module marked "
        f"'pytestmark = pytest.mark.{_MARKER}'"
        for it in items
        if not _is_spec_guided(it) and _looks_like_imp(it)
    ]
    if errors:
        raise pytest.UsageError(
            "Spec-guided scope errors:\n  " + "\n  ".join(errors)
        )


def _check_names(items) -> None:
    errors: list[str] = []
    for item in items:
        name = _base_name(item)
        if not _NAME.match(name):
            errors.append(
                f"{item.nodeid}: name must be "
                f"test_<ID>_<short_name>[{_SUFFIX}], e.g. test_T10_empty"
            )
            continue
        if item.cls is None:
            errors.append(
                f"{item.nodeid}: tests must live in a Test<X> or "
                f"Test<X>{_CLASS_SUFFIX} class"
            )
            continue
        in_imp_class = item.cls.__name__.endswith(_CLASS_SUFFIX)
        if _is_imp(item) != in_imp_class:
            errors.append(
                f"{item.nodeid}: *{_SUFFIX} tests belong in "
                f"Test<X>{_CLASS_SUFFIX}, all others in Test<X>"
            )
        if _is_imp(item) and _is_randomized(item):
            errors.append(
                f"{item.nodeid}: randomized tests (using rng) have no "
                f"{_SUFFIX} counterpart"
            )
    if errors:
        raise pytest.UsageError(
            "Test naming errors:\n  " + "\n  ".join(errors)
        )


def _check_pairs(items) -> None:
    hold: set[tuple] = set()
    imp: set[tuple] = set()
    for item in items:
        cls = item.cls.__name__
        name = _base_name(item)
        if _is_imp(item):
            imp.add(
                (
                    str(item.path),
                    cls[: -len(_CLASS_SUFFIX)],
                    name[: -len(_SUFFIX)],
                )
            )
        elif not _is_randomized(item):
            hold.add((str(item.path), cls, name))

    errors = [
        f"{c}.{n}: no counterpart {c}{_CLASS_SUFFIX}.{n}{_SUFFIX}"
        for _, c, n in sorted(hold - imp)
    ]
    errors += [
        f"{c}{_CLASS_SUFFIX}.{n}{_SUFFIX}: no holdout counterpart {c}.{n}"
        for _, c, n in sorted(imp - hold)
    ]
    if errors:
        raise pytest.UsageError(
            "Holdout pairing errors:\n  " + "\n  ".join(errors)
        )


# --------------------------------------------------------------------------
# Obscured errors in implementer mode
# --------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not item.config.getoption("--implementer") or not _is_spec_guided(item):
        return
    # Never show captured output of guided tests, passed or failed.
    report.sections = []
    if not report.failed:
        return

    test_id = _test_id(item)
    excinfo = call.excinfo
    if report.when != "call":
        line = f"{test_id}: error outside the check (test {report.when})"
    elif excinfo is not None and not excinfo.errisinstance(
        (AssertionError, Failed)
    ):
        line = f"{test_id}: unexpected exception ({excinfo.type.__name__})"
    else:
        line = f"{test_id}: wrong result"

    if hasattr(item, _INPUT_ATTR):
        line += f"; failing input: {getattr(item, _INPUT_ATTR)}"

    report.longrepr = line


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def rng(request) -> random.Random:
    """Per-test random generator derived from the session seed.

    Using this fixture marks a test as randomized: it needs no _IMP
    counterpart.
    """
    seed = request.config.stash[_SEED_KEY]
    return random.Random(f"{seed}:{request.node.nodeid}")


@pytest.fixture
def report_input(request):
    """Register the current generated input. Use ONLY in randomized tests.

    Call it right before checking each generated input; on failure the last
    registered input is shown.
    """

    def _register(value) -> None:
        setattr(request.node, _INPUT_ATTR, repr(value))

    return _register
