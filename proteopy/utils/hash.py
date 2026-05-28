from __future__ import annotations

import hashlib
from collections.abc import Iterable


def md5_hash(text: str, *, n_chars: int | None = None) -> str:
    """
    Compute the MD5 hex digest of a string, optionally truncated.

    Parameters
    ----------
    text : str
        The input string to hash. UTF-8 encoded internally.
    n_chars : int, optional
        If set, return only the first ``n_chars`` of the lowercase
        hex digest. Must lie in ``[1, 32]`` (32 is the full digest
        length).

    Returns
    -------
    str
        Lowercase MD5 hex digest, or its first ``n_chars`` characters.

    Examples
    --------
    >>> md5_hash("hello")
    '5d41402abc4b2a76b9719d911017c592'
    >>> md5_hash("hello", n_chars=7)
    '5d41402'
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a str; got {type(text).__name__}.")
    if n_chars is not None:
        if (
            not isinstance(n_chars, int)
            or isinstance(n_chars, bool)
            or n_chars < 1
            or n_chars > 32
        ):
            raise ValueError(
                f"n_chars must be an int in [1, 32]; got {n_chars!r}."
            )

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    if n_chars is None:
        return digest
    return digest[:n_chars]


def md5_hash_list(
    items: Iterable,
    *,
    sep: str = ";",
    n_chars: int | None = None,
) -> str:
    """
    Compute the MD5 hex digest of an iterable, joined into a string.

    Each item in ``items`` is coerced via ``str(...)`` and joined
    using ``sep``. The resulting string is passed to :func:`md5_hash`.

    Parameters
    ----------
    items : iterable
        Sequence of items to hash. A bare ``str`` is rejected to
        avoid the common bug of accidentally hashing characters
        individually.
    sep : str
        Separator used when joining items into a single string.
    n_chars : int, optional
        See :func:`md5_hash`.

    Returns
    -------
    str
        MD5 hex digest of the joined string (optionally truncated).

    Examples
    --------
    >>> md5_hash_list(["a", "b"], n_chars=7)
    '530a18b'
    >>> md5_hash_list(["a", "b"]) == md5_hash("a;b")
    True
    """
    if isinstance(items, str) or not hasattr(items, "__iter__"):
        raise ValueError(
            "items must be a non-string iterable; got "
            f"{type(items).__name__}."
        )
    if not isinstance(sep, str):
        raise ValueError(f"sep must be a str; got {type(sep).__name__}.")

    text = sep.join(str(x) for x in items)
    return md5_hash(text, n_chars=n_chars)
