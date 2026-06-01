"""Tests for proteopy.utils.hash."""
import hashlib

import pytest

from proteopy.utils.hash import md5_hash, md5_hash_list


class TestMd5Hash:
    """Tests for :func:`md5_hash`."""

    def test_full_digest_matches_hashlib(self):
        """Default call returns the canonical 32-char MD5 hex digest."""
        text = "hello"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert md5_hash(text) == expected
        assert len(md5_hash(text)) == 32

    def test_n_chars_truncates(self):
        """``n_chars=7`` returns the first seven hex characters."""
        assert md5_hash("hello", n_chars=7) == "5d41402"
        assert len(md5_hash("hello", n_chars=7)) == 7

    def test_deterministic(self):
        """Repeated calls return identical digests."""
        assert md5_hash("abc") == md5_hash("abc")
        assert md5_hash("abc", n_chars=7) == md5_hash("abc", n_chars=7)

    def test_empty_string_supported(self):
        """Empty string hashes to the canonical empty-string MD5."""
        assert md5_hash("") == "d41d8cd98f00b204e9800998ecf8427e"

    def test_n_chars_none_returns_full(self):
        """``n_chars=None`` is equivalent to omitting the argument."""
        assert md5_hash("abc", n_chars=None) == md5_hash("abc")

    @pytest.mark.parametrize(
        "bad_text",
        [123, 1.5, None, b"bytes", ["a"], ("a",)],
    )
    def test_non_string_text_raises(self, bad_text):
        """Non-string input is rejected."""
        with pytest.raises(ValueError, match=r"text must be a str"):
            md5_hash(bad_text)

    @pytest.mark.parametrize(
        "bad_n",
        [0, -1, 33, 100, 1.5, "7", True],
    )
    def test_invalid_n_chars_raises(self, bad_n):
        """``n_chars`` outside [1, 32] or non-int raises."""
        with pytest.raises(
            ValueError, match=r"n_chars must be an int in \[1, 32\]",
        ):
            md5_hash("abc", n_chars=bad_n)


class TestMd5HashList:
    """Tests for :func:`md5_hash_list`."""

    def test_default_separator_matches_joined_md5(self):
        """Default separator ``;`` matches ``md5_hash(';'.join(...))``."""
        items = ["a", "b", "c"]
        assert md5_hash_list(items) == md5_hash("a;b;c")

    def test_n_chars_propagates(self):
        """``n_chars`` is forwarded to the underlying ``md5_hash``."""
        items = ["a", "b"]
        assert md5_hash_list(items, n_chars=7) == md5_hash(
            "a;b", n_chars=7,
        )
        assert len(md5_hash_list(items, n_chars=7)) == 7

    def test_custom_separator_changes_digest(self):
        """Different separators produce different digests."""
        items = ["a", "b"]
        assert md5_hash_list(items, sep=";") != md5_hash_list(
            items, sep="|",
        )

    def test_order_sensitive(self):
        """Reordered items produce different digests."""
        assert md5_hash_list(["a", "b"]) != md5_hash_list(["b", "a"])

    def test_non_string_items_coerced(self):
        """Non-string items are coerced via ``str(...)``."""
        assert md5_hash_list([1, 2]) == md5_hash("1;2")

    def test_deterministic(self):
        """Repeated calls return identical digests."""
        items = ["x", "y", "z"]
        assert md5_hash_list(items, n_chars=7) == md5_hash_list(
            items, n_chars=7,
        )

    def test_empty_iterable_hashes_empty_string(self):
        """An empty iterable hashes the empty string."""
        assert md5_hash_list([]) == md5_hash("")

    def test_tuple_input_supported(self):
        """Tuples are treated like lists."""
        assert md5_hash_list(("a", "b")) == md5_hash_list(["a", "b"])

    def test_string_input_rejected(self):
        """Bare strings are rejected to prevent per-character hashing."""
        with pytest.raises(
            ValueError, match=r"non-string iterable",
        ):
            md5_hash_list("abc")

    def test_non_iterable_rejected(self):
        """Non-iterable input (e.g. int) is rejected."""
        with pytest.raises(
            ValueError, match=r"non-string iterable",
        ):
            md5_hash_list(42)

    def test_non_string_sep_rejected(self):
        """Non-string ``sep`` is rejected."""
        with pytest.raises(ValueError, match=r"sep must be a str"):
            md5_hash_list(["a", "b"], sep=1)
