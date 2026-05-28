"""Tests for the pluggable app-name resolution layer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from bot.app_name_resolver import (
    AppNameResolver,
    ChainedResolver,
    JsonFileAppNameResolver,
)


class JsonFileAppNameResolverTests(unittest.TestCase):

    def _write_map(self, mapping: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)
        self.addCleanup(os.unlink, path)
        return path

    def test_known_package_resolves(self) -> None:
        path = self._write_map({"deezer.android.app": "Deezer"})
        r = JsonFileAppNameResolver(path)
        self.assertEqual(r.resolve("deezer.android.app"), "Deezer")

    def test_unknown_package_returns_none(self) -> None:
        path = self._write_map({"deezer.android.app": "Deezer"})
        r = JsonFileAppNameResolver(path)
        self.assertIsNone(r.resolve("com.unknown.app"))

    def test_empty_input_returns_none(self) -> None:
        path = self._write_map({"deezer.android.app": "Deezer"})
        r = JsonFileAppNameResolver(path)
        self.assertIsNone(r.resolve(""))

    def test_missing_file_does_not_raise(self) -> None:
        r = JsonFileAppNameResolver("/nonexistent/path.json")
        self.assertIsNone(r.resolve("anything"))

    def test_corrupt_file_does_not_raise(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{this is not json")
        r = JsonFileAppNameResolver(path)
        self.assertIsNone(r.resolve("anything"))

    def test_reload_picks_up_edits(self) -> None:
        path = self._write_map({"a.b.c": "First"})
        r = JsonFileAppNameResolver(path)
        self.assertEqual(r.resolve("a.b.c"), "First")
        # Edit the file under it.
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"a.b.c": "Second"}, f)
        # Without reload, cached value is returned.
        self.assertEqual(r.resolve("a.b.c"), "First")
        r.reload()
        self.assertEqual(r.resolve("a.b.c"), "Second")

    def test_default_seed_includes_user_examples(self) -> None:
        # Sanity-check the shipped JSON file.
        r = JsonFileAppNameResolver()
        self.assertEqual(r.resolve("deezer.android.app"), "Deezer")
        self.assertEqual(r.resolve("eimycarolaym.blanco"), "Blanco")
        self.assertEqual(
            r.resolve("com.smokoko.careatscar3_appning"), "Car Eats Car 3",
        )


class ChainedResolverTests(unittest.TestCase):

    class _Static(AppNameResolver):
        def __init__(self, mapping):
            self.mapping = mapping
            self.calls = 0

        def resolve(self, package_name):
            self.calls += 1
            return self.mapping.get(package_name)

    class _Boom(AppNameResolver):
        def resolve(self, package_name):
            raise RuntimeError("backend exploded")

    def test_returns_first_non_empty_hit(self) -> None:
        a = self._Static({"x": "A"})
        b = self._Static({"x": "B", "y": "Y"})
        chain = ChainedResolver(a, b)
        self.assertEqual(chain.resolve("x"), "A")
        self.assertEqual(b.calls, 0)
        self.assertEqual(chain.resolve("y"), "Y")

    def test_falls_through_on_none(self) -> None:
        a = self._Static({})
        b = self._Static({"y": "Y"})
        chain = ChainedResolver(a, b)
        self.assertEqual(chain.resolve("y"), "Y")

    def test_failing_resolver_does_not_break_chain(self) -> None:
        chain = ChainedResolver(self._Boom(), self._Static({"z": "Z"}))
        # The broken one is skipped; we still get the hit from the next.
        self.assertEqual(chain.resolve("z"), "Z")


if __name__ == "__main__":
    unittest.main()
