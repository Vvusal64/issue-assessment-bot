"""Tests for the KnownAppCatalog text-search and lookups."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from bot.catalog import KnownApp, KnownAppCatalog, PackageVariant


def _app(name, package=None, aliases=None, packages=None):
    """Test helper for the new multi-variant ``KnownApp`` constructor."""
    if packages is None:
        if package:
            packages = [PackageVariant(package=package)]
        else:
            packages = []
    return KnownApp(
        name=name,
        aliases=list(aliases or []),
        packages=list(packages),
    )


class CatalogLookupTests(unittest.TestCase):

    def setUp(self) -> None:
        self.cat = KnownAppCatalog([
            _app("Zoom", package="us.zoom.videomeetings",
                 aliases=["Zoom Cloud Meetings", "Zoom Meetings"]),
            _app("Zoom for Cars", aliases=["Zoom Cars"]),
            _app("Spotify", package="com.spotify.music"),
            _app("Deezer", package="deezer.android.app"),
        ])

    def test_find_by_package(self):
        self.assertEqual(
            self.cat.find_by_package("us.zoom.videomeetings").name, "Zoom",
        )
        self.assertIsNone(self.cat.find_by_package("com.unknown.x"))

    def test_find_by_name_includes_aliases(self):
        self.assertEqual(self.cat.find_by_name("zoom").name, "Zoom")
        self.assertEqual(
            self.cat.find_by_name("Zoom Cloud Meetings").name, "Zoom",
        )

    def test_find_in_text_pairs_app_with_nearby_version(self):
        # The BMW-3402 case: lowercase "zoom" + a version on the same line.
        results = self.cat.find_in_text(
            "Error code 16 in zoom 1.0.8.9"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].app.name, "Zoom")
        self.assertEqual(results[0].nearby_version, "1.0.8.9")

    def test_longest_alias_wins(self):
        # "Zoom for Cars 1.0.8.7" must NOT also produce a separate
        # "Zoom" hit — the longer alias should claim the span.
        results = self.cat.find_in_text(
            "Reported on Zoom for Cars 1.0.8.7 yesterday."
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].app.name, "Zoom for Cars")
        self.assertEqual(results[0].nearby_version, "1.0.8.7")

    def test_whole_word_match_does_not_match_substring(self):
        # "zoomtech" must NOT trigger Zoom.
        results = self.cat.find_in_text("our zoomtech tool failed")
        self.assertEqual(results, [])

    def test_word_boundary_works_against_dotted_package(self):
        # "deezer" inside "deezer.android.app" — the dot is a valid
        # word boundary, so this DOES match (and we want it to:
        # detecting the app from a package coordinate is helpful).
        results = self.cat.find_in_text(
            "crash due to deezer.android.app"
        )
        names = [r.app.name for r in results]
        self.assertIn("Deezer", names)

    def test_multiple_distinct_mentions(self):
        results = self.cat.find_in_text(
            "Spotify worked but Zoom 5.0.0 crashed."
        )
        names = [r.app.name for r in results]
        self.assertEqual(set(names), {"Spotify", "Zoom"})

    def test_zoom_for_cars_in_comment_is_not_zoom(self):
        # Specifically exercise the BMW-3402 pattern: the summary has
        # "zoom 1.0.8.9" and a comment mentions "Zoom for Cars 1.0.8.7".
        # Each mention should resolve to the most-specific app entry,
        # NOT collapse to bare "Zoom".
        summary_results = self.cat.find_in_text(
            "Error code 16 in zoom 1.0.8.9"
        )
        comment_results = self.cat.find_in_text(
            "Previously also was reported on Zoom for Cars 1.0.8.7"
        )
        self.assertEqual(len(summary_results), 1)
        self.assertEqual(summary_results[0].app.name, "Zoom")
        self.assertEqual(len(comment_results), 1)
        self.assertEqual(comment_results[0].app.name, "Zoom for Cars")


class CatalogLoadingTests(unittest.TestCase):

    def test_loads_default_seed(self):
        # The shipped JSON includes Zoom and Bild TV.
        cat = KnownAppCatalog.load_default()
        self.assertIsNotNone(cat.find_by_name("Zoom"))
        self.assertIsNotNone(cat.find_by_name("Bild TV"))
        self.assertIsNotNone(cat.find_by_name("Deezer"))

    def test_missing_file_returns_empty_catalog(self):
        cat = KnownAppCatalog.load("/nonexistent/path.json")
        self.assertEqual(cat.apps, [])

    def test_load_from_temp_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "apps": [
                    {"name": "MyApp", "package": "com.my.app",
                     "aliases": ["My Application"]},
                ]
            }, f)
        cat = KnownAppCatalog.load(path)
        self.assertEqual(cat.find_by_name("My Application").name, "MyApp")
        self.assertEqual(cat.find_by_package("com.my.app").name, "MyApp")


class MultiVariantPickPackageTests(unittest.TestCase):
    """Verify that ``KnownApp.pick_package(version)`` resolves
    ambiguous app families (Zoom, …) to the right package coordinate
    using version_prefixes.
    """

    def setUp(self) -> None:
        # Mirror the shipped Zoom entry exactly — three variants.
        self.zoom = KnownApp(
            name="Zoom",
            aliases=[],
            packages=[
                PackageVariant("com.forvia.zoomapp",
                               version_prefixes=["1.0.", "1.1."]),
                PackageVariant("us.zoom.videomeetings",
                               version_prefixes=["5."]),
                PackageVariant("us.zoom.mercedesbenz",
                               version_prefixes=["6.7.", "7.0."]),
            ],
        )

    def test_forvia_chosen_for_v1_0_8_9(self):
        # The BMW-3402 case.
        self.assertEqual(
            self.zoom.pick_package("1.0.8.9"), "com.forvia.zoomapp",
        )

    def test_videomeetings_chosen_for_v5_6_0_1592(self):
        self.assertEqual(
            self.zoom.pick_package("5.6.0.1592"), "us.zoom.videomeetings",
        )

    def test_mercedesbenz_chosen_for_v6_7_2(self):
        self.assertEqual(
            self.zoom.pick_package("6.7.2"), "us.zoom.mercedesbenz",
        )

    def test_no_version_returns_none_when_ambiguous(self):
        # Without a version we refuse to guess between three variants.
        self.assertIsNone(self.zoom.pick_package(None))

    def test_unknown_version_prefix_returns_none(self):
        # ``9.9.9`` doesn't match any registered prefix.
        self.assertIsNone(self.zoom.pick_package("9.9.9"))

    def test_single_variant_app_always_resolves(self):
        spotify = KnownApp(
            name="Spotify",
            packages=[PackageVariant("com.spotify.music")],
        )
        # Even with no version, single-variant resolves cleanly.
        self.assertEqual(
            spotify.pick_package(None), "com.spotify.music",
        )
        self.assertEqual(
            spotify.pick_package("8.5.6"), "com.spotify.music",
        )

    def test_default_variant_takes_over_when_no_prefix_matches(self):
        # An entry with one default (empty prefixes) variant + others.
        app = KnownApp(
            name="MultiApp",
            packages=[
                PackageVariant("com.specific.beta",
                               version_prefixes=["2."]),
                PackageVariant("com.specific.fallback"),  # default
            ],
        )
        # Version matches the specific variant.
        self.assertEqual(app.pick_package("2.0.1"), "com.specific.beta")
        # Version doesn't match → the default variant takes over.
        self.assertEqual(app.pick_package("9.0.0"), "com.specific.fallback")
        # No version → also goes to default.
        self.assertEqual(app.pick_package(None), "com.specific.fallback")

    def test_resolved_package_via_catalog_match(self):
        # The CatalogMatch convenience accessor uses pick_package.
        cat = KnownAppCatalog([self.zoom])
        # Summary-style mention: "zoom 1.0.8.9".
        results = cat.find_in_text("crash in zoom 1.0.8.9 today")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].nearby_version, "1.0.8.9")
        self.assertEqual(
            results[0].resolved_package, "com.forvia.zoomapp",
        )


if __name__ == "__main__":
    unittest.main()
