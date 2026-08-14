"""The corpus graph: what becomes an edge, and — the part with teeth — what does not.

`dependencies` and `neededBy` are never model output and never a guess, because a wrong edge
silently changes a removal rating downstream and there is no signal that it happened. So most
of this file is negative: an optional library, a declared package query and an off-corpus
overlay target all have to come out as evidence, and none of them may appear in an edge list.
"""

from uadclaw.corpus import (
    EDGE_LIBRARY,
    EDGE_OVERLAY,
    CorpusPackage,
    build_graph,
    library_providers,
)


def graph_of(*corpus: CorpusPackage, platform: frozenset[str] = frozenset()):
    return build_graph(list(corpus), platform_libraries=platform)


# --- the two emitted classes ----------------------------------------------------------------


def test_an_overlay_becomes_an_edge_in_both_directions():
    graph = graph_of(
        CorpusPackage(package="com.android.settings"),
        CorpusPackage(
            package="com.android.settings.overlay.oriole", overlay_target="com.android.settings"
        ),
    )

    assert graph.dependencies_of("com.android.settings.overlay.oriole") == ("com.android.settings",)
    assert graph.needed_by("com.android.settings") == ("com.android.settings.overlay.oriole",)
    assert [edge.kind for edge in graph.edges] == [EDGE_OVERLAY]


def test_a_required_library_becomes_an_edge_from_consumer_to_provider():
    graph = graph_of(
        CorpusPackage(package="com.vendor.lib", libraries=("com.vendor.support",)),
        CorpusPackage(package="com.vendor.app", uses_libraries_required=("com.vendor.support",)),
    )

    assert graph.dependencies_of("com.vendor.app") == ("com.vendor.lib",)
    assert graph.needed_by("com.vendor.lib") == ("com.vendor.app",)
    assert [edge.detail for edge in graph.edges] == ["com.vendor.support"]
    assert [edge.kind for edge in graph.edges] == [EDGE_LIBRARY]


def test_a_static_library_provider_counts_as_a_provider():
    graph = graph_of(
        CorpusPackage(package="com.vendor.lib", static_libraries=("com.vendor.support",)),
        CorpusPackage(package="com.vendor.app", uses_libraries_required=("com.vendor.support",)),
    )

    assert graph.dependencies_of("com.vendor.app") == ("com.vendor.lib",)


def test_a_static_library_consumer_is_an_edge_in_both_directions():
    """The consumer half of the pair, on the graph side: a package whose required bucket holds
    a static library's name — populated by the `<uses-static-library>` parse — gets the same
    edge a `required="true"` consumer gets, against a static provider, in both directions."""
    graph = graph_of(
        CorpusPackage(
            package="com.google.android.trichromelibrary",
            static_libraries=("com.google.android.trichromelibrary",),
        ),
        CorpusPackage(
            package="com.android.chrome",
            uses_libraries_required=("com.google.android.trichromelibrary",),
        ),
    )

    assert graph.dependencies_of("com.android.chrome") == ("com.google.android.trichromelibrary",)
    assert graph.needed_by("com.google.android.trichromelibrary") == ("com.android.chrome",)
    assert [edge.detail for edge in graph.edges] == ["com.google.android.trichromelibrary"]
    assert [edge.kind for edge in graph.edges] == [EDGE_LIBRARY]


def test_library_providers_lists_every_declarer():
    corpus = [
        CorpusPackage(package="com.a", libraries=("shared",)),
        CorpusPackage(package="com.b", static_libraries=("shared",)),
    ]

    assert library_providers(corpus) == {"shared": ("com.a", "com.b")}


# --- what must never become an edge ----------------------------------------------------------


def test_an_optional_uses_library_is_not_an_edge():
    """`required="false"` documents the consumer as designed to work without the library. An
    edge here would pin a provider `Unsafe` for a dependent that does not have one."""
    graph = graph_of(
        CorpusPackage(package="com.vendor.lib", libraries=("com.vendor.support",)),
        CorpusPackage(package="com.vendor.app", uses_libraries_optional=("com.vendor.support",)),
    )

    assert graph.edges == ()
    assert graph.dependencies_of("com.vendor.app") == ()


def test_a_declared_package_query_is_evidence_and_never_an_edge():
    graph = graph_of(
        CorpusPackage(package="com.example.maps", queries_packages=("com.example.car", "com.gone")),
        CorpusPackage(package="com.example.car"),
    )
    evidence = graph.evidence["com.example.maps"]

    assert graph.edges == ()
    assert graph.dependencies_of("com.example.maps") == ()
    assert evidence.queries_packages_in_corpus == ("com.example.car",)
    assert evidence.queries_packages_absent == ("com.gone",)


def test_a_required_library_the_platform_provides_is_evidence_and_never_an_edge():
    """Measured on both corpora: every unresolved `uses-library` name is a Java shared library
    declared in `/etc/permissions/*.xml`, so its provider is the platform rather than a
    removable package and the edge genuinely does not exist."""
    graph = graph_of(
        CorpusPackage(
            package="com.google.android.dialer",
            uses_libraries_required=("com.google.android.dialer.support", "com.absent.library"),
        ),
        platform=frozenset({"com.google.android.dialer.support"}),
    )
    evidence = graph.evidence["com.google.android.dialer"]

    assert graph.edges == ()
    assert evidence.required_libraries_from_platform == ("com.google.android.dialer.support",)
    assert evidence.required_libraries_unresolved == ("com.absent.library",)


def test_an_overlay_pointing_outside_the_corpus_records_the_target_and_emits_nothing():
    graph = graph_of(CorpusPackage(package="com.example.overlay", overlay_target="com.absent.app"))

    assert graph.edges == ()
    assert graph.evidence["com.example.overlay"].overlay_target_absent == "com.absent.app"


def test_a_static_library_consumer_with_no_provider_in_corpus_is_evidence():
    """Folding the static consumer into the existing bucket means the evidence split fires
    unchanged: no in-corpus provider, no edge, and the name lands exactly where an unresolved
    `required="true"` name lands."""
    graph = graph_of(
        CorpusPackage(
            package="com.android.chrome",
            uses_libraries_required=(
                "com.google.android.trichromelibrary",
                "com.absent.library",
            ),
        ),
        platform=frozenset({"com.google.android.trichromelibrary"}),
    )

    assert graph.edges == ()
    evidence = graph.evidence["com.android.chrome"]
    assert evidence.required_libraries_from_platform == ("com.google.android.trichromelibrary",)
    assert evidence.required_libraries_unresolved == ("com.absent.library",)


def test_a_package_never_becomes_its_own_dependency():
    """`com.google.android.trichromelibrary` declares a static library under its own name on
    both corpora, and a self-edge would make it its own `neededBy`."""
    graph = graph_of(
        CorpusPackage(
            package="com.google.android.trichromelibrary",
            static_libraries=("com.google.android.trichromelibrary",),
            uses_libraries_required=("com.google.android.trichromelibrary",),
        ),
        CorpusPackage(package="com.example.overlay", overlay_target="com.example.overlay"),
    )

    assert graph.edges == ()
    assert graph.dependencies_of("com.google.android.trichromelibrary") == ()
    assert graph.dependencies_of("com.example.overlay") == ()


def test_the_content_uri_class_is_absent_rather_than_empty():
    """§4's third relation (dex `content://` strings against declared authorities) is not
    implemented: the APKs are deleted once their facts are stored, so there is nothing to
    scan. The key is absent on purpose — an empty list would claim it was looked for."""
    evidence = graph_of(
        CorpusPackage(package="com.example.app", provider_authorities=("com.example.provider",))
    ).evidence["com.example.app"]

    assert "content_uri_references" not in evidence.as_json()


def test_two_packages_declaring_one_authority_are_flagged_for_the_reviewer():
    graph = graph_of(
        CorpusPackage(package="com.a", provider_authorities=("shared.authority",)),
        CorpusPackage(package="com.b", provider_authorities=("shared.authority",)),
        CorpusPackage(package="com.c", provider_authorities=("own.authority",)),
    )

    assert graph.evidence["com.a"].shared_provider_authorities == ("shared.authority",)
    assert graph.evidence["com.b"].shared_provider_authorities == ("shared.authority",)
    assert graph.evidence["com.c"].shared_provider_authorities == ()
    assert graph.edges == ()


# --- shape ------------------------------------------------------------------------------------


def test_a_library_with_two_providers_yields_an_edge_to_each():
    graph = graph_of(
        CorpusPackage(package="com.vendor.lib.a", libraries=("shared",)),
        CorpusPackage(package="com.vendor.lib.b", libraries=("shared",)),
        CorpusPackage(package="com.vendor.app", uses_libraries_required=("shared",)),
    )

    assert graph.dependencies_of("com.vendor.app") == ("com.vendor.lib.a", "com.vendor.lib.b")


def test_the_corpus_order_does_not_change_the_graph():
    corpus = [
        CorpusPackage(package="com.vendor.lib", libraries=("shared",)),
        CorpusPackage(package="com.vendor.app", uses_libraries_required=("shared",)),
        CorpusPackage(package="com.vendor.app.overlay", overlay_target="com.vendor.app"),
        CorpusPackage(package="com.vendor.other", queries_packages=("com.vendor.app",)),
    ]

    forward = build_graph(corpus)
    backward = build_graph(list(reversed(corpus)))

    assert [edge.as_json() for edge in forward.edges] == [edge.as_json() for edge in backward.edges]
    assert {name: item.as_json() for name, item in forward.evidence.items()} == {
        name: item.as_json() for name, item in backward.evidence.items()
    }


def test_counts_by_kind_reports_both_classes_even_at_zero():
    graph = graph_of(
        CorpusPackage(package="com.example.app"),
        CorpusPackage(package="com.example.overlay", overlay_target="com.example.app"),
    )

    assert graph.counts_by_kind() == {EDGE_OVERLAY: 1, EDGE_LIBRARY: 0}
