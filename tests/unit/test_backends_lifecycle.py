"""Backends builds each backend on first use and frees it on release.

The point of the laziness is memory: pipeline.run uses the five backends
one stage at a time, and holding all of them pinned ~12GB of weights for a
whole episode, which pushed a 16GB machine into swap mid-synthesis. These
tests pin the two properties that behaviour depends on -- nothing is built
until it's asked for, and releasing actually drops the reference -- plus
the rebuild-on-reaccess guarantee that makes release() safe to call
without having to prove no later stage needs the backend again.
"""
from __future__ import annotations

import pytest

from dubtool.pipeline import Backends


class Spy:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def _counting_factory(tag: str, calls: list[str]):
    def build() -> Spy:
        calls.append(tag)
        return Spy(tag)
    return build


def _backends(calls: list[str]) -> Backends:
    return Backends(**{
        name: _counting_factory(name, calls)
        for name in ("separator", "diarizer", "transcriber", "translator", "tts")
    })


def test_builds_nothing_until_a_backend_is_accessed():
    calls: list[str] = []
    _backends(calls)
    assert calls == []


def test_builds_only_the_backend_asked_for():
    calls: list[str] = []
    backends = _backends(calls)
    backends.separator
    assert calls == ["separator"]


def test_caches_so_repeated_access_does_not_rebuild():
    calls: list[str] = []
    backends = _backends(calls)
    first = backends.tts
    assert backends.tts is first
    assert calls == ["tts"]


def test_release_drops_the_instance_and_reaccess_rebuilds():
    calls: list[str] = []
    backends = _backends(calls)
    first = backends.translator
    backends.release("translator")
    second = backends.translator
    assert second is not first
    assert calls == ["translator", "translator"]


def test_release_is_a_noop_for_a_backend_never_built():
    calls: list[str] = []
    backends = _backends(calls)
    backends.release("separator", "diarizer")
    assert calls == []


def test_release_rejects_an_unknown_name_rather_than_silently_passing():
    # A typo here would otherwise look like a successful free, and the
    # memory it was supposed to reclaim would stay held for the whole run.
    backends = _backends([])
    with pytest.raises(ValueError, match="unknown backend"):
        backends.release("transcriber", "nonsense")


def test_of_wraps_ready_instances_without_calling_a_factory():
    instances = {
        name: Spy(name)
        for name in ("separator", "diarizer", "transcriber", "translator", "tts")
    }
    backends = Backends.of(**instances)
    assert backends.separator is instances["separator"]
    assert backends.tts is instances["tts"]


def test_of_keeps_handing_back_the_caller_owned_instance_after_release():
    # The caller owns these, so dropping our reference can't free them --
    # and handing back a *different* object later would be worse than
    # useless. release() must not change what `of()` resolves to.
    instances = {
        name: Spy(name)
        for name in ("separator", "diarizer", "transcriber", "translator", "tts")
    }
    backends = Backends.of(**instances)
    original = backends.tts
    backends.release("tts")
    assert backends.tts is original
