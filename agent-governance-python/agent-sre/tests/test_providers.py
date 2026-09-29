# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the community fallbacks in agent_sre.providers."""

import pytest

from agent_sre import providers


@pytest.fixture(autouse=True)
def no_advanced_provider(monkeypatch):
    monkeypatch.setattr(providers, "_discover_provider", lambda group: None)


@pytest.mark.parametrize(
    "getter, group, missing",
    [
        (
            "get_slo_detector",
            "agent_sre.providers.slo_detection",
            "agent_sre.slo.detector.SLODetector",
        ),
        (
            "get_chaos_engine",
            "agent_sre.providers.chaos_engine",
            "agent_sre.chaos.engine.ChaosEngine",
        ),
    ],
)
def test_getter_without_community_implementation_raises_clear_error(getter, group, missing):
    with pytest.raises(NotImplementedError) as exc_info:
        getattr(providers, getter)()
    assert str(exc_info.value) == (
        f"{getter}() has no community implementation ({missing} does not exist). "
        f"Install a provider package that registers an entry point in the '{group}' group."
    )


def test_list_providers_marks_slots_without_community_implementation():
    assert providers.list_providers() == {
        "slo_detection": "unavailable",
        "replay_engine": "community",
        "chaos_engine": "unavailable",
        "cost_optimizer": "community",
        "delivery": "community",
        "incident": "community",
    }


def test_list_providers_reports_advanced_provider_for_slot_without_community_implementation(
    monkeypatch,
):
    monkeypatch.setattr(
        providers,
        "_discover_provider",
        lambda group: object if group == "agent_sre.providers.chaos_engine" else None,
    )
    assert providers.list_providers()["chaos_engine"] == "advanced"
