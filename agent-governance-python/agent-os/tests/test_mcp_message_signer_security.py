# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Adversarial regression tests for the MCP signed-envelope boundary."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta, timezone
from itertools import product
from threading import Barrier

import pytest

from agent_os.mcp_message_signer import MCPMessageSigner, MCPSignedEnvelope
from agent_os.mcp_protocols import DuplicateNonceError, InMemoryNonceStore

KEY = b"k" * 32
NOW = datetime(2026, 9, 24, 10, 0, 0, 123456, tzinfo=UTC)


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr("agent_os.mcp_message_signer._utcnow", lambda: NOW)

    def nonce_store(**kwargs):
        return InMemoryNonceStore(clock=lambda: NOW, **kwargs)

    monkeypatch.setattr("agent_os.mcp_message_signer.InMemoryNonceStore", nonce_store)


@pytest.mark.parametrize(
    ("sender", "payload", "forged_sender", "forged_payload"),
    [
        ("alice", "alpha|beta", "alice|alpha", "beta"),
        ("alice|INJECTED", "x", "alice", "INJECTED|x"),
        (None, "p", "", "p"),
        ("", "p", None, "p"),
        ("alice", '|{"method":"tools/call"}', "alice|", '{"method":"tools/call"}'),
    ],
)
def test_reframed_envelope_fails_without_consuming_nonce(
    sender, payload, forged_sender, forged_payload
):
    signed = MCPMessageSigner(KEY).sign_message(payload, sender_id=sender)
    forged = replace(signed, sender_id=forged_sender, payload=forged_payload)
    receiver = MCPMessageSigner(KEY)

    result = receiver.verify_message(forged)

    assert not result.is_valid
    assert result.failure_reason == "Invalid signature."
    assert result.payload is None
    assert result.sender_id is None
    assert receiver.cached_nonce_count == 0
    assert receiver.verify_message(signed).is_valid
    assert not receiver.verify_message(signed).is_valid


def test_nonce_timestamp_boundary_cannot_be_reframed(fixed_clock):
    later = NOW + timedelta(seconds=1)
    now_ms = int(NOW.timestamp() * 1000)
    later_ms = int(later.timestamp() * 1000)
    signer = MCPMessageSigner(KEY, nonce_generator=lambda: f"n|{now_ms}")
    signed = signer.sign_message("payload", sender_id="alice")
    signed = replace(
        signed,
        timestamp=later,
        signature=signer._compute_signature(
            nonce=signed.nonce, timestamp=later, sender_id="alice", payload="payload"
        ),
    )
    forged = replace(signed, nonce="n", timestamp=NOW, sender_id=f"{later_ms}|alice")
    receiver = MCPMessageSigner(KEY)

    assert not receiver.verify_message(forged).is_valid
    assert receiver.cached_nonce_count == 0
    assert receiver.verify_message(signed).is_valid


@pytest.mark.parametrize("sender", [None, "", "alice", "alice|INJECTED"])
def test_legacy_signatures_are_rejected_without_fallback(sender, fixed_clock):
    signed = MCPMessageSigner(KEY).sign_message("alpha|beta", sender_id=sender)
    legacy = (
        f"{signed.nonce}|{int(signed.timestamp.timestamp() * 1000)}|{sender or ''}|{signed.payload}"
    )
    digest = hmac.new(KEY, legacy.encode("utf-8"), hashlib.sha256).digest()
    envelope = replace(signed, signature=base64.b64encode(digest).decode("ascii"))
    receiver = MCPMessageSigner(KEY)

    assert not receiver.verify_message(envelope).is_valid
    assert receiver.cached_nonce_count == 0
    assert receiver.verify_message(signed).is_valid


def test_canonical_encoding_is_injective_over_adversarial_fields():
    values = ("", "|", "a|", "|a", "a|b", "1:a", "2:ab", "-", "\\", '"', "\x00", "\n")
    nonblank_values = tuple(value for value in values if value.strip())
    nonces = ("n",) + nonblank_values
    payloads = ("p",) + nonblank_values
    senders = (None,) + values
    timestamps = (NOW, NOW + timedelta(microseconds=1))
    encodings = set()

    for nonce, timestamp, sender_id, payload in product(nonces, timestamps, senders, payloads):
        canonical = MCPMessageSigner._build_canonical_string(
            nonce=nonce, timestamp=timestamp, sender_id=sender_id, payload=payload
        )
        assert canonical not in encodings
        encodings.add(canonical)

    assert len(encodings) == len(nonces) * len(timestamps) * len(senders) * len(payloads)


def test_versioned_canonical_encoding_and_signature_vector(fixed_clock):
    signer = MCPMessageSigner(KEY, nonce_generator=lambda: 'n|"\\\n')
    envelope = signer.sign_message('{"text":"caf\u00e9|\U0001f600"}', sender_id=None)
    canonical = (
        '["agent-os:mcp-message:v2","n|\\"\\\\\\n",'
        '"2026-09-24T10:00:00.123456+00:00",null,'
        '"{\\"text\\":\\"caf\u00e9|\U0001f600\\"}"]'
    )

    assert (
        signer._build_canonical_string(
            nonce=envelope.nonce,
            timestamp=envelope.timestamp,
            sender_id=envelope.sender_id,
            payload=envelope.payload,
        )
        == canonical
    )
    expected = base64.b64encode(
        hmac.new(KEY, canonical.encode("utf-8"), hashlib.sha256).digest()
    ).decode("ascii")
    assert envelope.signature == expected
    assert json.loads(canonical)[-2:] == [None, envelope.payload]
    assert MCPMessageSigner(KEY).verify_message(envelope).is_valid


@pytest.mark.parametrize("field", ["nonce", "timestamp", "sender_id", "payload", "signature"])
def test_each_signed_field_is_authenticated(field, fixed_clock):
    signed = MCPMessageSigner(KEY).sign_message("payload", sender_id="alice")
    value = getattr(signed, field)
    changed = value + timedelta(microseconds=1) if field == "timestamp" else value + "|"
    receiver = MCPMessageSigner(KEY)

    result = receiver.verify_message(replace(signed, **{field: changed}))

    assert not result.is_valid
    assert result.failure_reason == "Invalid signature."
    assert receiver.verify_message(signed).is_valid


def test_equivalent_timezone_representation_preserves_the_signed_instant(fixed_clock):
    signed = MCPMessageSigner(KEY).sign_message("payload")
    offset = timezone(timedelta(hours=5, minutes=30))
    envelope = replace(signed, timestamp=signed.timestamp.astimezone(offset))

    assert MCPMessageSigner(KEY).verify_message(envelope).is_valid


def test_envelope_survives_json_transport_round_trip(fixed_clock):
    signed = MCPMessageSigner(KEY).sign_message('{"method":"tools/call"}', sender_id="alice|")
    wire = json.dumps(asdict(signed), default=lambda value: value.isoformat())
    fields = json.loads(wire)
    fields["timestamp"] = datetime.fromisoformat(fields["timestamp"])
    received = MCPSignedEnvelope(**fields)

    assert received == signed
    assert MCPMessageSigner(KEY).verify_message(received).is_valid


@pytest.mark.parametrize("sender", [None, "", "a|b", '"\\\x00\n', "\u00e9", "e\u0301"])
@pytest.mark.parametrize("payload", ["a|b", '"\\\x00\n', "\u00e9", "e\u0301", "\U0001f600"])
def test_valid_special_characters_round_trip_unchanged(sender, payload):
    signer = MCPMessageSigner(KEY, nonce_generator=lambda: 'n|:"\\\x00\n')
    signed = signer.sign_message(payload, sender_id=sender)

    result = MCPMessageSigner(KEY).verify_message(signed)

    assert result.is_valid
    assert result.sender_id == sender
    assert result.payload == payload


@pytest.mark.parametrize("field", ["nonce", "sender_id", "payload"])
def test_unicode_normalization_is_not_silently_applied(field):
    signer = MCPMessageSigner(KEY, nonce_generator=lambda: "\u00e9")
    signed = signer.sign_message("\u00e9", sender_id="\u00e9")
    receiver = MCPMessageSigner(KEY)

    assert not receiver.verify_message(replace(signed, **{field: "e\u0301"})).is_valid
    assert receiver.verify_message(signed).is_valid


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload", None),
        ("payload", b"payload"),
        ("payload", 1),
        ("payload", ""),
        ("payload", " \t"),
        ("payload", "\ud83d\ude00"),
        ("sender_id", 0),
        ("sender_id", False),
        ("sender_id", []),
        ("sender_id", b"alice"),
        ("sender_id", "\ud800"),
        ("nonce", None),
        ("nonce", 1),
        ("nonce", b"nonce"),
        ("nonce", ""),
        ("nonce", " \n"),
        ("nonce", "\ud800"),
        ("timestamp", None),
        ("timestamp", "2026-09-24T10:00:00Z"),
        ("timestamp", NOW.replace(tzinfo=None)),
        ("signature", None),
        ("signature", b"signature"),
        ("signature", "\u00e9"),
    ],
)
def test_malformed_envelopes_fail_closed_without_consuming_nonce(field, value, fixed_clock):
    signed = MCPMessageSigner(KEY).sign_message("payload", sender_id="alice")
    receiver = MCPMessageSigner(KEY)

    result = receiver.verify_message(replace(signed, **{field: value}))

    assert not result.is_valid
    assert result.failure_reason
    assert result.payload is None
    assert result.sender_id is None
    assert receiver.cached_nonce_count == 0
    assert receiver.verify_message(signed).is_valid


@pytest.mark.parametrize("payload", [None, b"payload", 1, "", " \t", "\ud800"])
def test_signer_rejects_invalid_payloads(payload):
    with pytest.raises((TypeError, ValueError)):
        MCPMessageSigner(KEY).sign_message(payload)


@pytest.mark.parametrize("sender", [False, 0, [], b"alice", "\ud800"])
def test_signer_rejects_invalid_sender_types(sender):
    with pytest.raises((TypeError, ValueError)):
        MCPMessageSigner(KEY).sign_message("payload", sender_id=sender)


@pytest.mark.parametrize("nonce", [None, 0, [], b"nonce", "", " \t", "\ud800"])
def test_signer_rejects_invalid_generated_nonces(nonce):
    signer = MCPMessageSigner(KEY, nonce_generator=lambda: nonce)

    with pytest.raises((TypeError, ValueError)):
        signer.sign_message("payload")


@pytest.mark.parametrize("timestamp", [None, NOW.replace(tzinfo=None), "2026-09-24"])
def test_canonical_encoding_rejects_invalid_timestamps(timestamp):
    with pytest.raises(ValueError, match="timezone-aware datetime"):
        MCPMessageSigner._build_canonical_string(
            nonce="nonce", timestamp=timestamp, sender_id=None, payload="payload"
        )


def test_same_receiver_accepts_only_one_concurrent_delivery(monkeypatch):
    receiver = MCPMessageSigner(KEY)
    signed = receiver.sign_message("payload")
    barrier = Barrier(4)
    compute_signature = receiver._compute_signature

    def synchronized_signature(**kwargs):
        signature = compute_signature(**kwargs)
        barrier.wait(timeout=10)
        return signature

    monkeypatch.setattr(receiver, "_compute_signature", synchronized_signature)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(receiver.verify_message, [signed] * 4))

    assert sum(result.is_valid for result in results) == 1
    assert receiver.cached_nonce_count == 1
    assert all(
        result.is_valid or result.failure_reason == "Duplicate nonce (replay detected)."
        for result in results
    )


def test_shared_store_accepts_only_one_concurrent_delivery(monkeypatch):
    store = InMemoryNonceStore()
    receivers = [MCPMessageSigner(KEY, nonce_store=store) for _ in range(4)]
    signed = MCPMessageSigner(KEY).sign_message("payload")
    barrier = Barrier(4)
    has_nonce = store.has

    def synchronized_has(nonce):
        found = has_nonce(nonce)
        barrier.wait(timeout=10)
        return found

    monkeypatch.setattr(store, "has", synchronized_has)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda receiver: receiver.verify_message(signed), receivers))

    assert sum(result.is_valid for result in results) == 1
    assert store.count() == 1
    assert all(
        result.is_valid or result.failure_reason == "Duplicate nonce (replay detected)."
        for result in results
    )


def test_duplicate_claim_cannot_shorten_retention_at_capacity():
    now = [NOW]
    store = InMemoryNonceStore(clock=lambda: now[0], max_entries=1)
    expires_at = NOW + timedelta(minutes=5)
    store.add("nonce", expires_at)

    with pytest.raises(DuplicateNonceError):
        store.add("nonce", NOW)
    now[0] = expires_at
    assert store.has("nonce")
    with pytest.raises(DuplicateNonceError):
        store.add("nonce", expires_at + timedelta(minutes=5))
    now[0] += timedelta(microseconds=1)
    store.add("nonce", now[0] + timedelta(minutes=5))
    assert store.has("nonce")
    assert store.count() == 1


def test_falsey_injected_nonce_store_is_not_replaced():
    class EmptyStore(InMemoryNonceStore):
        def __bool__(self):
            return bool(self.count())

    store = EmptyStore()
    receiver = MCPMessageSigner(KEY, nonce_store=store)
    signed = MCPMessageSigner(KEY).sign_message("payload")

    assert receiver.verify_message(signed).is_valid
    assert store.has(signed.nonce)
    assert not MCPMessageSigner(KEY, nonce_store=store).verify_message(signed).is_valid


def test_message_expiring_during_signature_verification_is_rejected(monkeypatch):
    now = [NOW]
    monkeypatch.setattr("agent_os.mcp_message_signer._utcnow", lambda: now[0])
    receiver = MCPMessageSigner(KEY)
    signed = receiver.sign_message("payload")
    compute_signature = receiver._compute_signature

    def slow_signature(**kwargs):
        signature = compute_signature(**kwargs)
        now[0] += timedelta(minutes=5, microseconds=1)
        return signature

    monkeypatch.setattr(receiver, "_compute_signature", slow_signature)
    result = receiver.verify_message(signed)

    assert not result.is_valid
    assert result.failure_reason == "Message timestamp outside replay window."
    assert receiver.cached_nonce_count == 0


@pytest.mark.parametrize("offset", [-1, 1])
def test_replay_window_rejects_past_and_future_messages(offset, fixed_clock):
    signed = MCPMessageSigner(KEY).sign_message("payload")
    timestamp = NOW + offset * (timedelta(minutes=5) + timedelta(microseconds=1))
    signer = MCPMessageSigner(KEY)
    signed = replace(
        signed,
        timestamp=timestamp,
        signature=signer._compute_signature(
            nonce=signed.nonce, timestamp=timestamp, sender_id=None, payload=signed.payload
        ),
    )

    result = signer.verify_message(signed)

    assert not result.is_valid
    assert result.failure_reason == "Message timestamp outside replay window."
    assert signer.cached_nonce_count == 0


@pytest.mark.parametrize("offset", [-1, 1])
def test_replay_window_boundaries_accept_once(offset, fixed_clock):
    signer = MCPMessageSigner(KEY)
    signed = signer.sign_message("payload")
    timestamp = NOW + offset * timedelta(minutes=5)
    signed = replace(
        signed,
        timestamp=timestamp,
        signature=signer._compute_signature(
            nonce=signed.nonce, timestamp=timestamp, sender_id=None, payload=signed.payload
        ),
    )

    assert signer.verify_message(signed).is_valid
    assert not signer.verify_message(signed).is_valid
    assert signer.cached_nonce_count == 1


@pytest.mark.parametrize("automatic", [False, True])
def test_cleanup_never_reopens_a_live_replay_window(automatic, monkeypatch):
    now = [NOW]
    monkeypatch.setattr("agent_os.mcp_message_signer._utcnow", lambda: now[0])
    store = InMemoryNonceStore(clock=lambda: now[0], max_entries=1)
    signer = MCPMessageSigner(
        KEY, nonce_store=store, nonce_cache_cleanup_interval=timedelta(seconds=1)
    )
    first = signer.sign_message("first")
    assert signer.verify_message(first).is_valid
    now[0] += timedelta(minutes=5)
    if not automatic:
        assert signer.cleanup_nonce_cache() == 0
    assert not signer.verify_message(signer.sign_message("at boundary")).is_valid
    assert not signer.verify_message(first).is_valid
    now[0] += timedelta(seconds=1)
    if not automatic:
        assert signer.cleanup_nonce_cache() == 1
    second = signer.sign_message("second")

    assert signer.verify_message(second).is_valid
    assert store.count() == 1
    assert not signer.verify_message(first).is_valid
    assert not signer.verify_message(second).is_valid


def test_store_errors_fail_closed(monkeypatch):
    store = InMemoryNonceStore()
    receiver = MCPMessageSigner(KEY, nonce_store=store)
    signed = MCPMessageSigner(KEY).sign_message("payload")

    def unavailable_store(nonce, expires_at):
        raise OSError("nonce store unavailable")

    monkeypatch.setattr(store, "add", unavailable_store)
    result = receiver.verify_message(signed)

    assert not result.is_valid
    assert result.payload is None
    assert result.sender_id is None
    assert "fail-closed" in result.failure_reason
