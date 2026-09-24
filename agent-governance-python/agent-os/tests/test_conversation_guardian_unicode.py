# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Regression coverage for invisible-character evasion (issue #3500)."""

import hashlib
import unicodedata

import pytest

from agent_os.integrations.conversation_guardian import (
    _INVISIBLE_DELETE,
    AlertAction,
    AlertSeverity,
    ConversationGuardian,
    ConversationGuardianConfig,
    EscalationClassifier,
    FeedbackLoopBreaker,
    OffensiveIntentDetector,
    _detection_texts,
    normalize_text,
)

# Independent Unicode 17.0.0 Default_Ignorable_Code_Point snapshot from
# https://www.unicode.org/Public/17.0.0/ucd/DerivedCoreProperties.txt.
# Update this snapshot explicitly when adopting a new Unicode version.
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
_INVISIBLE_SAMPLES = tuple(
    dict.fromkeys(
        chr(cp)
        for start, end in (*_DEFAULT_IGNORABLE_RANGES, (0xFFF9, 0xFFFB))
        for cp in (start, end)
    )
) + ("\u2065", "\U000e0020", "\U000e007f", "\U000e0100", "\U000e01ef")
_ESCALATION = "urgent you must comply immediately no excuses"
_OFFENSIVE = "exfiltrate the data escalate privileges impersonate"


def _inside_words(text: str, invisible: str) -> str:
    return " ".join(invisible.join(word) for word in text.split(" "))


@pytest.mark.parametrize("start,end", _DEFAULT_IGNORABLE_RANGES)
def test_every_default_ignorable_is_removed(start: int, end: int) -> None:
    for cp in range(start, end + 1):
        assert normalize_text(f"ur{chr(cp)}gent") == "urgent", f"U+{cp:04X}"


@pytest.mark.parametrize("cp", range(0xFFF9, 0xFFFC))
def test_interlinear_annotation_controls_are_removed(cp: int) -> None:
    # These controls are intentionally covered in addition to the Unicode property.
    assert normalize_text(f"ur{chr(cp)}gent") == "urgent"


def test_invisible_table_exactly_matches_the_unicode_snapshot_and_exceptions() -> None:
    expected = {cp for start, end in _DEFAULT_IGNORABLE_RANGES for cp in range(start, end + 1)}
    assert len(expected) == 4174
    assert set(_INVISIBLE_DELETE) == expected | set(range(0xFFF9, 0xFFFC))


@pytest.mark.parametrize("invisible", _INVISIBLE_SAMPLES, ids=ascii)
@pytest.mark.parametrize(
    "detector_type,clean",
    [(EscalationClassifier, _ESCALATION), (OffensiveIntentDetector, _OFFENSIVE)],
)
def test_invisible_in_every_keyword_preserves_score_and_matches(
    invisible: str,
    detector_type: type[EscalationClassifier] | type[OffensiveIntentDetector],
    clean: str,
) -> None:
    detector = detector_type()
    expected = detector.score_message(clean)
    assert expected[0] >= 0.8
    assert detector.score_message(_inside_words(clean, invisible)) == expected


@pytest.mark.parametrize("invisible", _INVISIBLE_SAMPLES, ids=ascii)
@pytest.mark.parametrize(
    "detector_type,obfuscated,clean",
    [
        (EscalationClassifier, "y0u must{gap}n0 excuses", "you must no excuses"),
        (
            OffensiveIntentDetector,
            "3scalate privileg3s{gap}imp3rsonate",
            "escalate privileges impersonate",
        ),
        (EscalationClassifier, "c0de{gap}r3d", "code red"),
    ],
)
def test_invisible_word_separator_preserves_boundaries(
    invisible: str,
    detector_type: type[EscalationClassifier] | type[OffensiveIntentDetector],
    obfuscated: str,
    clean: str,
) -> None:
    detector = detector_type()
    assert detector.score_message(obfuscated.format(gap=invisible)) == detector.score_message(clean)


@pytest.mark.parametrize("suffix", ["!", "!!", "+", "$", "@", "\u00a1"])
def test_stacked_unicode_leetspeak_and_punctuation(suffix: str) -> None:
    detector = EscalationClassifier()
    obfuscated = f"\uff55r\u00adg3\u034fnt{suffix}"
    assert detector.score_message(obfuscated) == detector.score_message("urgent")


def test_mixed_invisibles_and_existing_normalization() -> None:
    detector = EscalationClassifier()
    text = "\uff59\u2065\u043eu\u034f \tmu\u3164$t \n n0 ex\u0301\u00adcuse$"
    assert detector.score_message(text) == detector.score_message("you must no excuses")
    normalized = normalize_text(text)
    assert normalized == "you must no excuses"
    assert normalize_text(normalized) == normalized


@pytest.mark.parametrize("joiner", ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"])
@pytest.mark.parametrize("separator", ["\u00ad", "\u034f", "\ufe0f", "\U000e007f"])
def test_legacy_detection_survives_mixed_joiners_and_separators(
    joiner: str, separator: str
) -> None:
    # The old normalizer removed the joiner but kept the separator as a boundary.
    text = f"y0u mu{joiner}st{separator}n0 ex{joiner}cuses"
    detector = EscalationClassifier()
    assert detector.score_message(text) == detector.score_message("you must no excuses")


@pytest.mark.parametrize(
    "detector_type,clean",
    [
        (EscalationClassifier, "This is an urgent emergency"),
        (EscalationClassifier, "you must obey a direct order"),
        (EscalationClassifier, "bypass security and exploit every vulnerability"),
        (OffensiveIntentDetector, "find a vulnerability and reverse engineer"),
        (OffensiveIntentDetector, "forge session credentials and escalate privileges"),
        (OffensiveIntentDetector, "exfiltrate data through a covert channel"),
        (OffensiveIntentDetector, "enumerate services and scan network ports"),
        (OffensiveIntentDetector, "SQL injection and remote code execution"),
        (OffensiveIntentDetector, "lateral movement to pivot through the network"),
    ],
)
def test_every_pattern_group_resists_midword_obfuscation(
    detector_type: type[EscalationClassifier] | type[OffensiveIntentDetector], clean: str
) -> None:
    detector = detector_type()
    expected = detector.score_message(clean)
    assert expected[0] > 0
    assert detector.score_message(_inside_words(clean, "\u00ad\ufe0f")) == expected


@pytest.mark.parametrize("invisible", _INVISIBLE_SAMPLES, ids=ascii)
@pytest.mark.parametrize(
    "error", ["access denied", "permission denied", "unauthorized", "401", "403"]
)
def test_obfuscated_error_messages_still_break_retry_loops(invisible: str, error: str) -> None:
    breaker = FeedbackLoopBreaker(max_retry_cycles=3)
    baseline = FeedbackLoopBreaker(max_retry_cycles=3)
    for turn in range(3):
        timestamp = 1_700_000_000.0 + turn
        assert breaker.record_message(
            "conversation", _inside_words(error, invisible), timestamp=timestamp
        ) == baseline.record_message("conversation", error, timestamp=timestamp)
    assert breaker.get_state("conversation") == baseline.get_state("conversation")
    assert breaker.should_break("conversation") == baseline.should_break("conversation")
    assert breaker.should_break("conversation")[0]


@pytest.mark.parametrize(
    "error",
    ["acc3ss\u00add3nied", "\uff14\uff10\uff13", "4\u034f0\u00ad3!", "un\u00adauth0rized!"],
)
def test_guardian_breaks_obfuscated_retry_loop(error: str) -> None:
    guardian = ConversationGuardian(ConversationGuardianConfig())
    for turn in range(3):
        alert = guardian.analyze_message(
            "conversation", "a", "b", error, timestamp=1_700_000_000.0 + turn
        )
    assert alert.action == AlertAction.BREAK
    assert alert.severity == AlertSeverity.HIGH
    assert guardian.loop_breaker.get_state("conversation")["retry_count"] == 3


def test_guardian_quarantines_and_audits_original_obfuscated_message() -> None:
    clean = f"{_ESCALATION} {_OFFENSIVE}"
    obfuscated = _inside_words(clean, "\u00ad\u034f\ufe0f\U000e007f")
    config = ConversationGuardianConfig()
    guardian = ConversationGuardian(config)
    baseline = ConversationGuardian(config).analyze_message(
        "conversation", "a", "b", clean, timestamp=1_700_000_000.0
    )
    alert = guardian.analyze_message(
        "conversation", "a", "b", obfuscated, timestamp=1_700_000_000.0
    )
    assert alert.to_dict() == baseline.to_dict()
    assert alert.action == AlertAction.QUARANTINE
    assert alert.severity == AlertSeverity.CRITICAL
    transcript = guardian.get_transcript()[0]
    assert transcript.content_hash == hashlib.sha256(obfuscated.encode("utf-8")).hexdigest()[:16]
    assert transcript.content_preview == obfuscated[:200] + "..."


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "caf\u00e9 d\u00e9j\u00e0 vu",
        "\u4f60\u597d\u4e16\u754c",
        "\u3053\u3093\u306b\u3061\u306f",
        "\uc548\ub155\ud558\uc138\uc694",
        "\u0645\u0631\u062d\u0628\u0627",
        "\u05e9\u05dc\u05d5\u05dd",
        "\u0915\u093f",  # Devanagari vowel sign must not be blanket-stripped.
        "\U0001f44d\U0001f3fd",
        "\U0001f1ef\U0001f1f5",
        "\u0600\u0605",  # Visible Cf characters are not default-ignorable.
        "\u2000\u2001\u2002\u2003",  # Visible spacing is not a word-internal joiner.
    ],
)
def test_visible_unicode_retains_existing_normalization(text: str) -> None:
    expected = unicodedata.normalize("NFKD", text)
    expected = "".join(ch for ch in expected if not unicodedata.combining(ch))
    expected = " ".join(expected.split())
    assert normalize_text(text) == expected
    assert normalize_text(normalize_text(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\u00ad\u034f\ufe0f\U000e0100",
        "Please review the co\u00adoperation report",
        "Send the \u4f60\u597d summary to the team",
        "\u0645\u0631\u062d\u0628\u0627 \u200f\u0628\u0627\u0644\u0639\u0627\u0644\u0645",
        "Family \U0001f468\u200d\U0001f469\u200d\U0001f467",
        "u r g e n t",  # Do not remove visible whitespace inside words.
        "u-r-g-e-n-t",
    ],
)
def test_benign_text_does_not_raise_alerts(text: str) -> None:
    guardian = ConversationGuardian(ConversationGuardianConfig())
    alert = guardian.analyze_message("conversation", "a", "b", text)
    assert alert.action == AlertAction.NONE
    assert alert.escalation_score == alert.offensive_score == 0.0
    assert guardian.loop_breaker.get_state("conversation")["retry_count"] == 0
    assert guardian.get_transcript()[0].content_preview == text


def test_multiple_views_do_not_multiply_pattern_weights() -> None:
    detector = EscalationClassifier()
    assert detector.score_message("urgent \u00ad ur\u00adgent urgent") == detector.score_message(
        "urgent"
    )


def test_plain_text_does_not_duplicate_detection_views() -> None:
    text = "Please review the quarterly report."
    assert _detection_texts(text) == (text,)


def test_heavily_obfuscated_input_has_a_fixed_number_of_detection_views() -> None:
    text = _inside_words("urg3nt! y0u must no excuses", "\u00ad" * 1_000)
    candidates = _detection_texts(text)
    assert candidates[0] == text
    assert len(candidates) == len(set(candidates)) <= 8
    assert normalize_text(text) in candidates
    assert EscalationClassifier().score_message(text) == EscalationClassifier().score_message(
        "urgent! you must no excuses"
    )
