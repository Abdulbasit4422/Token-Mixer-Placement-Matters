"""Privacy-safe handling for persisted case identifiers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_CASE_SINGULAR_KEYS = frozenset({"case_id", "caseId", "id"})
_CASE_PLURAL_KEYS = frozenset({"case_ids", "caseIds", "ids"})
_CASE_HASH_SINGULAR_KEY = "case_id_hash"
_CASE_HASH_PLURAL_KEY = "case_id_hashes"
_IDENTIFIER_SINGULAR_KEYS = frozenset(
    {
        "subject",
        "subject_id",
        "subjectId",
        "patient_id",
        "patientId",
        "participant_id",
        "participantId",
        "specimen_id",
        "specimenId",
        "sample_id",
        "sampleId",
    }
)
_IDENTIFIER_PLURAL_KEYS = frozenset(
    {
        "subjects",
        "subject_ids",
        "subjectIds",
        "patient_ids",
        "patientIds",
        "participant_ids",
        "participantIds",
        "specimen_ids",
        "specimenIds",
        "sample_ids",
        "sampleIds",
    }
)
_IDENTIFIER_HASH_SINGULAR_KEYS = frozenset(
    {"case_id_hash", "subject_id_hash", "subjectIdHash"}
)
_IDENTIFIER_HASH_PLURAL_KEYS = frozenset(
    {"case_id_hashes", "subject_id_hashes", "subjectIdHashes"}
)
_PRESERVED_REFERENCE_KEYS = frozenset(
    {
        "artifact",
        "artifacts",
        "artifact_ref",
        "artifact_refs",
        "artifact_reference",
        "source_artifact",
        "source_checkpoint_artifact",
        "project",
        "projects",
        "project_name",
        "projectName",
        "entity",
        "entity_name",
        "entityName",
    }
)
_MAX_SAFE_ERROR_LENGTH = 512
_REDACTED_CASE_ID = "[REDACTED_CASE_ID]"
_REDACTED_PATH = "[REDACTED_PATH]"
_REDACTED_SECRET = "[REDACTED_SECRET]"
_REDACTED_ERROR = "[REDACTED_ERROR]"
_MISSING = object()
_REDACTION_SENTINEL = "\ue000"
_PATH_REDACTION_SENTINEL = "\ue001"
_CASE_TOKEN_PATTERN = re.compile(
    r"\b(?:case|brats)-[A-Za-z0-9][A-Za-z0-9_.-]*", re.IGNORECASE
)
_CASE_SUFFIX_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z0-9]+[-_]case(?:[-_][A-Za-z0-9]+)*\b", re.IGNORECASE
)
_CASE_ALNUM_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:case|brats)[-_]?[A-Za-z]*\d[A-Za-z0-9]*\b", re.IGNORECASE
)
_PATIENT_TOKEN_PATTERN = re.compile(
    r"\bpatient[-_ .]+(?=[A-Za-z0-9_.-]*\d)[A-Za-z0-9][A-Za-z0-9_.-]*\b",
    re.IGNORECASE,
)
_GENERIC_IDENTIFIER_TOKEN_PATTERN = re.compile(
    r"\b(?:subject|participant|specimen|sample)[-_ .]+"
    r"(?=[A-Za-z0-9_.-]*\d)[A-Za-z0-9][A-Za-z0-9_.-]*\b",
    re.IGNORECASE,
)
_GENERIC_IDENTIFIER_SHAPE_PATTERN = re.compile(
    r"\b(?=[A-Z0-9_-]*\d)[A-Z][A-Z0-9]*(?:[_-][A-Z0-9]+)+\b"
)
_PREFIXED_IDENTIFIER_VALUE_PATTERN = re.compile(
    r"\b(?:subject|participant|specimen|sample|patient)[-_ .]?"
    r"[A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*\b",
    re.IGNORECASE,
)
_SHORT_IDENTIFIER_TOKEN_PATTERN = re.compile(r"\b[A-Z]{1,4}\d{1,}\b")
_IDENTIFIER_CONTEXT_KEYS = frozenset(
    {
        "case",
        "cases",
        "case_identifier",
        "case_identifiers",
        "identifier",
        "identifiers",
        "patient",
        "patient_name",
        "patientName",
        "sample_name",
        "specimen_name",
        "subject_name",
        "participant_name",
    }
)
_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\|\.\.?[\\/])[^\s,;)}\]]+"
    r"|(?<![A-Za-z0-9])/(?:[^/\s,;)}\]]+/)+[^/\s,;)}\]]+"
    r"|(?<![A-Za-z0-9])(?:[A-Za-z0-9_.-]+[\\/])+[^\\/\s,;)}\]]+"
)
_LOCAL_PATH_PREFIX_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|\.\.?[\\/])[^\s,;)}\]]+"
    r"|(?<![A-Za-z0-9])/(?:[^/\s,;)}\]]+/)+[^/\s,;)}\]]+"
)
_SECRET_PATTERN = re.compile(
    r"\b(?:api[_-]?key|token|secret|password|passwd)\b\s*[:=]\s*"
    r"\S+",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]+",
    re.IGNORECASE,
)
_LOCAL_ARTIFACT_IDENTIFIER_PATTERNS = (
    _CASE_TOKEN_PATTERN,
    _CASE_SUFFIX_IDENTIFIER_PATTERN,
    _CASE_ALNUM_IDENTIFIER_PATTERN,
    _PATIENT_TOKEN_PATTERN,
    _GENERIC_IDENTIFIER_TOKEN_PATTERN,
    _GENERIC_IDENTIFIER_SHAPE_PATTERN,
    _PREFIXED_IDENTIFIER_VALUE_PATTERN,
    _SHORT_IDENTIFIER_TOKEN_PATTERN,
)


def _is_local_path_candidate(value: str) -> bool:
    """Return whether path-shaped match is local path, not metric namespace."""

    if re.match(r"(?:[A-Za-z]:[\\/]|\\\\|\.\.?[\\/])", value):
        return True
    if value.startswith("/"):
        return True
    if not re.search(r"[\\/]", value):
        return False
    final_segment = re.split(r"[\\/]", value)[-1]
    return bool(re.fullmatch(r"[^\\/\s]+\.[A-Za-z0-9]{1,8}", final_segment))


def _replace_local_paths(value: str, replacement: str) -> str:
    sanitized = _LOCAL_PATH_PREFIX_PATTERN.sub(lambda _match: replacement, value)
    return _PATH_PATTERN.sub(
        lambda match: replacement
        if _is_local_path_candidate(match.group(0))
        else match.group(0),
        sanitized,
    )


def hash_case_id(case_id: str) -> str:
    """Return the stable short hash used for persisted case references."""

    if not isinstance(case_id, str):
        raise TypeError("case_id must be a string")
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]


def _hash_case_value(value: Any) -> str:
    return hash_case_id(value if isinstance(value, str) else str(value))


def _hash_case_values(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return [_hash_case_value(value)]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            listed = tolist()
        except (TypeError, ValueError, RuntimeError):
            listed = value
        if listed is not value:
            return _hash_case_values(listed)
    try:
        values = list(value)
    except (TypeError, ValueError):
        values = [value]
    return [_hash_case_value(item) for item in values]


def _iter_identifier_values(value: Any):
    if value is None:
        return
    if isinstance(value, (str, bytes, bytearray)):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_identifier_values(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_identifier_values(item)
        return
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            listed = tolist()
        except (TypeError, ValueError, RuntimeError):
            listed = value
        if listed is not value:
            yield from _iter_identifier_values(listed)
            return
    yield value


def _iter_known_case_ids(value: Any, *, _root: bool = True):
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in _CASE_SINGULAR_KEYS or key_text in _CASE_PLURAL_KEYS:
                yield from _iter_identifier_values(item)
            else:
                yield from _iter_known_case_ids(item, _root=False)
        return
    if _root:
        yield from _iter_identifier_values(value)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            if isinstance(item, Mapping):
                yield from _iter_known_case_ids(item, _root=False)


def _redact_identifier_tokens(value: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    sanitized = value
    for pattern in patterns:
        sanitized = pattern.sub(_REDACTION_SENTINEL, sanitized)
    return sanitized.replace(_REDACTION_SENTINEL, _REDACTED_CASE_ID)


def _safe_local_artifact_text(value: str) -> str:
    """Keep local artifact paths while removing identifiers and secrets."""

    sanitized = _redact_identifier_tokens(
        value, _LOCAL_ARTIFACT_IDENTIFIER_PATTERNS
    )
    for pattern in (_SECRET_PATTERN, _BEARER_PATTERN):
        sanitized = pattern.sub(_REDACTED_SECRET, sanitized)
    return sanitized


def _safe_error_text(message: str, known_case_ids: Any) -> tuple[str, bool]:
    sanitized = message
    changed = False
    known_values = {
        str(value)
        for value in _iter_known_case_ids(known_case_ids)
        if value is not None and str(value)
    }
    for case_id in sorted(known_values, key=len, reverse=True):
        replacement = sanitized.replace(case_id, _REDACTED_CASE_ID)
        changed = changed or replacement != sanitized
        sanitized = replacement

    replacement = _replace_local_paths(sanitized, _PATH_REDACTION_SENTINEL)
    changed = changed or replacement != sanitized
    sanitized = replacement

    replacement = _redact_identifier_tokens(
        sanitized,
        (
            _CASE_TOKEN_PATTERN,
            _CASE_SUFFIX_IDENTIFIER_PATTERN,
            _CASE_ALNUM_IDENTIFIER_PATTERN,
            _PATIENT_TOKEN_PATTERN,
            _GENERIC_IDENTIFIER_TOKEN_PATTERN,
            _GENERIC_IDENTIFIER_SHAPE_PATTERN,
        ),
    )
    changed = changed or replacement != sanitized
    sanitized = replacement

    for pattern in (_SECRET_PATTERN, _BEARER_PATTERN):
        replacement = pattern.sub(_REDACTED_SECRET, sanitized)
        changed = changed or replacement != sanitized
        sanitized = replacement
    sanitized = sanitized.replace(_PATH_REDACTION_SENTINEL, _REDACTED_PATH)
    return sanitized, changed


def safe_error_message(
    error: BaseException | str,
    *,
    known_case_ids: Any = (),
) -> str:
    """Return bounded error text without case IDs, paths, or credentials.

    Ordinary fixture messages remain unchanged. When sensitive content is
    replaced, the exception type is retained as a useful diagnostic prefix.
    """

    error_type = type(error).__name__ if isinstance(error, BaseException) else None
    try:
        original = str(error)
    except BaseException:
        original = ""
    sanitized, changed = _safe_error_text(original, known_case_ids)
    if not sanitized:
        sanitized = _REDACTED_ERROR
        changed = True
    if changed and error_type:
        sanitized = f"{error_type}: {sanitized}"
    if len(sanitized) > _MAX_SAFE_ERROR_LENGTH:
        sanitized = sanitized[: _MAX_SAFE_ERROR_LENGTH - 3].rstrip() + "..."
    return sanitized


def _hash_identifier_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_hash_case_value(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            listed = tolist()
        except (TypeError, ValueError, RuntimeError):
            listed = value
        if listed is not value:
            return _hash_identifier_value(listed)
    return _hash_case_value(value)


def _looks_like_raw_identifier(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    return any(
        pattern.search(value)
        for pattern in (
            _CASE_TOKEN_PATTERN,
            _CASE_SUFFIX_IDENTIFIER_PATTERN,
            _CASE_ALNUM_IDENTIFIER_PATTERN,
            _PATIENT_TOKEN_PATTERN,
            _GENERIC_IDENTIFIER_TOKEN_PATTERN,
            _GENERIC_IDENTIFIER_SHAPE_PATTERN,
            _PATH_PATTERN,
        )
    )


def _normalize_hash_value(value: Any, *, plural: bool = False) -> Any:
    def normalize(item: Any) -> Any:
        if item is None:
            return None
        if isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{16}", item):
            return item.lower()
        return _hash_case_value(item)

    if plural:
        if value is None:
            return None
        if isinstance(value, (str, bytes, bytearray)):
            values = [value]
        else:
            try:
                values = list(value)
            except (TypeError, ValueError):
                values = [value]
        return [normalize(item) for item in values]
    return normalize(value)


def _redact_reference_sequence(
    value: Any,
    key: str,
    *,
    preserve_opaque_hashes: bool = False,
    preserve_local_paths: bool = False,
) -> Any:
    if isinstance(value, list):
        return [
            _redact_value(
                item,
                key=key,
                preserve_opaque_hashes=preserve_opaque_hashes,
                preserve_local_paths=preserve_local_paths,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_value(
                item,
                key=key,
                preserve_opaque_hashes=preserve_opaque_hashes,
                preserve_local_paths=preserve_local_paths,
            )
            for item in value
        )
    if isinstance(value, (set, frozenset)):
        return [
            _redact_value(
                item,
                key=key,
                preserve_opaque_hashes=preserve_opaque_hashes,
                preserve_local_paths=preserve_local_paths,
            )
            for item in sorted(value, key=repr)
        ]
    return value


def _safe_mapping_key(value: str) -> str:
    patterns = (
        _CASE_TOKEN_PATTERN,
        _CASE_SUFFIX_IDENTIFIER_PATTERN,
        _CASE_ALNUM_IDENTIFIER_PATTERN,
        _PREFIXED_IDENTIFIER_VALUE_PATTERN,
        _GENERIC_IDENTIFIER_SHAPE_PATTERN,
        _SHORT_IDENTIFIER_TOKEN_PATTERN,
    )

    # Slash/backslash-delimited keys are metric/table namespaces (for example
    # ``power/sample_count``), not filesystem paths.  Replace only local path
    # spans first, then sanitize each remaining segment while retaining
    # namespace delimiters and ordinary metric field names.
    sanitized = _replace_local_paths(value, _PATH_REDACTION_SENTINEL)
    sanitized = re.sub(
        r"[^/\\]+",
        lambda match: _redact_identifier_tokens(match.group(0), patterns),
        sanitized,
    )
    return sanitized.replace(_PATH_REDACTION_SENTINEL, _REDACTED_PATH)


def _redact_value(
    value: Any,
    *,
    key: str | None = None,
    preserve_opaque_hashes: bool = False,
    preserve_local_paths: bool = False,
) -> Any:
    if isinstance(value, Path):
        return (
            _safe_local_artifact_text(str(value))
            if preserve_local_paths
            else _REDACTED_PATH
        )
    if key in _IDENTIFIER_CONTEXT_KEYS:
        if isinstance(value, str) and _replace_local_paths(value, _PATH_REDACTION_SENTINEL) != value:
            return _safe_error_text(value, ())[0]
        return _hash_identifier_value(value)
    if isinstance(value, str):
        if key in _PRESERVED_REFERENCE_KEYS:
            return value
        return (
            _safe_local_artifact_text(value)
            if preserve_local_paths
            else _safe_error_text(value, ())[0]
        )
    if key in _PRESERVED_REFERENCE_KEYS and isinstance(
        value, (list, tuple, set, frozenset)
    ):
        return _redact_reference_sequence(
            value,
            key,
            preserve_opaque_hashes=preserve_opaque_hashes,
            preserve_local_paths=preserve_local_paths,
        )
    if isinstance(value, Mapping):
        raw_singular = next(
            (
                item
                for raw_key, item in value.items()
                if str(raw_key) in _CASE_SINGULAR_KEYS
            ),
            _MISSING,
        )
        raw_plural = next(
            (
                item
                for raw_key, item in value.items()
                if str(raw_key) in _CASE_PLURAL_KEYS
            ),
            _MISSING,
        )
        canonical_singular = (
            _hash_case_value(raw_singular)
            if raw_singular is not _MISSING
            else _MISSING
        )
        canonical_plural = (
            _hash_case_values(raw_plural)
            if raw_plural is not _MISSING
            else _MISSING
        )
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key_text = str(raw_key)
            if key_text in _CASE_SINGULAR_KEYS:
                if not (
                    preserve_opaque_hashes
                    and _CASE_HASH_SINGULAR_KEY in value
                ):
                    result[_CASE_HASH_SINGULAR_KEY] = canonical_singular
                continue
            if key_text in _CASE_PLURAL_KEYS:
                if not (
                    preserve_opaque_hashes
                    and _CASE_HASH_PLURAL_KEY in value
                ):
                    result[_CASE_HASH_PLURAL_KEY] = canonical_plural
                continue
            if key_text == _CASE_HASH_SINGULAR_KEY:
                result[key_text] = (
                    _redact_value(
                        item,
                        preserve_opaque_hashes=preserve_opaque_hashes,
                        preserve_local_paths=preserve_local_paths,
                    )
                    if preserve_opaque_hashes
                    else (
                        canonical_singular
                        if canonical_singular is not _MISSING
                        else _normalize_hash_value(item)
                    )
                )
                continue
            if key_text == _CASE_HASH_PLURAL_KEY:
                result[key_text] = (
                    _redact_value(
                        item,
                        preserve_opaque_hashes=preserve_opaque_hashes,
                        preserve_local_paths=preserve_local_paths,
                    )
                    if preserve_opaque_hashes
                    else (
                        canonical_plural
                        if canonical_plural is not _MISSING
                        else _normalize_hash_value(item, plural=True)
                    )
                )
                continue
            if key_text in _IDENTIFIER_SINGULAR_KEYS:
                result[key_text] = _hash_identifier_value(item)
                continue
            if key_text in _IDENTIFIER_PLURAL_KEYS:
                result[key_text] = _hash_identifier_value(item)
                continue
            if key_text in _IDENTIFIER_HASH_SINGULAR_KEYS:
                result[key_text] = _normalize_hash_value(item)
                continue
            if key_text in _IDENTIFIER_HASH_PLURAL_KEYS:
                result[key_text] = _normalize_hash_value(item, plural=True)
                continue
            safe_key = (
                _safe_local_artifact_text(key_text)
                if preserve_local_paths
                else _safe_mapping_key(key_text)
            )
            result[safe_key] = _redact_value(
                item,
                key=key_text,
                preserve_opaque_hashes=preserve_opaque_hashes,
                preserve_local_paths=preserve_local_paths,
            )
        if canonical_singular is not _MISSING and _CASE_HASH_SINGULAR_KEY not in result:
            result[_CASE_HASH_SINGULAR_KEY] = canonical_singular
        if canonical_plural is not _MISSING and _CASE_HASH_PLURAL_KEY not in result:
            result[_CASE_HASH_PLURAL_KEY] = canonical_plural
        return result
    if isinstance(value, list):
        return [
            _redact_value(
                item,
                preserve_opaque_hashes=preserve_opaque_hashes,
                preserve_local_paths=preserve_local_paths,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_value(
                item,
                preserve_opaque_hashes=preserve_opaque_hashes,
                preserve_local_paths=preserve_local_paths,
            )
            for item in value
        )
    if isinstance(value, (set, frozenset)):
        return [
            _redact_value(
                item,
                preserve_opaque_hashes=preserve_opaque_hashes,
                preserve_local_paths=preserve_local_paths,
            )
            for item in sorted(value, key=repr)
        ]
    return value


def redact_case_identifiers(value: Any) -> Any:
    """Recursively replace raw singular/plural case fields with hashes.

    Valid ``case_id_hash`` and ``case_id_hashes`` values are retained (and
    normalized to lowercase); invalid caller values and raw aliases are
    converted to canonical hashes. The input containers are never mutated,
    which keeps in-memory training results and benchmark inputs available to
    their callers.
    """

    return _redact_value(value)


def _redact_local_artifact(value: Any) -> Any:
    """Apply disk-artifact compatibility without relaxing external sanitization."""

    return _redact_value(
        value,
        preserve_opaque_hashes=True,
        preserve_local_paths=True,
    )


def sanitize_namespace_key(value: Any) -> str:
    """Sanitize identifier content in a metric namespace without changing separators."""

    return _safe_mapping_key(str(value))


def redact_image_metadata(value: Any) -> str:
    """Sanitize image names/captions before they cross a tracker boundary."""

    sanitized, _changed = _safe_error_text(str(value), ())
    return _redact_identifier_tokens(
        sanitized, (_SHORT_IDENTIFIER_TOKEN_PATTERN,)
    )


__all__ = [
    "hash_case_id",
    "redact_case_identifiers",
    "safe_error_message",
    "sanitize_namespace_key",
    "redact_image_metadata",
]
