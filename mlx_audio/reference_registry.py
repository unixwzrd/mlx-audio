"""Persistent, immutable voice-reference pairs for remote TTS cloning."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from mlx_audio.audio_io import read as audio_read

logger = logging.getLogger(__name__)


def _safe_info(message: str, *args: Any) -> None:
    logger.info(message, *args)
    logging.getLogger("uvicorn.error").info(message, *args)


MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 8 * 1024
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 30.0
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
REFERENCE_ID_RE = re.compile(r"^ref_[0-9a-f]{32}$")


class ReferenceError(ValueError):
    """Raised when a reference pair is invalid or unavailable."""


@dataclass(frozen=True)
class ResolvedReference:
    """A validated reference ready for model inference."""

    audio: str | Any
    transcript: str | None
    reference_id: str
    pair_sha256: str


def default_registry_root() -> Path:
    """Return the configured XDG-compatible registry root."""

    explicit = os.getenv("MLX_AUDIO_REFERENCE_STORE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    data_home = os.getenv("XDG_DATA_HOME", "").strip()
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local/share"
    return base / "mlx-audio/references"


def configured_reference_roots() -> tuple[Path, ...]:
    """Return canonical roots from which legacy server paths may be read."""

    raw = os.getenv("MLX_AUDIO_REFERENCE_ROOTS", "")
    roots = [
        Path(item).expanduser().resolve() for item in raw.split(os.pathsep) if item
    ]
    return tuple(roots)


def _within(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def resolve_allowed_file(value: str, roots: tuple[Path, ...]) -> Path:
    """Resolve one existing file beneath an explicitly configured root."""

    if value.startswith(("http://", "https://", "file://")):
        raise ReferenceError("reference URLs are not supported")
    if not roots:
        raise ReferenceError("server-path references are disabled")
    path = Path(value).expanduser().resolve()
    if not _within(path, roots):
        raise ReferenceError("reference path is outside configured roots")
    if not path.is_file():
        raise ReferenceError("reference file was not found")
    return path


def validate_transcript(transcript: str | None, *, required: bool) -> str | None:
    """Validate without altering exact transcript content."""

    if transcript is None:
        if required:
            raise ReferenceError("reference transcript is required")
        return None
    encoded = transcript.encode("utf-8")
    if not transcript.strip():
        raise ReferenceError("reference transcript must not be empty")
    if len(encoded) > MAX_TRANSCRIPT_BYTES:
        raise ReferenceError("reference transcript exceeds 8 KiB")
    return transcript


def validate_audio(audio: bytes, filename: str) -> tuple[float, int, int, str]:
    """Validate uploaded audio and return duration, rate, frames, and suffix."""

    if not audio:
        raise ReferenceError("reference audio must not be empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ReferenceError("reference audio exceeds 20 MiB")
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ReferenceError("reference audio must be WAV, FLAC, MP3, or M4A")
    try:
        samples, sample_rate = audio_read(BytesIO(audio))
    except Exception as exc:  # noqa: BLE001
        raise ReferenceError("reference audio could not be decoded") from exc
    frames = int(samples.shape[0])
    duration = frames / int(sample_rate)
    if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
        raise ReferenceError(
            "reference audio duration must be between 1 and 30 seconds"
        )
    return duration, int(sample_rate), frames, suffix


def pair_digest(audio: bytes, transcript: str | None) -> tuple[str, str]:
    """Return audio and immutable-pair SHA-256 identities."""

    audio_sha = hashlib.sha256(audio).hexdigest()
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(audio_sha))
    digest.update(b"\0")
    if transcript is not None:
        digest.update(transcript.encode("utf-8"))
    return audio_sha, digest.hexdigest()


class ReferenceRegistry:
    """Filesystem-backed immutable registry with atomic publication."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_registry_root()).expanduser().resolve()

    def register(
        self,
        *,
        audio: bytes,
        filename: str,
        transcript: str | None,
        name: str,
        language: str = "",
        emotion: str = "",
        intensity: float | None = None,
        instruction: str = "",
        transcript_required: bool = True,
    ) -> dict[str, Any]:
        """Validate and atomically register one immutable reference."""

        transcript = validate_transcript(transcript, required=transcript_required)
        duration, sample_rate, frames, suffix = validate_audio(audio, filename)
        clean_name = name.strip()
        if not clean_name:
            raise ReferenceError("reference name is required")
        if intensity is not None and not 0.0 <= intensity <= 1.0:
            raise ReferenceError("reference intensity must be between 0 and 1")
        audio_sha, digest = pair_digest(audio, transcript)
        reference_id = f"ref_{digest[:32]}"
        record = {
            "id": reference_id,
            "name": clean_name,
            "audio_sha256": audio_sha,
            "pair_sha256": digest,
            "audio_bytes": len(audio),
            "duration_seconds": round(duration, 6),
            "sample_rate": sample_rate,
            "frames": frames,
            "transcript_present": transcript is not None,
            "capabilities": {
                "clone_mode": "audio_transcript" if transcript is not None else "audio",
                "transcript_conditioned": transcript is not None,
            },
            "language": language.strip(),
            "style": {
                "emotion": emotion.strip(),
                "intensity": intensity,
                "instruction": instruction,
            },
            "audio_file": f"audio{suffix}",
            "transcript_file": "transcript.txt" if transcript is not None else "",
        }
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / reference_id
        if destination.exists():
            existing = self.get(reference_id, internal=True)
            if existing["pair_sha256"] != digest:
                raise ReferenceError("reference identity collision")
            _safe_info(
                "reference registry hit id=%s pair=%s bytes=%d duration=%.3f",
                reference_id,
                digest[:12],
                len(audio),
                duration,
            )
            return self.public_record(existing)

        temporary = Path(tempfile.mkdtemp(prefix=f".{reference_id}.", dir=self.root))
        try:
            (temporary / record["audio_file"]).write_bytes(audio)
            if transcript is not None:
                (temporary / "transcript.txt").write_text(transcript, encoding="utf-8")
            (temporary / "manifest.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
                existing = self.get(reference_id, internal=True)
                if existing["pair_sha256"] != digest:
                    raise ReferenceError("reference identity collision")
                shutil.rmtree(temporary, ignore_errors=True)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        _safe_info(
            "registered reference id=%s pair=%s bytes=%d duration=%.3f",
            reference_id,
            digest[:12],
            len(audio),
            duration,
        )
        return self.public_record(record)

    @staticmethod
    def public_record(record: dict[str, Any]) -> dict[str, Any]:
        """Remove internal filenames while preserving safe metadata."""

        return {
            key: value
            for key, value in record.items()
            if key not in {"audio_file", "transcript_file"}
        }

    def list(self) -> list[dict[str, Any]]:
        """List valid reference metadata without exposing paths or transcripts."""

        if not self.root.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("ref_*/manifest.json")):
            try:
                records.append(
                    self.public_record(json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def get(self, reference_id: str, *, internal: bool = False) -> dict[str, Any]:
        """Return one registry record."""

        if not REFERENCE_ID_RE.fullmatch(reference_id):
            raise ReferenceError("invalid reference ID")
        manifest = self.root / reference_id / "manifest.json"
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReferenceError("reference was not found") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceError("reference manifest is unreadable") from exc
        return record if internal else self.public_record(record)

    def resolve(self, reference_id: str) -> ResolvedReference:
        """Resolve a registry ID into its inseparable model inputs."""

        record = self.get(reference_id, internal=True)
        directory = self.root / reference_id
        audio_path = directory / record["audio_file"]
        if not audio_path.is_file():
            raise ReferenceError("registered reference audio is missing")
        transcript = None
        if record.get("transcript_file"):
            try:
                transcript = (directory / record["transcript_file"]).read_text(
                    encoding="utf-8"
                )
            except OSError as exc:
                raise ReferenceError(
                    "registered reference transcript is unreadable"
                ) from exc
        return ResolvedReference(
            audio=str(audio_path),
            transcript=transcript,
            reference_id=reference_id,
            pair_sha256=record["pair_sha256"],
        )

    def delete(self, reference_id: str) -> None:
        """Remove one exact registry entry."""

        self.get(reference_id, internal=True)
        shutil.rmtree(self.root / reference_id)
        _safe_info("deleted reference id=%s", reference_id)

    def rename(self, reference_id: str, name: str) -> dict[str, Any]:
        """Atomically rename metadata without changing pair identity."""

        clean_name = name.strip()
        if not clean_name:
            raise ReferenceError("reference name is required")
        record = self.get(reference_id, internal=True)
        directory = self.root / reference_id
        manifest = directory / "manifest.json"
        updated = dict(record)
        updated["name"] = clean_name
        fd, temporary_name = tempfile.mkstemp(
            prefix=".manifest.", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(json.dumps(updated, indent=2, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, manifest)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        _safe_info("renamed reference id=%s", reference_id)
        return self.public_record(updated)
