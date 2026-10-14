import io
import struct
import wave
from pathlib import Path

import pytest

from mlx_audio.reference_registry import (
    ReferenceError,
    ReferenceRegistry,
    resolve_allowed_file,
)


def wav_bytes(seconds: float = 1.1, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(
            b"".join(
                struct.pack("<h", 500 if index % 2 else -500)
                for index in range(int(seconds * sample_rate))
            )
        )
    return buffer.getvalue()


def test_registry_preserves_exact_pair_and_survives_restart(tmp_path):
    registry = ReferenceRegistry(tmp_path / "references")
    transcript = "  Exact transcript.\nDo not normalize this.  \n"
    record = registry.register(
        audio=wav_bytes(),
        filename="voice.wav",
        transcript=transcript,
        name="Neutral",
        language="en",
        emotion="neutral",
    )

    restarted = ReferenceRegistry(tmp_path / "references")
    resolved = restarted.resolve(record["id"])
    assert resolved.transcript == transcript
    assert resolved.pair_sha256 == record["pair_sha256"]
    assert "audio_file" not in record
    assert "transcript_file" not in record
    assert transcript not in str(restarted.list())


def test_registry_identity_changes_with_exact_transcript(tmp_path):
    registry = ReferenceRegistry(tmp_path)
    audio = wav_bytes()
    first = registry.register(
        audio=audio, filename="v.wav", transcript="Hello", name="one"
    )
    duplicate = registry.register(
        audio=audio, filename="v.wav", transcript="Hello", name="same pair"
    )
    whitespace = registry.register(
        audio=audio, filename="v.wav", transcript="Hello ", name="two"
    )
    assert first["id"] == duplicate["id"]
    assert first["id"] != whitespace["id"]


def test_audio_only_registration_is_explicit(tmp_path):
    registry = ReferenceRegistry(tmp_path)
    with pytest.raises(ReferenceError, match="transcript is required"):
        registry.register(
            audio=wav_bytes(), filename="v.wav", transcript=None, name="voice"
        )
    record = registry.register(
        audio=wav_bytes(),
        filename="v.wav",
        transcript=None,
        name="voice",
        transcript_required=False,
    )
    assert record["capabilities"]["clone_mode"] == "audio"


@pytest.mark.parametrize(
    ("filename", "audio", "message"),
    [
        ("voice.txt", wav_bytes(), "WAV, FLAC, MP3, or M4A"),
        ("voice.wav", b"not audio", "could not be decoded"),
        ("voice.wav", wav_bytes(0.5), "between 1 and 30 seconds"),
    ],
)
def test_registry_rejects_invalid_audio(tmp_path, filename, audio, message):
    with pytest.raises(ReferenceError, match=message):
        ReferenceRegistry(tmp_path).register(
            audio=audio,
            filename=filename,
            transcript="text",
            name="voice",
        )


def test_allowed_file_rejects_urls_traversal_and_missing_roots(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    sample = root / "sample.wav"
    sample.write_bytes(wav_bytes())
    assert resolve_allowed_file(str(sample), (root.resolve(),)) == sample.resolve()
    with pytest.raises(ReferenceError, match="URLs"):
        resolve_allowed_file("https://example.test/sample.wav", (root.resolve(),))
    with pytest.raises(ReferenceError, match="outside configured roots"):
        resolve_allowed_file(str(root / ".." / "escape.wav"), (root.resolve(),))
    with pytest.raises(ReferenceError, match="disabled"):
        resolve_allowed_file(str(sample), ())


def test_delete_removes_only_selected_reference(tmp_path):
    registry = ReferenceRegistry(tmp_path)
    record = registry.register(
        audio=wav_bytes(), filename="v.wav", transcript="text", name="voice"
    )
    registry.delete(record["id"])
    with pytest.raises(ReferenceError, match="not found"):
        registry.get(record["id"])


def test_registry_rename_preserves_pair_identity_and_files(tmp_path):
    registry = ReferenceRegistry(tmp_path)
    audio = wav_bytes()
    transcript = "Exact transcript."
    record = registry.register(
        audio=audio,
        filename="voice.wav",
        transcript=transcript,
        name="Legacy source name",
        language="en",
        emotion="warm",
    )

    renamed = registry.rename(record["id"], "Mia1")

    assert renamed["id"] == record["id"]
    assert renamed["pair_sha256"] == record["pair_sha256"]
    assert renamed["audio_sha256"] == record["audio_sha256"]
    assert renamed["name"] == "Mia1"
    resolved = registry.resolve(record["id"])
    assert Path(resolved.audio).read_bytes() == audio
    assert resolved.transcript == transcript


def test_registry_logs_metadata_without_transcript_or_audio(tmp_path, caplog):
    transcript = "private exact transcript"
    audio = wav_bytes()
    with caplog.at_level("INFO"):
        record = ReferenceRegistry(tmp_path).register(
            audio=audio,
            filename="v.wav",
            transcript=transcript,
            name="voice",
        )
    rendered = caplog.text
    assert record["id"] in rendered
    assert record["pair_sha256"][:12] in rendered
    assert transcript not in rendered
    assert repr(audio[:20]) not in rendered
