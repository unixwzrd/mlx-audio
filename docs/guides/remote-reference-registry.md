# Remote Reference Registry (unixwzrd fork)

MLX-Audio `0.5.9+unixwzrd.1` adds remote-safe voice cloning while retaining compatible `ref_audio` and `ref_text` requests for trusted local bridges.

The fork adds:

- Atomic immutable audio/transcript registration under `${XDG_DATA_HOME:-~/.local/share}/mlx-audio/references`.
- Registered `reference_id` and inline base64 `reference` inputs for `/v1/audio/speech`.
- Exact transcript preservation, authenticated registry mutation, upload and duration bounds, and path allowlisting without remote URL fetching.
- Qwen CustomVoice reference-pair precedence and a 32-entry SHA-256 LRU for ICL encodings.
- `/v1/audio/capabilities` and safe reference discovery that expose no audio, transcript content, or local paths.

Set `MLX_AUDIO_REFERENCE_ADMIN_TOKEN` before creating, renaming, or deleting references. Set `MLX_AUDIO_REFERENCE_ROOTS` to an operating-system-path-separated list only when trusted legacy server paths are required. Registry reads and speech use are intentionally token-free so the service can remain compatible with existing OpenAI TTS clients behind the LLM-Ops bridge.

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/audio/references \
  -H "Authorization: Bearer $MLX_AUDIO_REFERENCE_ADMIN_TOKEN" \
  -F 'name=reference-a' \
  -F 'audio=@reference-a.wav;type=audio/wav' \
  -F 'transcript=<reference-a.txt'
```

Public display metadata can be renamed atomically without changing the immutable pair or its `reference_id`:

```bash
curl -fsS -X PATCH http://127.0.0.1:8000/v1/audio/references/ref_0123456789abcdef0123456789abcdef \
  -H "Authorization: Bearer $MLX_AUDIO_REFERENCE_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"name":"Mia1"}'
```

Qwen Base and patched CustomVoice cloning require both audio and its exact transcript. Chatterbox, Confucius, and ZONOS2 can register audio-only references by supplying their model ID during registration. Text without audio is never accepted. A built-in `voice` and any clone reference are mutually exclusive at the HTTP layer.

This registry guarantees atomic pair identity; it does not prove that the transcript semantically matches the audio.
