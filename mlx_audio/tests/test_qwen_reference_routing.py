from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx

from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model


def _routing_model(kind: str):
    model = SimpleNamespace(
        config=SimpleNamespace(tts_model_type=kind),
        sample_rate=24000,
        speech_tokenizer=SimpleNamespace(has_encoder=True),
    )
    model._generate_icl = MagicMock(return_value=iter(["icl"]))
    model.generate_custom_voice = MagicMock(return_value=iter(["named"]))
    model.generate_voice_design = MagicMock(return_value=iter(["designed"]))
    return model


def test_custom_voice_pair_takes_icl_route():
    model = _routing_model("custom_voice")
    with patch(
        "mlx_audio.tts.models.qwen3_tts.qwen3_tts.load_audio",
        return_value=mx.zeros((24000,)),
    ):
        result = list(
            Model.generate(
                model,
                "target",
                ref_audio="voice.wav",
                ref_text="exact transcript",
            )
        )
    assert result == ["icl"]
    model._generate_icl.assert_called_once()
    model.generate_custom_voice.assert_not_called()


def test_custom_voice_without_pair_keeps_named_speaker_route():
    model = _routing_model("custom_voice")
    result = list(Model.generate(model, "target", voice="Ethan"))
    assert result == ["named"]
    model.generate_custom_voice.assert_called_once()
    model._generate_icl.assert_not_called()


def test_custom_voice_pair_is_eligible_for_batch_icl():
    model = _routing_model("custom_voice")
    assert Model.supports_tts_batch(
        model,
        ref_audio=mx.zeros((24000,)),
        ref_text="exact transcript",
    )


def test_icl_cache_identity_includes_revision_audio_and_exact_text():
    model = Model.__new__(Model)
    model._model_revision = "revision-a"
    audio = mx.array([0.0, 0.25, -0.25])
    first = model._icl_cache_key(audio, "words")
    assert first == model._icl_cache_key(audio, "words")
    assert first != model._icl_cache_key(audio, "words ")
    assert first != model._icl_cache_key(mx.array([0.0, 0.25, -0.2]), "words")
    model._model_revision = "revision-b"
    assert first != model._icl_cache_key(audio, "words")


def test_icl_cache_is_lru_bounded():
    model = Model.__new__(Model)
    model._icl_cache = OrderedDict()
    model._icl_cache_max_entries = 2
    model._put_icl_cache(("rev", "one"), 1)
    model._put_icl_cache(("rev", "two"), 2)
    assert model._get_icl_cache(("rev", "one")) == 1
    model._put_icl_cache(("rev", "three"), 3)
    assert list(model._icl_cache) == [("rev", "one"), ("rev", "three")]
