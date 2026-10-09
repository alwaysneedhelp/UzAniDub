import pytest

from dubtool.backends.translate_gemini import GeminiTranslator


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_call = None

    def generate_content(self, model, contents, config=None):
        self.last_call = {"model": model, "contents": contents, "config": config}
        return _FakeResponse(self._response_text)


def _patch_client(monkeypatch, response_text="Salom dunyo"):
    """Returns the `.models` fake so tests can inspect what was sent."""
    fake_models = _FakeModels(response_text)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = fake_models

    monkeypatch.setattr("google.genai.Client", _FakeClient)
    return fake_models


def test_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiTranslator(api_key=None)


def test_translates_using_the_configured_model(monkeypatch):
    fake_models = _patch_client(monkeypatch)
    translator = GeminiTranslator(model_name="gemini-2.5-flash", api_key="fake-key")

    result = translator.translate("Hello, for real?", source_lang="en", target_lang="uz")

    assert result == "Salom dunyo"
    assert fake_models.last_call["model"] == "gemini-2.5-flash"
    assert "Hello, for real?" in fake_models.last_call["contents"]
    assert "English" in fake_models.last_call["contents"]
    assert "Uzbek" in fake_models.last_call["contents"]


def test_strips_wrapping_quotes(monkeypatch):
    _patch_client(monkeypatch, response_text='"Salom dunyo"')
    translator = GeminiTranslator(api_key="fake-key")

    assert translator.translate("hi", source_lang="en", target_lang="uz") == "Salom dunyo"


def test_empty_text_short_circuits_without_calling_the_api(monkeypatch):
    fake_models = _patch_client(monkeypatch)
    translator = GeminiTranslator(api_key="fake-key")

    assert translator.translate("   ", source_lang="en", target_lang="uz") == ""
    assert fake_models.last_call is None


def test_reads_api_key_from_environment(monkeypatch):
    _patch_client(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    GeminiTranslator()  # should not raise


def test_thinking_is_disabled_so_reasoning_cannot_leak_into_the_text(monkeypatch):
    # Observed on real output: thinking is on by default for the 2.5 models,
    # and on a prompt this short the trace came back inside response.text
    # ("THINK: The user wants...") followed by four numbered candidates. All
    # of it reached the TTS and was synthesized as speech -- 17s of audio for
    # a one-second line. There is nothing here worth reasoning about.
    _patch_client(monkeypatch)
    translator = GeminiTranslator(api_key="fake-key")
    assert translator._config.thinking_config.thinking_budget == 0


def test_length_hint_is_sent_when_the_slot_duration_is_known(monkeypatch):
    # Uzbek runs longer than English: on a real clip 13 of 16 segments
    # overran their slot and mix.py cut them off mid-word. A seq2seq model
    # can't be asked for concision; this one can.
    fake_models = _patch_client(monkeypatch)
    translator = GeminiTranslator(api_key="fake-key")

    translator.translate("hello", source_lang="en", target_lang="uz", target_duration=2.0)

    sent = fake_models.last_call["contents"]
    assert "LENGTH TARGET" in sent
    assert "2.0 seconds" in sent
    assert "5 words" in sent          # 2.0s * 2.5 words/sec
    # The budget is a target, not a ceiling: over-constraining it made Gemini
    # buy brevity with accuracy ("Is he high?" went from "U nashamandmi?" to
    # "U tinib qolibdimi?" -- "has he gone quiet?").
    assert "Accuracy outranks length" in sent


def test_no_length_hint_when_no_duration_is_given(monkeypatch):
    fake_models = _patch_client(monkeypatch)
    translator = GeminiTranslator(api_key="fake-key")

    translator.translate("hello", source_lang="en", target_lang="uz")

    assert "LENGTH TARGET" not in fake_models.last_call["contents"]


def test_a_tiny_slot_still_asks_for_something_sayable(monkeypatch):
    # Floor the word budget: 0.2s * 2.5 would round to 0, and asking for a
    # zero-word translation gets nonsense back.
    fake_models = _patch_client(monkeypatch)
    translator = GeminiTranslator(api_key="fake-key")

    translator.translate("no", source_lang="en", target_lang="uz", target_duration=0.2)

    assert "2 words" in fake_models.last_call["contents"]
