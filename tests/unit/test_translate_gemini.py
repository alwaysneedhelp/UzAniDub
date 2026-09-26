import pytest

from dubtool.backends.translate_gemini import GeminiTranslator


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_call = None

    def generate_content(self, model, contents):
        self.last_call = {"model": model, "contents": contents}
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
