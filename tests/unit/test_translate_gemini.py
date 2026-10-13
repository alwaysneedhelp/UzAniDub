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


class _SequenceModels:
    """Answers each call with the next response, recording every prompt."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append(contents)
        return _FakeResponse(self._responses.pop(0))


def _patch_sequence(monkeypatch, *responses):
    fake_models = _SequenceModels(responses)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.models = fake_models

    monkeypatch.setattr("google.genai.Client", _FakeClient)
    return fake_models


def test_uzbek_is_requested_in_latin_script(monkeypatch):
    # Without a script named, Gemini answered some lines of one run in
    # Cyrillic ("Ва хаёлий масса яратиш учун.") and the rest in Latin.
    fake_models = _patch_client(monkeypatch)
    GeminiTranslator(api_key="fake-key").translate("hello", source_lang="en", target_lang="uz")

    sent = fake_models.last_call["contents"]
    assert "Latin alphabet" in sent
    assert "never Cyrillic" in sent


def test_no_source_words_may_be_left_in(monkeypatch):
    fake_models = _patch_client(monkeypatch)
    GeminiTranslator(api_key="fake-key").translate("hello", source_lang="en", target_lang="uz")

    assert "Leave no English words in the output" in fake_models.last_call["contents"]


def test_names_are_sent_as_hints(monkeypatch):
    fake_models = _patch_client(monkeypatch, response_text="Shuning uchun Cheksizlik texnikasi haqida bilasiz.")
    GeminiTranslator(api_key="fake-key").translate(
        "That's why you know about the Limitless Technique.", source_lang="en", target_lang="uz",
        names=["Limitless Technique"],
    )

    sent = fake_models.last_call["contents"]
    assert "detected as names: Limitless Technique" in sent
    assert "translate it" in sent


def test_no_names_rule_without_names(monkeypatch):
    fake_models = _patch_client(monkeypatch)
    GeminiTranslator(api_key="fake-key").translate("hello", source_lang="en", target_lang="uz")

    assert "detected as names" not in fake_models.last_call["contents"]


def test_a_cyrillic_answer_is_retried_in_latin(monkeypatch):
    fake_models = _patch_sequence(monkeypatch, "Хаёлий техника.", "Xayoliy texnika.")
    result = GeminiTranslator(api_key="fake-key").translate(
        "Imaginary technique.", source_lang="en", target_lang="uz",
    )

    assert result == "Xayoliy texnika."
    assert len(fake_models.calls) == 2
    assert "Cyrillic" in fake_models.calls[1]
    assert "Хаёлий техника." in fake_models.calls[1]


def test_a_leftover_source_word_with_a_suffix_is_retried(monkeypatch):
    # Real output: "the reversal" came back as "Reversalni".
    fake_models = _patch_sequence(
        monkeypatch, "Kuchaytirilganini va Reversalni olib.", "Kuchaytirilganini va teskarisini olib.",
    )
    result = GeminiTranslator(api_key="fake-key").translate(
        "Take the amplified and the reversal.", source_lang="en", target_lang="uz",
    )

    assert result == "Kuchaytirilganini va teskarisini olib."
    assert "Reversalni" in fake_models.calls[1]


def test_retry_is_dropped_when_it_is_no_better(monkeypatch):
    # The model was right to keep a name it wasn't told about; the first,
    # unreworded answer stays.
    fake_models = _patch_sequence(monkeypatch, "Toji keldi.", "Toji yetib keldi.")
    result = GeminiTranslator(api_key="fake-key").translate("Toji is here.", source_lang="en", target_lang="uz")

    assert result == "Toji keldi."
    assert len(fake_models.calls) == 2


def test_hinted_names_and_placeholders_do_not_trigger_a_retry(monkeypatch):
    fake_models = _patch_sequence(monkeypatch, "Gojo XPRESERVEDTERM0Xni ishlatdi.")
    result = GeminiTranslator(api_key="fake-key").translate(
        "Gojo used XPRESERVEDTERM0X.", source_lang="en", target_lang="uz", names=["Gojo"],
    )

    assert result == "Gojo XPRESERVEDTERM0Xni ishlatdi."
    assert len(fake_models.calls) == 1
