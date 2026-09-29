# Use your own voice model

The dubbing pipeline calls a `TTSBackend` for each translated speech segment.
You can use a model you trained or fine-tuned by selecting `backends.tts: custom`.
The website in `site/` accepts the **finished dubbed video**; it does not run
model training or dubbing in the browser.

## Start with a short experiment

First decide what you want to replace. A short voice reference can change the
speaker identity in a voice-cloning model, while a new **Uzbek TTS checkpoint**
changes pronunciation and synthesis behavior. The current Navoiy backend uses
both CosyVoice2 and Navoiy's Uzbek fine-tuned LLM checkpoint. Navoiy is not a
standalone engine. If you want your own checkpoint with no Navoiy dependency,
fine-tune a model with an inference runtime you can use, then connect it through
the custom backend below. Training a full TTS model from scratch requires far
more data and compute than a personal voice fine-tune.

Record a small pilot first, evaluate it against the current output, and collect
more only where the test fails: Uzbek sounds, names, numbers, emotion, or
timing. The amount of data needed depends on the base model and objective, so
there is no reliable universal hour count.

## Collect and label recordings

1. Record only your own voice or speakers who explicitly agree to voice-model
   training and the intended uses. Keep the consent and any dataset licenses.
2. Write varied, original Uzbek prompts in the script you will synthesize
   (usually Latin). Cover `oʻ`, `gʻ`, `q`, `x`, borrowings, names, numbers,
   dates, questions, short reactions, and longer sentences. Include neutral and
   expressive delivery, but label the style instead of mixing it silently.
3. Use a quiet room and a consistent microphone position. Record lossless WAV,
   mono, at a consistent sample rate (for example 24 kHz or 48 kHz). Avoid
   clipped peaks, background music, echo, and aggressive denoising.
4. Split recordings on sentence boundaries into short clips, usually a few
   seconds each. Give every clip an exact transcript of what was actually
   spoken. Re-record misreads; do not fix a mismatch by changing only the text.
5. Keep a manifest such as:

   ```csv
   audio_path,text,speaker_id,style,split
   audio/000001.wav,Bugun yangi hikoya boshlanadi.,me,neutral,train
   audio/000002.wav,Rostdanmi?,me,surprised,validation
   ```

6. Listen to a sample of each recording session. Reject wrong transcripts,
   clipped or noisy files, duplicate lines, and long silence. Keep a separate
   validation set of unseen sentences and a small final test set that you never
   use to tune the model. Track prompts, recording device, and permission in
   your dataset notes.

The upstream [CosyVoice training example](https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/libritts/cosyvoice2/run.sh)
shows its training data preparation and required files (`wav.scp`, `text`,
`utt2spk`, embeddings, speech tokens, and parquet lists). Convert your curated
manifest to the format required by the model and revision you choose. The
[Mozilla Common Voice guidelines](https://commonvoice.mozilla.org/en/guidelines?tab=sentence)
are useful for checking recordings and transcripts.

## Connect the trained model

Create a Python module available on `PYTHONPATH` with a factory that loads your
checkpoint **once** and returns an object with `sample_rate` and `synthesize`:

```python
# my_voice.py — replace the two model-specific lines with your model's API.
from pathlib import Path
import numpy as np

class MyVoice:
    sample_rate = 24000

    def __init__(self, checkpoint: Path):
        self.model = load_your_model(checkpoint)  # your model's loader

    def synthesize(self, text: str, reference_audio: Path,
                   reference_text: str = "", emotion: str | None = None,
                   speed: float = 1.0) -> np.ndarray:
        audio = self.model.generate(text=text, reference_audio=str(reference_audio),
                                    reference_text=reference_text, speed=speed)
        return np.asarray(audio, dtype=np.float32)  # mono, at sample_rate

def build_tts(config):
    return MyVoice(Path("models/my-voice/checkpoint.pt"))
```

This is an interface example, not a runnable model implementation. Your model
may use a fixed voice and ignore `reference_audio`, `reference_text`, or
`emotion`. Return a one-dimensional mono float32 waveform at `sample_rate`;
the pipeline uses that rate for mixing. If the model lacks a native speed
control, ignore `speed` and let the alignment stage make its limited correction.

Create `my-voice.yaml`:

```yaml
backends:
  translate: madlad
  tts: custom
  transcribe: faster_whisper
  diarize: auto
  separate: demucs
models:
  custom_tts_factory: my_voice:build_tts
```

Then run `dubtool input.mp4 --config my-voice.yaml -o output/episode-01.mp4`.
Listen to the first short scene before processing a full episode. Compare
pronunciation, intelligibility, speaker consistency, emotion, and sync. Review
the translation text too: a perfect voice model cannot repair a mistranslated
line.

The finished video can then be uploaded in the site's studio, provided you
have permission to distribute the episode and its associated audio and art.
