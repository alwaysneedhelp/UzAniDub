from pathlib import Path

from dubtool.config import DubConfig


def test_defaults():
    cfg = DubConfig()
    assert cfg.target_language == "uz"
    assert cfg.backends["translate"] == "madlad"
    assert cfg.backends["tts"] == "cosyvoice_navoiy"
    assert isinstance(cfg.models.cosyvoice_base_dir, Path)


def test_yaml_round_trip(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
target_language: uz
keep_intermediate: true
models:
  whisper_model: base
"""
    )
    cfg = DubConfig.from_yaml(yaml_path)
    assert cfg.keep_intermediate is True
    assert cfg.models.whisper_model == "base"
    # untouched fields keep their dataclass defaults
    assert cfg.backends["translate"] == "madlad"
    assert isinstance(cfg.models.cosyvoice_base_dir, Path)


def test_yaml_coerces_string_paths_to_path_objects(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
work_dir: /tmp/somewhere
models:
  cosyvoice_base_dir: /tmp/models/cv2
"""
    )
    cfg = DubConfig.from_yaml(yaml_path)
    assert cfg.work_dir == Path("/tmp/somewhere")
    assert cfg.models.cosyvoice_base_dir == Path("/tmp/models/cv2")
