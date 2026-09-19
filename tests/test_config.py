"""Configuration: defaults, environment overrides, validation failures."""

import pytest

import config

ALL_VARIABLES = [
    "OLLAMA_URL", "OLLAMA_MODEL", "BRAIN_SPARK_PROFILE", "OLLAMA_TIMEOUT_SECONDS",
    "BRAIN_SPARK_LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for variable in ALL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_defaults():
    loaded = config.load_config()
    assert loaded.ollama_url == config.DEFAULT_OLLAMA_URL
    assert loaded.ollama_model == config.DEFAULT_OLLAMA_MODEL
    assert loaded.profile_path == config.DEFAULT_PROFILE_PATH
    assert loaded.request_timeout_seconds == config.DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert loaded.log_level == config.DEFAULT_LOG_LEVEL


def test_default_profile_path_is_private_user_location():
    assert config.DEFAULT_PROFILE_PATH.endswith("brain-spark/user_profile.json")
    assert config.DEFAULT_PROFILE_PATH != "user_profile.json"


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", " https://remote:1/api/generate ")
    monkeypatch.setenv("OLLAMA_MODEL", "other:tag")
    monkeypatch.setenv("BRAIN_SPARK_PROFILE", "/tmp/p.json")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("BRAIN_SPARK_LOG_LEVEL", "debug")
    loaded = config.load_config()
    assert loaded.ollama_url == "https://remote:1/api/generate"
    assert loaded.ollama_model == "other:tag"
    assert loaded.profile_path == "/tmp/p.json"
    assert loaded.request_timeout_seconds == 42
    assert loaded.log_level == "DEBUG"


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_variables_fall_back_to_defaults_or_fail(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", value)
    assert config.load_config().request_timeout_seconds == config.DEFAULT_REQUEST_TIMEOUT_SECONDS
    monkeypatch.setenv("OLLAMA_MODEL", value)
    with pytest.raises(config.ConfigError, match="OLLAMA_MODEL must not be empty"):
        config.load_config()


@pytest.mark.parametrize("value", ["abc", "0", "-5", "1.5", "10s", "9" * 400 + "x"])
def test_invalid_timeout_values(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", value)
    with pytest.raises(config.ConfigError, match="OLLAMA_TIMEOUT_SECONDS"):
        config.load_config()


@pytest.mark.parametrize("value", ["localhost:11434", "ftp://x", "file:///etc/passwd", "javascript:alert(1)"])
def test_invalid_url_scheme(monkeypatch, value):
    monkeypatch.setenv("OLLAMA_URL", value)
    with pytest.raises(config.ConfigError, match="OLLAMA_URL"):
        config.load_config()


@pytest.mark.parametrize("value", ["TRACE", "verbose", "1"])
def test_invalid_log_level(monkeypatch, value):
    monkeypatch.setenv("BRAIN_SPARK_LOG_LEVEL", value)
    with pytest.raises(config.ConfigError, match="BRAIN_SPARK_LOG_LEVEL"):
        config.load_config()


def test_config_is_immutable():
    loaded = config.load_config()
    with pytest.raises(AttributeError):
        loaded.ollama_model = "changed"
