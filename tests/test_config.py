"""Configuration: defaults, environment overrides, validation failures."""

import os

import pytest

import config

ALL_VARIABLES = [
    "OPENAI_API_KEY", "OPENAI_MODEL", "BRAIN_SPARK_PROFILE", "OPENAI_TIMEOUT_SECONDS",
    "BRAIN_SPARK_LOG_LEVEL", "XDG_CONFIG_HOME",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for variable in ALL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_defaults():
    loaded = config.load_config()
    assert loaded.openai_api_key == "test-key"
    assert loaded.openai_model == config.DEFAULT_OPENAI_MODEL
    assert loaded.profile_path == config.default_profile_path()
    assert loaded.request_timeout_seconds == config.DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert loaded.log_level == config.DEFAULT_LOG_LEVEL


def test_default_profile_path_is_private_user_location():
    path = config.default_profile_path()
    assert path.endswith("brain-spark/user_profile.json")
    assert path.startswith(os.path.expanduser("~/.config"))


def test_default_profile_path_resolves_environment_at_load_time(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.load_config().profile_path == str(tmp_path / "brain-spark" / "user_profile.json")
    monkeypatch.setenv("XDG_CONFIG_HOME", "   ")
    assert config.load_config().profile_path.startswith(os.path.expanduser("~/.config"))


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", " another-test-key ")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2")
    monkeypatch.setenv("BRAIN_SPARK_PROFILE", "/tmp/p.json")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("BRAIN_SPARK_LOG_LEVEL", "debug")
    loaded = config.load_config()
    assert loaded.openai_api_key == "another-test-key"
    assert loaded.openai_model == "gpt-5.2"
    assert loaded.profile_path == "/tmp/p.json"
    assert loaded.request_timeout_seconds == 42
    assert loaded.log_level == "DEBUG"


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_variables_fall_back_to_defaults_or_fail(monkeypatch, value):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", value)
    assert config.load_config().request_timeout_seconds == config.DEFAULT_REQUEST_TIMEOUT_SECONDS
    monkeypatch.setenv("OPENAI_MODEL", value)
    with pytest.raises(config.ConfigError, match="OPENAI_MODEL must not be empty"):
        config.load_config()


@pytest.mark.parametrize("value", ["abc", "0", "-5", "1.5", "10s", "9" * 400 + "x"])
def test_invalid_timeout_values(monkeypatch, value):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", value)
    with pytest.raises(config.ConfigError, match="OPENAI_TIMEOUT_SECONDS"):
        config.load_config()


@pytest.mark.parametrize("value", [None, "", "   ", "key\ninjected", "key with spaces", "key\x01", "clé"])
def test_missing_or_invalid_api_key(monkeypatch, value):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if value is not None:
        monkeypatch.setenv("OPENAI_API_KEY", value)
    with pytest.raises(config.ConfigError, match="OPENAI_API_KEY") as error:
        config.load_config()
    if value and value.strip():
        assert value not in str(error.value)


def test_config_repr_hides_api_key():
    assert "test-key" not in repr(config.load_config())


@pytest.mark.parametrize("value", ["TRACE", "verbose", "1"])
def test_invalid_log_level(monkeypatch, value):
    monkeypatch.setenv("BRAIN_SPARK_LOG_LEVEL", value)
    with pytest.raises(config.ConfigError, match="BRAIN_SPARK_LOG_LEVEL"):
        config.load_config()


def test_config_is_immutable():
    loaded = config.load_config()
    with pytest.raises(AttributeError):
        loaded.openai_model = "changed"
