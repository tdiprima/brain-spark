"""Profile storage: validation, symlink rejection, atomic replacement, permissions."""

import json
import os
import stat

import pytest

import user_profile as profile
from config import MAX_NAME_LENGTH


@pytest.mark.parametrize("raw_name", ["", "   ", "\n"])
def test_validate_name_rejects_empty(raw_name):
    with pytest.raises(ValueError, match="empty"):
        profile.validate_name(raw_name)


def test_validate_name_length_boundary():
    assert profile.validate_name("a" * MAX_NAME_LENGTH) == "a" * MAX_NAME_LENGTH
    with pytest.raises(ValueError, match="at most"):
        profile.validate_name("a" * (MAX_NAME_LENGTH + 1))


@pytest.mark.parametrize("raw_name", ["Alex; rm -rf /", "<script>", "a\x1bb", "../etc"])
def test_validate_name_rejects_unsafe_characters(raw_name):
    with pytest.raises(ValueError, match="may only contain"):
        profile.validate_name(raw_name)


@pytest.mark.parametrize("raw_name", ["Alex", "  Mary-Jane O'Neil  ", "José"])
def test_validate_name_accepts_normal_names(raw_name):
    assert profile.validate_name(raw_name) == raw_name.strip()


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", "42", json.dumps({}), json.dumps({"name": None}),
     json.dumps({"name": 7}), json.dumps({"name": ["x"]}), json.dumps({"name": ""}),
     json.dumps({"name": "a" * (MAX_NAME_LENGTH + 1)})],
)
def test_load_name_rejects_malformed_profiles(tmp_path, content):
    path = tmp_path / "p.json"
    path.write_text(content)
    assert profile.load_name(str(path)) is None


def test_load_name_missing_file_returns_none(tmp_path):
    assert profile.load_name(str(tmp_path / "missing" / "p.json")) is None


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "deeper" / "p.json"
    profile.save_name(str(path), "  Alex  ")
    assert profile.load_name(str(path)) == "Alex"
    assert json.loads(path.read_text()) == {"name": "Alex"}


def test_profile_rejects_symlinks(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    existing_link = tmp_path / "existing.json"
    existing_link.symlink_to(victim)
    dangling_link = tmp_path / "dangling.json"
    dangling_link.symlink_to(tmp_path / "nowhere")

    for link in (existing_link, dangling_link):
        with pytest.raises(profile.ProfileError, match="symlink"):
            profile.save_name(str(link), "Alex")
        with pytest.raises(profile.ProfileError, match="symlink"):
            profile.load_name(str(link))

    assert victim.read_text() == "keep me"
    assert not (tmp_path / "nowhere").exists()


def test_profile_replace_failure_preserves_old_data(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    profile.save_name(str(path), "Old")
    old_bytes = path.read_bytes()

    def failing_replace(source, destination):
        raise PermissionError("injected replace failure")

    monkeypatch.setattr(profile.os, "replace", failing_replace)
    with pytest.raises(profile.ProfileError, match="Could not save profile"):
        profile.save_name(str(path), "New")

    assert path.read_bytes() == old_bytes
    leftovers = [entry for entry in tmp_path.iterdir() if entry.name != "p.json"]
    assert leftovers == []


def test_profile_permissions(tmp_path):
    directory = tmp_path / "brain-spark"
    path = directory / "p.json"
    profile.save_name(str(path), "Alex")
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_save_name_rejects_invalid_before_touching_disk(tmp_path):
    path = tmp_path / "p.json"
    with pytest.raises(ValueError):
        profile.save_name(str(path), "")
    assert not path.exists()
