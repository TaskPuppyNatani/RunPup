from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from runpup.ui.preferences import ImportPreferences


@pytest.fixture
def preferences(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return ImportPreferences(QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat), home=home)


def reload(preferences):
    return ImportPreferences(QSettings(preferences.settings.fileName(), QSettings.Format.IniFormat), home=preferences.home)


def test_selected_run_remembers_parent_across_settings_reload(preferences, tmp_path):
    run = tmp_path / "arbitrary trainer outputs" / "run-one"
    run.mkdir(parents=True)
    assert preferences.remember_last is True
    preferences.remember_selection(run)
    assert preferences.initial_directory() == str(run.parent)
    assert reload(preferences).initial_directory() == str(run.parent)
    assert str(run.parent) in Path(preferences.settings.fileName()).read_text()


def test_default_and_remember_toggle_persist_and_respect_priority(preferences, tmp_path):
    default = tmp_path / "preferred outputs"
    default.mkdir()
    run = tmp_path / "other outputs" / "run"
    run.mkdir(parents=True)
    preferences.default_directory = default
    preferences.remember_selection(run)
    assert preferences.initial_directory() == str(run.parent)
    preferences.remember_last = False
    loaded = reload(preferences)
    assert loaded.default_directory == str(default)
    assert loaded.remember_last is False
    assert loaded.initial_directory() == str(default)
    ignored = tmp_path / "ignored outputs" / "another-run"
    ignored.mkdir(parents=True)
    loaded.remember_selection(ignored)
    loaded.remember_last = True
    assert reload(loaded).initial_directory() == str(run.parent)


def test_missing_remembered_and_default_paths_fall_through(preferences, tmp_path):
    default = tmp_path / "default"
    default.mkdir()
    run = tmp_path / "old outputs" / "run"
    run.mkdir(parents=True)
    preferences.default_directory = default
    preferences.remember_selection(run)
    run.rmdir()
    run.parent.rmdir()
    assert preferences.initial_directory() == str(default)
    default.rmdir()
    assert preferences.initial_directory() == str(preferences.home)
    preferences.remember_selection(tmp_path / "missing run")
    assert preferences.initial_directory() == str(preferences.home)


def test_unsloth_is_optional_suggestion_not_a_created_directory(preferences, tmp_path):
    unsloth = preferences.home / ".unsloth/studio/outputs"
    assert preferences.initial_directory() == str(preferences.home)
    assert not unsloth.exists()
    unsloth.mkdir(parents=True)
    assert preferences.initial_directory() == str(unsloth)
    preferences.default_directory = tmp_path / "missing"
    assert preferences.initial_directory() == str(unsloth)
    custom = tmp_path / "custom outputs"
    custom.mkdir()
    preferences.default_directory = custom
    assert preferences.initial_directory() == str(custom)


def test_clearing_default_persists_without_creating_folders(preferences, tmp_path):
    preferences.default_directory = tmp_path / "not-created"
    assert not Path(preferences.default_directory).exists()
    preferences.default_directory = ""
    assert reload(preferences).default_directory == ""
    assert reload(preferences).initial_directory() == str(preferences.home)
