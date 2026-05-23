import json
import os
from config import PROFILE_FILE


def load_profile():
    """Return profile dict or None if missing/invalid."""
    if not os.path.exists(PROFILE_FILE):
        return None
    try:
        with open(PROFILE_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or not data.get("name"):
        return None
    return data


def save_profile(name):
    """Persist user profile."""
    with open(PROFILE_FILE, "w") as f:
        json.dump({"name": name}, f, indent=2)


def get_name():
    """Return saved name or None."""
    profile = load_profile()
    return profile["name"] if profile else None
