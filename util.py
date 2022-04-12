from pathlib import Path
import shutil


def clean_dir(d):
    d = Path(d)
    try:
        shutil.rmtree(d)
    except FileNotFoundError:
        # If the output directory doesn't exist, that's fine, we just want a clean
        # slate.
        pass
    d.mkdir(parents=True)
    return d


def ensure_dir(d):
    if not d.exists():
        d.mkdir()
    return d
