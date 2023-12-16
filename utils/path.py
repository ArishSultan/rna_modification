import os
from pathlib import Path


def _find_project_root(marker='.root'):
    path = os.path.abspath(os.curdir)
    while path != '/':
        if os.path.isfile(os.path.join(path, marker)):
            return path
        path = os.path.dirname(path)
    raise Exception("Project root not found. Place a .root file at the project root.")


def get_path(relative_path: str | Path = '') -> Path:
    return Path(os.path.join(_find_project_root(), relative_path))


def get_dump_path(path_in_dump: str | Path = '') -> Path:
    dump = get_path(Path('dump') / path_in_dump)
    dump.mkdir(parents=True, exist_ok=True)

    return dump
