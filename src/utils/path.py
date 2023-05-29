import os
from pathlib import Path


def _find_project_root(marker='.root'):
    path = os.path.abspath(os.curdir)
    while path != '/':
        if os.path.isfile(os.path.join(path, marker)):
            return path
        path = os.path.dirname(path)
    raise Exception("Project root not found. Place a .root file at the project root.")


def get_path(relative_path):
    """
    Generates a platform-independent absolute path from a project-relative path.

    The function locates the root of the project by searching for a file named ".root",
    then appends the input relative path to this root path. This ensures the function
    can generate consistent paths regardless of the current working directory.

    Args:
        relative_path (str): The path relative to the root of the project. For instance,
            to reach a file "data.csv" in a subdirectory "data", the relative path would
            be 'data/data.csv'.

    Returns:
        str: The absolute path of the file or directory, which can be used with various
            file and directory operations.

    Raises:
        Exception: If a ".root" file cannot be found by traversing up from the current
            working directory.

    Example:
        >>> from src.utils import get_path
        >>> print(get_path('data/data.csv'))
        '/absolute/path/to/rna_modification/data/data.csv'
    """
    return Path(os.path.join(_find_project_root(), relative_path))
