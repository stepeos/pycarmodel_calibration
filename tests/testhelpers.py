"""helper functions for testing"""

import contextlib
import os
from pathlib import Path

def flatten(cont):
    """method to flatten dictionary to values only"""
    for vals in cont.values():
        if isinstance(vals, dict):
            yield from flatten(vals)
        else:
            yield vals

@contextlib.contextmanager
def chdir(target_dir):
    """change to target_dir within context"""
    old_dir = os.getcwd()
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(old_dir)
