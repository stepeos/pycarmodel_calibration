"""module with helper classes for tests"""
import os
import glob
import shutil
from pathlib import Path

from textwrap import dedent

class TmpDir:
    """helper class that makes a new dir"""
    tmp_work_dir = Path(".tmp").absolute()
    base_dir = Path(os.getcwd())

    def prepare_tmpdir(self):
        """prepares temporary directory and cd's into it"""
        if self.tmp_work_dir.exists():
            if self.tmp_work_dir.is_file():
                self.tmp_work_dir.unlink()
        if not self.tmp_work_dir.exists():
            self.tmp_work_dir.mkdir()
        os.chdir(self.tmp_work_dir)
        for item in glob.glob("*"):
            item = Path(item)
            if item.is_file():
                item.unlink()
                continue
            shutil.rmtree(item)

    def clean_tmpdir(self):
        """removes all files and directories recursively"""
        os.chdir(self.tmp_work_dir)
        for item in glob.glob("*"):
            item = Path(item)
            if item.is_file():
                item.unlink()
                continue
            shutil.rmtree(item)


    def __exit__(self, *_):
        os.chdir(self.base_dir)

    def create_file(self, filename, content):
        """create test_file"""
        content = dedent(content)
        with open(filename, 'w', encoding="utf-8") as file:
            file.write(content)
