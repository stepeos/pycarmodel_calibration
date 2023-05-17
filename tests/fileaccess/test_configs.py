"""module to test fileaccess"""

import os

from pathlib import Path
from unittest import TestCase
import json

from calibration_tool.fileaccess.configs import JSON
from tests.testhelpers import flatten, chdir

test_values = {
    "version": "1.0",
    "nested1": {
        "nested2": "nestedvalue",
        "nested3":{
            "nested4": "doublenestedvalue"
        }
    }
}
TMP_WORKING_DIR = str(Path("tests/resources/fileaccess").absolute())

class TestJSON(TestCase):
    """test the JSON class"""

    @chdir(TMP_WORKING_DIR)
    def setUp(self):
        """fix paths and local paths"""
        self.test_file = Path(Path(os.getcwd()) / "FILE.txt")
        self.test_instance = JSON("FILE.txt")
        with open("available.json", 'w', encoding="utf-8") as file:
            json.dump(test_values, file, indent=4)
        self.readable_test_file = self.test_file.parent / "available.json"

    @chdir(TMP_WORKING_DIR)
    def tearDown(self) -> None:
        file = Path("available.json")
        if file.exists():
            file.unlink()
        if self.test_file.exists():
            self.test_file.unlink()

    @chdir(TMP_WORKING_DIR)
    def _remove_test_file(self):
        if self.test_file.exists():
            self.test_file.unlink()

    @chdir(TMP_WORKING_DIR)
    def test_set_values_write_values(self):
        """method to test setting and writing """
        assert self.test_instance.get_filename()
        self.test_instance.write_file()
        self.test_instance.set_values(test_values)
        self.test_instance.write_file()
        assert self.test_file.read_text(encoding="utf-8") == json.dumps(
            test_values, indent=4)

    @chdir(TMP_WORKING_DIR)
    def test_set_values_write_values_none(self):
        """method to test cleaning of none containing values"""
        new_test_values = test_values.copy()
        new_test_values.update({"ransom": None})
        self.test_instance.set_values(new_test_values)
        assert self.test_instance.is_valid(None) is False
        assert self.test_instance.is_valid(self.test_instance.get_values())

    @chdir(TMP_WORKING_DIR)
    def test_set_values_set_value(self):
        """method to test setting value by path"""
        new_test_values = test_values.copy()
        new_test_values.update({"ransom": None})
        self.test_instance.set_values(new_test_values)
        assert self.test_instance.is_valid(self.test_instance.get_values())
        self.test_instance.set_value(["ransom"], "test")
        assert "test" in flatten(self.test_instance.get_values())

    @chdir(TMP_WORKING_DIR)
    def test_get_value(self):
        """test getting value by path"""
        self.test_instance.set_values(test_values)
        target = test_values["nested1"]["nested3"]
        assert self.test_instance.get_value("nested1","nested3") == target

    @chdir(TMP_WORKING_DIR)
    def test_load_values(self):
        """method to test reading json data"""
        self.test_instance = JSON(str(self.readable_test_file))
        self.test_instance.load_values()
        assert self.test_instance.get_values() == test_values

    @chdir(TMP_WORKING_DIR)
    def test_set_nested_value(self):
        """test setting a nested value"""
        self.test_instance.set_value(["nested1", "nested10"],"success")
        assert self.test_instance.get_value("nested1", "nested10") == "success"