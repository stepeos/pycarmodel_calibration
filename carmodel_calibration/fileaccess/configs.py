"""module to handle config files like data-config file"""

import json
import logging

from carmodel_calibration.fileaccess.fileaccess import File

_LOGGER = logging.getLogger(__name__)

class JSON(File):
    """class to handle JSON files"""

    def load_values(self):
        try:
            with open(self._file, 'r', encoding="utf-8") as file:
                self._values = json.load(file)
        except FileNotFoundError:
            exists = "" if self._file.parent.exists() else "not "
            _LOGGER.error(
                "could not find %s in directory %s which does %s exist",
                self.get_filename(), str(self._file.parent), exists)

    def is_valid(self, update):
        if isinstance(update, dict):
            if None not in list(self._flatten(update)):
                return True
        return False

    def write_file(self) -> bool:
        if self._dirty:
            if self.is_valid(self._values):
                with open(self._file, 'w+', encoding="utf-8") as file:
                    json.dump(self._values, file, indent=4)
                return True
        return False
