"""
module to handle data-config files
"""

from pathlib import Path

from abc import ABC, abstractmethod
from functools import reduce


class File(ABC):
    """dataFile class for file access"""

    def __init__(self, filename):
        self._file = Path(filename).absolute()
        self._values = {}
        self._dirty = False

    def get_filename(self):
        """name returns the filename as string"""
        return str(self._file)

    def get_values(self):
        """read values from file"""
        return self._values

    @classmethod
    def _flatten(cls, cont):
        for vals in cont.values():
            if isinstance(vals, dict):
                yield from cls._flatten(vals)
            else:
                yield vals

    @abstractmethod
    def is_valid(self, update) -> bool:
        """checks if values fit the file format"""
        return None

    @abstractmethod
    def write_file(self) -> bool:
        """save file to filesystem"""
        return None

    @abstractmethod
    def load_values(self) -> None:
        """loads all values from the file"""
        return None

    @classmethod
    def _update(cls, data, update):
        for key, value in update.items():
            if isinstance(value, dict):
                data[key] = cls._update(data.get(key, {}), value)
            else:
                data[key] = value
        return data

    @classmethod
    def _clean_values(cls, values):
        new_values = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, dict):
                if len(list(value.keys())) == 0:
                    continue
                new_values[key] = cls._clean_values(value)
            new_values[key] = value
        return new_values

    def set_values(self, values):
        """set vaules from dict"""
        if values is not None:
            self._values = self._clean_values(values)
            self._dirty = True

    def set_value(self, path, value):
        """"sets the new value"""
        update = {}
        current_level = update
        for idx, part in enumerate(path):
            last = (idx == (len(path) - 1))
            if part not in current_level:
                current_level[part] = {} if not last else value
            current_level = current_level[part]
        if self.is_valid(update):
            self._values = self._update(self._values, update)
            self._dirty = True

    def _deep_get(self, *keys):
        return reduce(lambda d, key: d.get(key, None) if isinstance(d, dict)\
            else None, keys, self._values)

    def get_value(self, *path):
        """returns the value of the file"""
        return self._deep_get(*path)
        