# pylint: disable=W0107
"""file with module-specific exceptions"""

class MultipleConfigurations(Exception):
    """raised, when redudant configurations exist"""
    pass
class MissingConfiguration(Exception):
    """raised, when no configuration exists"""
    pass
class FolderNotFound(Exception):
    """raised, when no directory does not exist"""
    pass
class OptimizationFailed(Exception):
    """
    raised, when Optimization worker fails execution the sumo simulation not
    exist
    """
    pass

class InvalidOptions(Exception):
    """
    Raised when the optins for the calibration handler have invalid
    combinations
    """
    pass
class MissingRequirements(Exception):
    """
    Raised when not all requirements are fulfilled
    """
    pass
