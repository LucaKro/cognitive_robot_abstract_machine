"""
Tests that coraplex modules used in simulation import without ROS 2.
"""

import os
import subprocess
import sys
from enum import Enum, StrEnum
from pathlib import Path

# %% importing without ROS


class ImportScript(Enum):
    """
    Scripts that import a module in an interpreter where ROS 2 cannot be imported.
    """

    PERCEPTION = Path(__file__).with_name("import_perception_without_ros.py")


class RosEnvironmentVariable(StrEnum):
    """
    Environment variables a sourced ROS installation sets.
    """

    VERSION = "ROS_VERSION"


def test_perception_imports_without_ros():
    environment_without_ros = {
        variable: value
        for variable, value in os.environ.items()
        if variable != RosEnvironmentVariable.VERSION
    }

    completed = subprocess.run(
        [sys.executable, str(ImportScript.PERCEPTION.value)],
        env=environment_without_ros,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
