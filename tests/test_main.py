"""Pruebas del punto de entrada del dashboard."""

from __future__ import annotations

import sys
from types import ModuleType
import unittest
from unittest.mock import patch


try:
    import cv2 as _cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = ModuleType("cv2")

try:
    import mediapipe as _mediapipe  # noqa: F401
except ModuleNotFoundError:
    sys.modules["mediapipe"] = ModuleType("mediapipe")

import main


class MainTests(unittest.TestCase):
    def test_main_validates_settings_before_opening_the_gui(self) -> None:
        calls: list[str] = []
        with patch.object(
            main,
            "validate_settings",
            side_effect=lambda: calls.append("validate"),
        ), patch.object(
            main,
            "run_app",
            side_effect=lambda: calls.append("gui"),
        ):
            main.main()

        self.assertEqual(calls, ["validate", "gui"])


if __name__ == "__main__":
    unittest.main()
