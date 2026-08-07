"""Punto de entrada del dashboard de Robot Mimic."""

from config.settings import validate_settings
from modules.m5_gui import run_app


def main() -> None:
    validate_settings()
    run_app()


if __name__ == "__main__":
    main()
