"""Command-line entry point for the desktop application."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from emg_live_marker.config import DEFAULT_BAUDRATE
from emg_live_marker.paths import ProjectPaths, add_path_arguments, resolve_paths_from_args
from emg_live_marker.ui.main_window import MainWindow
from emg_live_marker.ui.student_window import StudentMainWindow, TeachingConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emg_live_marker")
    parser.add_argument(
        "--mode",
        choices=("student", "teacher"),
        default="teacher",
        help="application mode; defaults to teacher",
    )
    parser.add_argument("--simulate", action="store_true", help="start with a simulated data source")
    parser.add_argument("--port", help="serial port name, for example COM4")
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"serial baudrate, default {DEFAULT_BAUDRATE}",
    )
    add_path_arguments(parser)
    return parser


def create_window(args: argparse.Namespace, paths: ProjectPaths) -> MainWindow | StudentMainWindow:
    """Create the mode-specific top-level window without starting its event loop."""

    if args.mode == "student":
        return StudentMainWindow(paths=paths)
    simulate = args.simulate or args.port is None
    return MainWindow(simulate=simulate, port=args.port, baudrate=args.baudrate, paths=paths)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = resolve_paths_from_args(args)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        window = create_window(args, paths)
    except TeachingConfigError as exc:
        parser.error(str(exc))
    if args.mode == "teacher" and args.port:
        window.statusBar().showMessage(f"Selected port: {args.port} at {args.baudrate} baud.", 3000)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
