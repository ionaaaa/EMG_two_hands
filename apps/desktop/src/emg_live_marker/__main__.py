"""Command-line entry point for the desktop application."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from emg_live_marker.config import DEFAULT_BAUDRATE
from emg_live_marker.paths import add_path_arguments, resolve_paths_from_args
from emg_live_marker.ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emg_live_marker")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = resolve_paths_from_args(args)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    simulate = args.simulate or args.port is None
    window = MainWindow(simulate=simulate, port=args.port, baudrate=args.baudrate, paths=paths)
    if args.port:
        window.statusBar().showMessage(f"Selected port: {args.port} at {args.baudrate} baud.", 3000)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
