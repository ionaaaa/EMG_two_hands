"""Command-line entry point for the desktop application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emg_live_marker.config import DEFAULT_BAUDRATE
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    simulate = args.simulate or args.port is None
    window = MainWindow(simulate=simulate, port=args.port, baudrate=args.baudrate)
    if args.port:
        window.statusBar().showMessage(f"Selected port: {args.port} at {args.baudrate} baud.", 3000)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
