"""Application-wide Qt style sheet."""

APP_QSS = """
QMainWindow {
    background-color: #F3F4F6;
}

QWidget {
    color: #111827;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}

QFrame#Sidebar {
    background-color: #FFFFFF;
    border-right: 1px solid #D1D5DB;
}

QWidget#MainRoot,
QWidget#MainDisplay,
QWidget#TopWorkspace,
QGroupBox#GameDecoderBottom,
QFrame#RightSidebar {
    background-color: #F3F4F6;
}

QFrame#RightSidebar {
    border-left: 1px solid #D1D5DB;
}

QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #F28C28;
    font-weight: 600;
    background-color: #FFFFFF;
}

QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    min-height: 30px;
    padding: 3px 10px;
}

QPushButton:hover {
    background-color: #F3F4F6;
}

QPushButton:pressed {
    background-color: #F28C28;
    color: #FFFFFF;
}

QPushButton:disabled {
    color: #9CA3AF;
    background-color: #F3F4F6;
}

QComboBox,
QLineEdit,
QSpinBox,
QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    min-height: 30px;
    padding: 4px 8px;
    color: #111827;
}

QCheckBox {
    color: #111827;
    min-height: 24px;
    spacing: 8px;
}

QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F9FAFB;
    color: #111827;
    gridline-color: #E5E7EB;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
}

QHeaderView::section {
    background-color: #F3F4F6;
    color: #111827;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #D1D5DB;
}

QStatusBar {
    background-color: #FFFFFF;
    color: #111827;
    border-top: 1px solid #D1D5DB;
}
"""
