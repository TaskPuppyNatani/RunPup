"""Restrained dark palette shared by the desktop shell."""

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = QFont()
    font.setFamilies(["Inter", "Noto Sans", "DejaVu Sans", "Arial"])
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#191c24",
        QPalette.ColorRole.WindowText: "#e4e6ee",
        QPalette.ColorRole.Base: "#20242f",
        QPalette.ColorRole.AlternateBase: "#252a36",
        QPalette.ColorRole.Text: "#e4e6ee",
        QPalette.ColorRole.Button: "#2c3140",
        QPalette.ColorRole.ButtonText: "#e4e6ee",
        QPalette.ColorRole.Highlight: "#59417c",
        QPalette.ColorRole.HighlightedText: "#f0eaf7",
        QPalette.ColorRole.ToolTipBase: "#303645",
        QPalette.ColorRole.ToolTipText: "#e4e6ee",
        QPalette.ColorRole.PlaceholderText: "#a6aec1",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#858da1"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QWidget { font-size: 13px; }
        QMainWindow, QWidget#shell { background: #191c24; }
        QFrame#navigation, QFrame#inspector { background: #20242f; border-radius: 12px; }
        QLabel#brand { font-size: 26px; font-weight: 700; color: #c3a1ee; }
        QLabel#heading { font-size: 26px; font-weight: 650; }
        QLabel#emptyHeading { font-size: 21px; font-weight: 600; }
        QLabel#inspectorHeading { font-size: 16px; font-weight: 600; }
        QLabel#muted { color: #aeb7ca; }
        QLabel#local { color: #95d5b1; }
        QPushButton { background: #2d3342; border: 1px solid #41495d;
                      border-radius: 7px; padding: 9px 15px; }
        QPushButton:hover { background: #363e51; border-color: #75829e; }
        QPushButton:focus { border: 2px solid #8ecdd6; }
        QPushButton#primary { background: #73509e; border-color: #9572c1; color: #f0eaf7; }
        QPushButton#primary:hover { background: #8260ac; }
        QPushButton#nav { text-align: left; border: 1px solid transparent;
                         padding: 12px 14px; background: transparent; }
        QPushButton#nav:checked { background: #3d3054; border-color: #9972c5; }
        QPushButton#nav:disabled { color: #8e98af; }
        QTableWidget { background: #20242f; border: 1px solid #394254;
                       border-radius: 8px; gridline-color: #32394a; selection-background-color: #443658; }
        QTableWidget::item { padding: 6px; }
        QHeaderView::section { background: #292f3d; color: #c5cddd; padding: 10px 8px;
                               border: none; border-bottom: 1px solid #444e63; }
        QSplitter::handle { background: #191c24; width: 10px; }
        QToolTip { color: #e4e6ee; background: #303645; border: 1px solid #75829e; }
    """)
