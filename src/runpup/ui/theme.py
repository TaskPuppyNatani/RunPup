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
        QTreeWidget#comparisonTree { background: #20242f; border: 1px solid #394254;
                                    border-radius: 8px; selection-background-color: #443658; }
        QTreeWidget#comparisonTree::item { padding: 5px 3px; }
        QTreeWidget#comparisonTree:focus { border-color: #8ecdd6; }
        QListView#runList { background: transparent; border: none; outline: none; }
        QListView#runList:focus { border: 1px solid #75829e; border-radius: 8px; }
        QHeaderView::section { background: #292f3d; color: #c5cddd; padding: 10px 8px;
                               border: none; border-bottom: 1px solid #444e63; }
        QSplitter::handle { background: #191c24; width: 10px; }
        QToolTip { color: #e4e6ee; background: #303645; border: 1px solid #75829e; }
        QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
        QFrame#sourcePanel, QFrame#reviewCard, QFrame#warningCard,
        QFrame#blockingCard, QFrame#noteCard {
            background: #20242f; border: 1px solid #3b465a; border-radius: 8px;
        }
        QFrame#warningCard { border-color: #88703c; background: #282720; }
        QFrame#reviewCard[conflict="true"] { border-color: #b56480; }
        QFrame#blockingCard { border-color: #88506c; background: #2b222d; }
        QFrame#noteCard { border-color: #3e657d; background: #202b38; }
        QLabel#sectionHeading { font-size: 15px; font-weight: 650; color: #e0dceb; padding-bottom: 5px; }
        QLabel#stage { color: #8994ab; border-top: 2px solid #394154; padding: 10px 1px 3px; }
        QLabel#stage[active="true"] { color: #d2a9ff; border-color: #b282e8; font-weight: 600; }
        QLabel#stage[complete="true"] { color: #a9cbbf; border-color: #8061ab; }
        QLabel#eyebrow { font-size: 10px; color: #a4b4cd; letter-spacing: 1px; }
        QLabel#sourcePath { color: #abdae3; }
        QLabel#fieldTitle { color: #cdd5e4; font-size: 12px; }
        QLabel#stateBadge { border: 1px solid #6a597c; background: #30273d;
                            color: #d4b9f6; border-radius: 4px; padding: 2px 4px; font-size: 9px; }
        QLabel#stateBadge[state="detected"] { color: #a2e1ba; border-color: #426c55; background: #23382e; }
        QLabel#stateBadge[state="conflict"] { color: #ffb3c3; border-color: #ae576b; background: #492b39; }
        QLabel#stateBadge[state="user_edited"] { color: #9fe0eb; border-color: #4c7b87; background: #253b45; }
        QLabel#evidenceText { color: #abb8cf; font-size: 10px; }
        QLabel#errorText { color: #ffc1ca; font-size: 11px; }
        QLabel#counter { border: 1px solid #3c455a; border-radius: 6px; background: #252b38;
                        font-size: 11px; padding: 9px 2px; color: #b4e0ce; }
        QLabel#counter[state="needs_attention"] { color: #ceb5ed; }
        QLabel#counter[state="unavailable"] { color: #abb8cf; }
        QLabel#counter[state="conflict"] { color: #f2a5ba; }
        QLabel#counter[state="artifacts"] { color: #a1cce9; }
        QLineEdit, QComboBox { background: #191f2b; border: 1px solid #45536a; border-radius: 4px;
                              padding: 5px 7px; min-height: 17px; selection-background-color: #665088; }
        QLineEdit:focus, QComboBox:focus { border: 1px solid #a38dd0; }
        QTextEdit { background: #191f2b; border: 1px solid #45536a; border-radius: 4px;
                    padding: 7px; selection-background-color: #665088; }
        QTextEdit:focus { border: 1px solid #a38dd0; }
        QToolButton#detailDisclosure { text-align: left; color: #b9d8e9; padding: 6px;
                                      font-size: 12px; }
        QComboBox::drop-down { border: none; width: 18px; }
        QComboBox QAbstractItemView { background: #252c3a; selection-background-color: #59417c; }
        QPushButton#compact { padding: 4px 7px; font-size: 11px; }
        QToolButton { border: 1px solid #45536a; border-radius: 4px; padding: 4px; }
        QToolButton:focus { border: 2px solid #8ecdd6; }
        QToolButton#evidence { border: none; color: #a4b8d2; font-size: 10px; text-align: left; padding: 1px; }
        QToolButton#evidence:hover { color: #cedff4; }
        QToolButton#evidence:focus { border: 1px solid #8ecdd6; }
        QPushButton:disabled { color: #9699ad; background: #282b36; border-color: #3e4354; }
        QPushButton#primary:disabled { background: #3c3050; color: #a99bb8; border-color: #554466; }
        QProgressBar { border: none; background: #303646; }
        QProgressBar::chunk { background: #9871c6; }
        QScrollBar:vertical { background: #1b202a; width: 9px; margin: 0; }
        QScrollBar::handle:vertical { background: #4b5267; border-radius: 4px; min-height: 24px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QListWidget#artifactList { background: #1d2330; border: none; font-size: 11px; }
        QListWidget#artifactList::item { padding: 6px; border-bottom: 1px solid #303b4c; }
        QListWidget#storageRootList { background: #20242f; border: 1px solid #394254;
                                     border-radius: 8px; }
        QListWidget#storageRootList:focus { border-color: #8ecdd6; }
        QListWidget#storageRootList::item { padding: 9px; border-bottom: 1px solid #394254; }
        QListWidget#storageRootList::item:selected { background: #3d3054; color: #f0eaf7; }
    """)
