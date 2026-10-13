"""Small UI guards shared by the main and sample-library windows."""
from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QAbstractSpinBox, QComboBox


class WheelChangeGuard(QObject):
    """Prevent a wheel passing over an input from changing its value."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched, event)


def disable_wheel_changes(parent):
    guard = WheelChangeGuard(parent)
    widgets = parent.findChildren(QAbstractSpinBox) + parent.findChildren(QComboBox)
    for widget in widgets:
        widget.installEventFilter(guard)
    return guard
