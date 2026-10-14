"""Keep validation isolated from an analysis running in the desktop app."""
import onnxruntime  # Load its C++ runtime before Qt/OpenCV on Windows.
import pytest
from PyQt5.QtCore import QSettings


@pytest.fixture(scope='session', autouse=True)
def isolated_ui_settings(tmp_path_factory):
    original = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                      str(tmp_path_factory.mktemp('ui-settings')))
    yield
    QSettings.setDefaultFormat(original)
