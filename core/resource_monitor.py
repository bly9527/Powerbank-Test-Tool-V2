"""Low-frequency, read-only Windows resource sampling for the main window."""
import ctypes
import os
import subprocess

from PyQt5.QtCore import QThread, pyqtSignal


def _hidden_flags():
    return getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def physical_cores():
    if os.name != 'nt':
        return None
    try:
        result = subprocess.check_output(['powershell', '-NoProfile', '-Command',
            '(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum'],
            timeout=8, creationflags=_hidden_flags(), text=True)
        return int(result.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _system_times():
    class FileTime(ctypes.Structure):
        _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_uint32)]

        def value(self):
            return (self.high << 32) + self.low

    idle, kernel, user = FileTime(), FileTime(), FileTime()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        raise OSError('GetSystemTimes failed')
    return idle.value(), kernel.value() + user.value()


def nvidia_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu',
            '--format=csv,noheader,nounits'], timeout=4,
            creationflags=_hidden_flags(), text=True, stderr=subprocess.DEVNULL)
        name, percent = result.splitlines()[0].rsplit(',', 1)
        return name.strip(), int(percent.strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, None


class ResourceMonitor(QThread):
    updated = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        cores = physical_cores()
        logical = os.cpu_count() or 1
        previous = _system_times() if os.name == 'nt' else None
        while self.running:
            for _ in range(20):
                if not self.running:
                    break
                self.msleep(100)
            if not self.running:
                break
            percent = None
            if os.name == 'nt':
                current = _system_times()
                elapsed = current[1] - previous[1]
                if elapsed > 0:
                    percent = max(0, min(100, round(100 * (1 - (current[0] - previous[0]) / elapsed))))
                previous = current
            gpu_name, gpu_percent = nvidia_usage()
            self.updated.emit({'cpu_percent': percent, 'gpu_name': gpu_name,
                'gpu_percent': gpu_percent, 'physical_cores': cores, 'logical_threads': logical})
