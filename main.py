# 必须在最顶部导入 onnxruntime，避免 Windows 下 C++ 运行时 DLL 冲突
from core.runtime_bootstrap import prepare_runtime
import onnxruntime
import sys
import os
import traceback
import ctypes

def global_exception_handler(exctype, value, tb):
    """全局未捕获异常处理，将崩溃信息写入日志文件并弹窗告知，绝不静默闪退"""
    err_text = "".join(traceback.format_exception(exctype, value, tb))
    print(err_text, file=sys.stderr)
    try:
        with open("crash_log.txt", "w", encoding="utf-8") as f:
            f.write("=== 软件异常退出崩溃日志 ===\n")
            f.write(f"Python 解释器: {sys.executable}\n")
            f.write(f"工作目录: {os.getcwd()}\n")
            f.write(err_text)
    except Exception:
        pass

    try:
        # 使用 Windows 原生弹窗显示错误，避免 Qt 崩溃时无弹窗
        ctypes.windll.user32.MessageBoxW(
            0,
            f"软件运行发生异常：\n\n{str(value)}\n\n详细崩溃日志已保存至 crash_log.txt",
            "充电宝测试工具 - 异常提醒",
            0x10 | 0x10000  # MB_ICONERROR | MB_SETFOREGROUND
        )
    except Exception:
        pass

sys.excepthook = global_exception_handler

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow

def main():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        font = app.font()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        app.setFont(font)

        window = MainWindow()
        window.show()

        sys.exit(app.exec_())
    except Exception as e:
        global_exception_handler(type(e), e, e.__traceback__)

if __name__ == '__main__':
    main()

