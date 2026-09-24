# This runtime hook ensures onnxruntime is loaded before PyQt5 to prevent DLL conflicts
import onnxruntime
