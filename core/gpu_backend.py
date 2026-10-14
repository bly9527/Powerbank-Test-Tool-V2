"""Optional OCR acceleration; no changes to frame selection or segment matching."""
import ctypes
import os
import threading
from uuid import UUID


def directml_devices():
    """Use DXGI enumeration order, which is the DirectML device_id order."""
    if os.name != 'nt':
        return []
    from ctypes import wintypes as w

    class Luid(ctypes.Structure):
        _fields_ = [('low', w.DWORD), ('high', w.LONG)]

    class Description(ctypes.Structure):
        _fields_ = [('name', w.WCHAR * 128), ('vendor', w.UINT),
                    ('device', w.UINT), ('subsystem', w.UINT), ('revision', w.UINT),
                    ('video_memory', ctypes.c_size_t), ('system_memory', ctypes.c_size_t),
                    ('shared_memory', ctypes.c_size_t), ('luid', Luid)]

    def method(obj, index, result, *args):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *args)(table[index])

    factory = ctypes.c_void_p()
    adapters = []
    try:
        iid = (ctypes.c_ubyte * 16).from_buffer_copy(
            UUID('7b7166ec-21c7-44ae-b21a-c9ae321ae369').bytes_le)
        create = ctypes.WinDLL('dxgi').CreateDXGIFactory
        create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        create.restype = w.LONG
        if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
            return []
        for index in range(32):
            adapter = ctypes.c_void_p()
            result = method(factory, 7, w.LONG, w.UINT, ctypes.POINTER(ctypes.c_void_p))(
                factory, index, ctypes.byref(adapter))
            if result < 0:
                break
            try:
                desc = Description()
                if method(adapter, 8, w.LONG, ctypes.POINTER(Description))(
                        adapter, ctypes.byref(desc)) >= 0 and desc.vendor != 0x1414:
                    adapters.append({'id': index, 'name': desc.name,
                                     'memory': desc.video_memory, 'vendor': desc.vendor})
            finally:
                method(adapter, 2, w.ULONG)(adapter)
    except (OSError, ValueError):
        return []
    finally:
        if factory:
            method(factory, 2, w.ULONG)(factory)
    return adapters


def preferred_device(devices):
    return max(devices, key=lambda d: (d['memory'], d['vendor'] == 0x10de)) if devices else None


def error_summary(exc):
    # Keep a driver traceback from expanding the left panel by hundreds of lines.
    return (str(exc).strip() or type(exc).__name__).splitlines()[0][:240]


class AcceleratedOCR:
    """All-or-nothing session replacement with an exact CPU retry on GPU errors.

    Each engine owns its sessions. A lock also prevents concurrent DirectML Run
    calls if an external caller shares an engine. Original CPU sessions remain
    available so a GPU failure does not discard an image or reuse old readings.
    """
    def __init__(self, engine, device_id=0, status_callback=None):
        import onnxruntime as ort
        self.engine = engine
        self.lock = threading.Lock()
        self.active = False
        self.status_callback = status_callback
        self.status = 'OCR：CPU'
        self.originals = []
        try:
            if 'DmlExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('未安装 DirectML 运行库，请运行“安装GPU支持.bat”后重开软件')
            device = next((d for d in directml_devices() if d['id'] == device_id), None)
            if device is None:
                raise RuntimeError('所选显卡已不可用，请重新选择 GPU')
            wrappers = [engine.text_det.infer, engine.text_cls.infer, engine.text_rec.session]
            replacements = []
            for wrapper in wrappers:
                original = wrapper.session
                options = original.get_session_options()
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.log_severity_level = 3
                session = ort.InferenceSession(
                    original._model_path, sess_options=options,
                    providers=[('DmlExecutionProvider', {'device_id': str(device_id)}),
                               'CPUExecutionProvider'])
                if session.get_providers()[0] != 'DmlExecutionProvider':
                    raise RuntimeError('GPU 初始化未成功，运行库已退回 CPU')
                # Handle runtime failures here, rather than an unreported ORT fallback.
                session.disable_fallback()
                replacements.append((wrapper, original, session))
            for wrapper, original, session in replacements:
                self.originals.append((wrapper, original))
                wrapper.session = session
            self.active = True
            self.status = f'OCR：GPU · {device["name"]}（DirectML；部分算子可能使用 CPU）'
        except Exception as exc:
            self._restore()
            self.status = f'OCR：CPU（GPU 未启用：{error_summary(exc)}）'
        self._notify()

    def _notify(self):
        if self.status_callback:
            self.status_callback(self.status)

    def _restore(self):
        for wrapper, original in self.originals:
            wrapper.session = original
        self.originals.clear()
        self.active = False

    def __call__(self, *args, **kwargs):
        with self.lock:
            try:
                return self.engine(*args, **kwargs)
            except Exception as exc:
                if not self.active:
                    raise
                self._restore()
                self.status = f'OCR：CPU（GPU 运行失败，已切换 CPU 重读当前图片：{error_summary(exc)}）'
                self._notify()
                return self.engine(*args, **kwargs)
