from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from core.gpu_backend import AcceleratedOCR, preferred_device
from core.ocr_engine import OCREngine


def fake_engine():
    originals = [Mock(_model_path=f'model{i}.onnx') for i in range(3)]
    engine = Mock()
    engine.text_det = SimpleNamespace(infer=SimpleNamespace(session=originals[0]))
    engine.text_cls = SimpleNamespace(infer=SimpleNamespace(session=originals[1]))
    engine.text_rec = SimpleNamespace(session=SimpleNamespace(session=originals[2]))
    return engine, originals


def sessions(engine):
    return [engine.text_det.infer.session, engine.text_cls.infer.session,
            engine.text_rec.session.session]


@pytest.fixture
def gpu_environment():
    with patch('onnxruntime.get_available_providers', return_value=['DmlExecutionProvider']), \
         patch('core.gpu_backend.directml_devices', return_value=[
             dict(id=1, name='Test GPU', memory=2048, vendor=0x10de)]), \
         patch('onnxruntime.InferenceSession') as factory:
        def make(*_, **kwargs):
            session = Mock()
            session.get_providers.return_value = ['DmlExecutionProvider', 'CPUExecutionProvider']
            return session
        factory.side_effect = make
        yield factory


def test_device_choice_uses_dxgi_id_and_dedicated_memory():
    devices = [dict(id=0, vendor=0x8086, memory=128), dict(id=3, vendor=0x10de, memory=2048)]
    assert preferred_device(devices)['id'] == 3
    assert preferred_device([]) is None


def test_all_three_sessions_use_requested_device(gpu_environment):
    import onnxruntime as ort
    engine, originals = fake_engine()
    callback = Mock()
    accelerated = AcceleratedOCR(engine, 1, callback)
    assert accelerated.active
    assert sessions(engine) != originals
    assert gpu_environment.call_count == 3
    for call in gpu_environment.call_args_list:
        assert call.kwargs['providers'][0] == ('DmlExecutionProvider', {'device_id': '1'})
        assert call.kwargs['sess_options'].enable_mem_pattern is False
        assert call.kwargs['sess_options'].execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    for session in sessions(engine):
        session.disable_fallback.assert_called_once()
    callback.assert_called_once_with(accelerated.status)


def test_partial_init_failure_leaves_all_cpu_sessions(gpu_environment):
    engine, originals = fake_engine()
    good = Mock()
    good.get_providers.return_value = ['DmlExecutionProvider']
    gpu_environment.side_effect = [good, RuntimeError('out of memory')]
    accelerated = AcceleratedOCR(engine, 1)
    assert not accelerated.active
    assert sessions(engine) == originals
    assert 'out of memory' in accelerated.status


def test_silent_provider_fallback_is_reported(gpu_environment):
    engine, originals = fake_engine()
    cpu = Mock()
    cpu.get_providers.return_value = ['CPUExecutionProvider']
    gpu_environment.side_effect = None
    gpu_environment.return_value = cpu
    accelerated = AcceleratedOCR(engine, 1)
    assert not accelerated.active
    assert sessions(engine) == originals


def test_runtime_failure_retries_same_image_on_cpu(gpu_environment):
    engine, originals = fake_engine()
    engine.side_effect = [RuntimeError('device removed'), ('fresh image result', None)]
    callback = Mock()
    accelerated = AcceleratedOCR(engine, 1, callback)
    image = object()
    assert accelerated(image, use_cls=False) == ('fresh image result', None)
    assert sessions(engine) == originals
    assert not accelerated.active
    assert callback.call_count == 2
    assert engine.call_args_list[0] == engine.call_args_list[1]
    assert engine.call_args.args[0] is image


def test_missing_gpu_and_missing_provider_are_cpu_fallbacks(gpu_environment):
    engine, originals = fake_engine()
    assert not AcceleratedOCR(engine, 99).active
    with patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
        accelerated = AcceleratedOCR(engine, 1)
    assert not accelerated.active
    assert '安装GPU支持' in accelerated.status
    assert sessions(engine) == originals
    gpu_environment.assert_not_called()


def test_default_ocr_does_not_initialize_gpu():
    with patch('core.ocr_engine.RapidOCR') as rapid, \
         patch('core.gpu_backend.AcceleratedOCR') as gpu:
        engine = OCREngine(cpu_threads=4)
    rapid.assert_called_once_with(intra_op_num_threads=4, inter_op_num_threads=1)
    gpu.assert_not_called()
    assert engine.backend_status == 'OCR：CPU'
