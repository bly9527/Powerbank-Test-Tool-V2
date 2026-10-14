"""Load the optional, project-local DirectML runtime before Qt/OpenCV."""
import os
from pathlib import Path
import sys


def prepare_runtime():
    root = Path(__file__).resolve().parents[1] / '.gpu-runtime'
    if (os.name == 'nt' and not getattr(sys, 'frozen', False)
            and not os.environ.get('PB_DISABLE_GPU_RUNTIME')
            and 'onnxruntime' not in sys.modules
            and (root / 'onnxruntime' / '__init__.py').is_file()):
        sys.path.insert(0, str(root))


prepare_runtime()
