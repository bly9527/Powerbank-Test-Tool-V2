# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('core', 'core'), ('ui', 'ui'), ('C:/Users/30583/AppData/Local/Programs/Python/Python312/Lib/site-packages/rapidocr_onnxruntime', 'rapidocr_onnxruntime'), ('C:/Users/30583/AppData/Local/Programs/Python/Python312/Lib/site-packages/onnxruntime', 'onnxruntime')],
    hiddenimports=['shapely', 'pyclipper', 'yaml', 'six', 'tqdm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['rapidocr_onnxruntime', 'onnxruntime'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PB_Test_Tool_v2_Console',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PB_Test_Tool_v2_Console',
)
