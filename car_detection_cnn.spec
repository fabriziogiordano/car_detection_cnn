# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

# Path to the model file
model_file = './models/prod/car_detection_cnn_scripted.pt'

a = Analysis(
    ['your_flask_script.py'],  # Replace with your actual script filename
    pathex=[],
    binaries=[],
    datas=[(model_file, 'models/prod')],  # Include model in the executable
    hiddenimports=['torch', 'flask', 'PIL', 'torchvision'],  # Include necessary hidden imports
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='car_detection_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
