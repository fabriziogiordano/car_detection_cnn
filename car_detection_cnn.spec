# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller import log as logger
import os

# Verify the model file exists
model_file = './models/v2/car_detection_cnn_scripted_quantized.pt'
if not os.path.isfile(model_file):
    logger.error(f"Model file {model_file} does not exist.")
    raise FileNotFoundError(f"Model file {model_file} does not exist.")

a = Analysis(
    ['run_inference_pt_percentages.py'],
    pathex=[],
    binaries=[],
    datas=[(model_file, '.')],  # Ensure this path is correct
    hiddenimports=[],
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
    name='car_detection',
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
