# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = []
# Collect package metadata for packages that use importlib.metadata
metadata_packages = [
    'agentpool',
    'llmling-models',
    'pydantic-ai-slim',
    'genai_prices',
    'schemez',
    'tokonomics',
    'pydantic',
    'fastmcp',
    'mcp',
    'typer',
    'rich',
    'httpx',
    'openai',
    'anthropic',
    'google-generativeai',
    'mistralai',
    'opentelemetry-sdk',
    'opentelemetry-api',
    'structlog',
    'sqlmodel',
    'sqlalchemy',
    'pydantic-settings',
    'platformdirs',
]

for pkg in metadata_packages:
    try:
        datas += copy_metadata(pkg, recursive=True)
    except Exception:
        pass  # Package might not be installed or have metadata

# Collect data files for packages that need them
datas += collect_data_files('certifi')
try:
    datas += collect_data_files('tzdata')
except Exception:
    pass
try:
    datas += collect_data_files('zoneinfo')
except Exception:
    pass


a = Analysis(
    ['src/agentpool/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'agentpool',
        'agentpool_cli',
        'agentpool_config',
        'agentpool_commands',
        'agentpool_storage',
        'agentpool_prompts',
        'agentpool_server',
        'agentpool_toolsets',
        'acp',
        'acp.bridge',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
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
    name='agentpool',
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
