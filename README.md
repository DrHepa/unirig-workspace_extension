# UniRig Workspace Extension

## Runtime vendor bootstrap behavior

- `vendor/` en la raíz de la extensión es un **snapshot embebido**.
- El vendor **activo** tras instalar siempre vive en `dependencies/unirig-workspace-v1/vendor` (runtime local).
- La instalación prepara siempre el vendor local:
  - si el snapshot embebido es válido, se copia al runtime local;
  - si no es válido, se reconstruye con `build_vendor.py --dest <runtime_vendor_dir>`.
- Estado `ready` requiere `venv` + vendor local válido + `vendor/unirig/run.py` accesible.

No se cambió la lógica de rigging ni el soporte de entrada `.glb`.
