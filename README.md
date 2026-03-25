# UniRig Workspace Tool

Workspace-tool extension for Modly.

What it does:
- takes the currently selected mesh already present in the workspace;
- bootstraps a local isolated UniRig runtime under Modly dependencies;
- runs the upstream UniRig skeleton stage, then the skinning stage;
- merges the predicted rig back into the original mesh and returns a rigged `.glb` to the same workspace collection.

## Runtime layout

By default the tool stores its runtime cache under:

- `Settings.dependenciesDir/unirig-workspace-v1/`

If Modly settings are unavailable, it falls back to:

- `extensions/unirig-workspace-v1/_runtime/`

The runtime contains:
- a Python venv dedicated to UniRig;
- a local checkout/cache of the official UniRig repo;
- stage logs and bootstrap state.

## Environment overrides

Useful overrides when packaging or debugging:

- `MODLY_UNIRIG_REPO_DIR`
  - Use an already downloaded local UniRig repo instead of auto-downloading the upstream tarball.
- `MODLY_UNIRIG_RUNTIME_DIR`
  - Override the runtime/cache directory.
- `MODLY_UNIRIG_FORCE_BOOTSTRAP=1`
  - Recreate/reinstall the UniRig runtime on the next execution.
- `MODLY_UNIRIG_TORCH_SPEC`
  - Override the torch install spec used by the isolated venv.
- `MODLY_UNIRIG_TORCH_INDEX_URL`
  - Optional extra index for CUDA-specific torch wheels.
- `MODLY_UNIRIG_SPCONV_PACKAGE`
  - Override the spconv package name (default: `spconv-cu120`).
- `MODLY_UNIRIG_PYG_WHEEL_URL`
  - Override the PyG wheel index URL for `torch_scatter` / `torch_cluster`.
- `MODLY_UNIRIG_ENFORCE_GPU=0`
  - Allow bootstrap/inference without enforcing CUDA availability.
- `MODLY_UNIRIG_MIN_VRAM_GB`
  - Override the minimum VRAM requirement (default: `8`).
- `MODLY_UNIRIG_ENABLE_FLASH_ATTN=1`
  - Keep `flash_attn` from upstream requirements instead of filtering it out during bootstrap.

## Notes

- The upstream UniRig project currently expects Python 3.11 and a CUDA GPU.
- The tool intentionally isolates UniRig dependencies from Modly's main runtime.
- Intermediate stage FBX files are temporary; the persistent output returned to the workspace is a new rigged `.glb` plus `*.rigmeta.json`.
