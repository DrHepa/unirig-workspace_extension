param(
  [Parameter(Mandatory=$true)][string]$ArtifactZip,
  [Parameter(Mandatory=$true)][string]$RepoDir,
  [string]$PythonExe = "py -3.11"
)

$ErrorActionPreference = "Stop"
$TempRoot = Join-Path $env:TEMP ("unirig-validate-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
Expand-Archive -Path $ArtifactZip -DestinationPath $TempRoot -Force

$LockFile = Join-Path $TempRoot "runtime-lock.txt"
$Wheelhouse = Join-Path $TempRoot "wheelhouse"
& $PythonExe -m venv (Join-Path $TempRoot "venv")
$VenvPython = Join-Path $TempRoot "venv\\Scripts\\python.exe"

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install --no-index --find-links $Wheelhouse -r $LockFile
& $VenvPython -c "import lightning,pytorch_lightning,transformers,bpy,torch,torch_scatter,torch_cluster,spconv,flash_attn"
& $VenvPython (Join-Path $RepoDir "run.py") --help

Write-Host "Runtime artifact validation passed."
