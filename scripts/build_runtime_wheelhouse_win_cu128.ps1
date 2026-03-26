param(
  [string]$WorkDir = ".\\dist\\runtime-win-cu128-stable",
  [string]$PythonExe = "py -3.11",
  [string]$RuntimeProfile = "win-cu128-stable"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$WheelhouseDir = Join-Path $WorkDir "wheelhouse"
$LockFile = Join-Path $WorkDir "runtime-lock.txt"
$ManifestFile = Join-Path (Resolve-Path ".").Path "runtime-manifest.win-cu128-stable.json"

if (Test-Path $WheelhouseDir) { Remove-Item -Recurse -Force $WheelhouseDir }
New-Item -ItemType Directory -Force -Path $WheelhouseDir | Out-Null

& $PythonExe -m venv (Join-Path $WorkDir "builder-venv")
$VenvPython = Join-Path $WorkDir "builder-venv\\Scripts\\python.exe"

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
& $VenvPython -m pip install spconv-cu120
& $VenvPython -m pip install torch-scatter==2.1.2+pt27cu128 torch-cluster==1.6.3+pt27cu128 -f https://data.pyg.org/whl/torch-2.7.1+cu128.html --only-binary=:all:
& $VenvPython -m pip install flash_attn==2.7.4.post1 --no-build-isolation

@(
  "torch==2.7.1+cu128",
  "torchvision==0.22.1+cu128",
  "torchaudio==2.7.1+cu128",
  "spconv-cu120",
  "torch-scatter==2.1.2+pt27cu128",
  "torch-cluster==1.6.3+pt27cu128",
  "flash_attn==2.7.4.post1"
) | Set-Content -Encoding utf8 $LockFile

& $VenvPython -m pip download --dest $WheelhouseDir -r $LockFile --index-url https://download.pytorch.org/whl/cu128
& $VenvPython -m pip download --dest $WheelhouseDir torch-scatter==2.1.2+pt27cu128 torch-cluster==1.6.3+pt27cu128 -f https://data.pyg.org/whl/torch-2.7.1+cu128.html --only-binary=:all:

Compress-Archive -Path $WheelhouseDir, $LockFile -DestinationPath (Join-Path $WorkDir "unirig-runtime-wheelhouse-win-cu128-stable.zip") -Force
Write-Host "Runtime wheelhouse bundle created. Update sha256 in $ManifestFile before release."
