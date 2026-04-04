param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.10",
    [string]$VenvDir = "",
    [string]$TorchVersion = "2.3.1",
    [string]$CudaFlavor = "cu118"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $VenvDir) {
    $VenvDir = Join-Path (Split-Path -Parent $ScriptDir) ".venv4090"
}

& (Join-Path $ScriptDir "install_env.ps1") `
    -PythonLauncher $PythonLauncher `
    -PythonVersion $PythonVersion `
    -VenvDir $VenvDir `
    -TorchVersion $TorchVersion `
    -CudaFlavor $CudaFlavor
