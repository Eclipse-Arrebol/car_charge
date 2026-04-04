param(
    [ValidateSet('cpu', 'cu118', 'cu121', 'cu126', 'cu128')]
    [string]$CudaFlavor = "cu128",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.10",
    [string]$VenvDir = "",
    [string]$TorchVersion = "2.7.1",
    [string]$TorchVisionVersion = "",
    [string]$TorchAudioVersion = "",
    [string]$TorchGeometricVersion = "",
    [string]$NumpyVersion = "1.26.4",
    [string]$PipIndexUrl = "",
    [string]$PipFallbackIndexUrl = "https://pypi.org/simple"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $VenvDir) {
    $VenvDir = Join-Path $ProjectRoot ".venv"
}

function Get-PythonCommand {
    if ([string]::IsNullOrWhiteSpace($PythonVersion)) {
        return @($PythonLauncher)
    }

    if ($PythonLauncher -eq "py") {
        return @($PythonLauncher, "-$PythonVersion")
    }

    return @($PythonLauncher)
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $pythonCmd = Get-PythonCommand
    if ($pythonCmd.Length -gt 1) {
        & $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length - 1)] @Arguments
    } else {
        & $pythonCmd[0] @Arguments
    }
}

function Get-VenvPython {
    return Join-Path $VenvDir "Scripts\python.exe"
}

function Invoke-VenvPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $venvPython = Get-VenvPython
    & $venvPython @Arguments
}

function Pip-Install {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $pipArgs = @("-m", "pip", "install") + $Arguments
    if ($PipIndexUrl) {
        $pipArgs += @("--index-url", $PipIndexUrl)
    }

    try {
        Invoke-VenvPython -Arguments $pipArgs
        return
    } catch {
        if (-not $PipIndexUrl) {
            throw
        }

        Write-Host "Primary pip index failed, retrying with fallback: $PipFallbackIndexUrl"
        Invoke-VenvPython -Arguments (@("-m", "pip", "install") + $Arguments + @("--index-url", $PipFallbackIndexUrl))
    }
}

function Resolve-TorchPackageVersions {
    switch -Wildcard ($TorchVersion) {
        "2.3.*" {
            if (-not $TorchVisionVersion) { $script:TorchVisionVersion = "0.18.1" }
            if (-not $TorchAudioVersion) { $script:TorchAudioVersion = "2.3.1" }
            if (-not $TorchGeometricVersion) { $script:TorchGeometricVersion = "2.5.3" }
            break
        }
        "2.7.*" {
            if (-not $TorchVisionVersion) { $script:TorchVisionVersion = "0.22.1" }
            if (-not $TorchAudioVersion) { $script:TorchAudioVersion = "2.7.1" }
            if (-not $TorchGeometricVersion) { $script:TorchGeometricVersion = "2.7.0" }
            break
        }
        default {
            throw "Unsupported TORCH_VERSION for automatic torchvision/torchaudio matching: $TorchVersion"
        }
    }
}

function Assert-PythonVersion {
    $pythonCmd = Get-PythonCommand
    if ($pythonCmd.Length -gt 1) {
        $versionText = & $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length - 1)] -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    } else {
        $versionText = & $pythonCmd[0] -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    }
    $version = [Version]($versionText.Trim() + ".0")
    if ($version -lt [Version]"3.9.0") {
        throw "Python $($versionText.Trim()) is too old. Please use Python 3.9+."
    }
    if ($version -lt [Version]"3.10.0") {
        Write-Warning "Python $($versionText.Trim()) can work, but Python 3.10+ is recommended for a newer CUDA/PyTorch stack."
    }
}

Write-Host "[1/6] Checking Python..."
Assert-PythonVersion

Write-Host "[2/6] Creating virtual environment at $VenvDir ..."
Invoke-Python -Arguments @("-m", "venv", $VenvDir)

Write-Host "[3/6] Upgrading pip tooling..."
try {
    Pip-Install -Arguments @("--upgrade", "pip", "setuptools", "wheel")
} catch {
    Write-Warning "Failed to upgrade pip tooling. Continuing with the existing pip version."
}

Resolve-TorchPackageVersions

if ($CudaFlavor -eq "cpu") {
    $torchIndexUrl = "https://download.pytorch.org/whl/cpu"
} else {
    $torchIndexUrl = "https://download.pytorch.org/whl/$CudaFlavor"
}

Write-Host "[4/6] Installing PyTorch $TorchVersion / torchvision $TorchVisionVersion / torchaudio $TorchAudioVersion ($CudaFlavor)..."
Pip-Install -Arguments @(
    "torch==$TorchVersion",
    "torchvision==$TorchVisionVersion",
    "torchaudio==$TorchAudioVersion",
    "--index-url",
    $torchIndexUrl
)

$torchMajorMinor = (& (Get-VenvPython) -c "import torch; parts = torch.__version__.split('+', 1)[0].split('.'); print(f'{parts[0]}.{parts[1]}.0')").Trim()
$pygWhlUrl = "https://data.pyg.org/whl/torch-$torchMajorMinor+$CudaFlavor.html"

Write-Host "[5/6] Installing PyTorch Geometric and project dependencies..."
Pip-Install -Arguments @(
    "--force-reinstall",
    "--no-cache-dir",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
    "-f",
    $pygWhlUrl
)

Pip-Install -Arguments @(
    "--force-reinstall",
    "--no-cache-dir",
    "torch_geometric==$TorchGeometricVersion",
    "numpy==$NumpyVersion",
    "networkx",
    "matplotlib",
    "cvxpy",
    "osmnx",
    "scipy",
    "pandas",
    "shapely"
)

Write-Host "[6/6] Running sanity checks..."
Invoke-VenvPython -Arguments @(
    "-c",
    @"
import torch
import torch_geometric
import cvxpy
import osmnx
from env.Traffic import TrafficPowerEnv

env = TrafficPowerEnv()
state = env.get_graph_state()

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("torch_geometric:", torch_geometric.__version__)
print("cvxpy:", cvxpy.__version__)
print("osmnx:", osmnx.__version__)
print("state.x shape:", tuple(state.x.shape))
assert state.x.shape[1] == 15, state.x.shape
"@
)

Write-Host ""
Write-Host "Environment is ready."
Write-Host ""
Write-Host "Activate it with:"
Write-Host "  .\$((Split-Path -Leaf $VenvDir))\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Typical commands:"
Write-Host "  python main.py train"
Write-Host "  python main.py train-real"
Write-Host "  python evaluation/run_evaluation.py"
