param(
    [string]$VenvPath = ".venv",
    [switch]$Install
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[python-setup] $Message"
}

function ResolvePython312 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) {
        throw "Python launcher 'py' was not found. Install Python 3.12 from python.org and ensure the launcher is on PATH."
    }
    $output = & $launcher.Path -0p
    $pythonPath = $null
    foreach ($line in $output -split "`r?`n") {
        if ($line -match "3\.12") {
            $candidate = $line.Trim() -split '\s+'
            $candidatePath = $candidate[-1]
            if (Test-Path $candidatePath) {
                $pythonPath = $candidatePath
                break
            }
        }
    }
    if (-not $pythonPath) {
        throw "Python 3.12 is not registered with the launcher. Install 3.12.5 from python.org (Install for all users)."
    }
    return $pythonPath
}

$python312 = ResolvePython312
Write-Info "Using interpreter $python312"

$venvPython = Join-Path $VenvPath "Scripts/python.exe"
if (Test-Path $venvPython) {
    $versionOutput = & $venvPython --version
    if ($versionOutput -notmatch "3\.12") {
        Write-Info "Existing virtualenv reports '$versionOutput' - recreating with Python 3.12."
        Remove-Item -Recurse -Force $VenvPath
    } else {
        Write-Info "Virtualenv already on $versionOutput"
    }
}

if (-not (Test-Path $venvPython)) {
    Write-Info "Creating virtualenv at $VenvPath"
    & $python312 -m venv $VenvPath
}

if ($Install) {
    $pip = Join-Path $VenvPath "Scripts/pip.exe"
    Write-Info "Upgrading pip"
    & $python312 -m pip install --upgrade pip
    Write-Info "Installing project in editable mode with tests extras"
    & $pip install -e ".[tests]"
}

Write-Info "All set. Activate with:`n`t& $((Join-Path $VenvPath 'Scripts/Activate.ps1'))"
