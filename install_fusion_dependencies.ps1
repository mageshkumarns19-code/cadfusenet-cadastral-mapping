# =====================================================================
# install_fusion_dependencies.ps1
# Installs all packages needed for fusion_pipeline.py, using Python 3.13
# specifically (since TensorFlow does not yet support Python 3.14, but
# DOES support 3.13).
# =====================================================================
# HOW TO RUN THIS SCRIPT:
#   1. Make sure you've already installed Python 3.13 from python.org
#      (see earlier instructions) before running this.
#   2. Right-click this file -> "Run with PowerShell"
#      OR open PowerShell in this folder and type:  .\install_fusion_dependencies.ps1
#
#   If you get "running scripts is disabled on this system", run this
#   ONE line first, type Y when asked, then try again:
#      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# =====================================================================

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Checking for Python 3.13..." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Check Python 3.13 is actually available via the 'py' launcher
$pyCheck = py -3.13 --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! Python 3.13 was not found via the 'py' launcher." -ForegroundColor Red
    Write-Host "   Please install it from https://www.python.org/downloads/windows" -ForegroundColor Red
    Write-Host "   (download a 3.13.x installer, check 'Add python.exe to PATH')" -ForegroundColor Red
    Read-Host "`nPress Enter to close this window"
    exit
} else {
    Write-Host "Found: $pyCheck" -ForegroundColor Green
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "Installing required packages using Python 3.13..." -ForegroundColor Cyan
Write-Host "This may take several minutes (tensorflow is a large package)." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Upgrade pip first for this Python version
py -3.13 -m pip install --upgrade pip

$packages = @("tensorflow", "rasterio", "geopandas", "shapely", "numpy", "matplotlib")

foreach ($pkg in $packages) {
    Write-Host "`n--- Installing $pkg (Python 3.13) ---" -ForegroundColor Yellow
    py -3.13 -m pip install $pkg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!! $pkg failed to install." -ForegroundColor Red
        Write-Host "   (rasterio/geopandas sometimes need extra Windows setup -" -ForegroundColor Red
        Write-Host "   if this happens, send the error screenshot for help.)" -ForegroundColor Red
    } else {
        Write-Host "$pkg installed successfully." -ForegroundColor Green
    }
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "DONE. You can now run: py -3.13 fusion_pipeline.py" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

Read-Host "`nPress Enter to close this window"
