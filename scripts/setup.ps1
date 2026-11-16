$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$deps = Join-Path $root '.deps'
New-Item -ItemType Directory -Force -Path $deps | Out-Null
if (-not (Test-Path (Join-Path $deps 'opencv/build/OpenCVConfig.cmake'))) {
    $installer = Join-Path $deps 'opencv-4.10.0-windows.exe'
    Invoke-WebRequest 'https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe' -OutFile $installer
    $process = Start-Process -FilePath $installer -ArgumentList @("-o$deps", '-y') -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0) { throw 'OpenCV extraction failed' }
}
Write-Host 'Dependencies ready. Build: .\scripts\build.cmd'
