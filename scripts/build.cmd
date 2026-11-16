@echo off
setlocal
cd /d "%~dp0.."
set "MOVIRA_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%MOVIRA_VSWHERE%" (
  echo Visual Studio C++ Build Tools are required.
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%MOVIRA_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "MOVIRA_VS=%%i"
if not defined MOVIRA_VS (
  echo Install the Desktop development with C++ workload.
  exit /b 1
)
call "%MOVIRA_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "MOVIRA_CMAKE=%MOVIRA_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%MOVIRA_CMAKE%" set "MOVIRA_CMAKE=cmake"
"%MOVIRA_CMAKE%" -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
"%MOVIRA_CMAKE%" --build build
if errorlevel 1 exit /b 1
"%MOVIRA_CMAKE%" --build build --target test
exit /b %errorlevel%
