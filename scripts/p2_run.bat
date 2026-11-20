@echo off
REM Run a P2 Python script inside the VS 2022 x64 environment so the JIT CUDA extension can build.
REM   scripts\p2_run.bat p2-ingp\tests\test_cuda.py
setlocal
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set DISTUTILS_USE_SDK=1
set "CL=/Zc:preprocessor %CL%"
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=12.0
"%~dp0..\.venv\Scripts\python.exe" %*
