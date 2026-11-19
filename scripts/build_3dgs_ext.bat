@echo off
REM Build the official 3DGS CUDA extensions into .venv (Windows, MSVC + CUDA 13.x).
REM Usage: scripts\build_3dgs_ext.bat [module ...]   (default: all three)
setlocal
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set DISTUTILS_USE_SDK=1
REM CUDA 13 CCCL refuses MSVC's traditional preprocessor; cl.exe picks up flags from %CL%,
REM which also covers the host-compiler calls nvcc makes.
set "CL=/Zc:preprocessor %CL%"
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=12.0
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "MODS=%*"
if "%MODS%"=="" set "MODS=diff-gaussian-rasterization simple-knn fused-ssim"
for %%m in (%MODS%) do (
  echo === building %%m ===
  uv pip install --python "%PY%" --no-build-isolation --reinstall "%ROOT%\third_party\gaussian-splatting\submodules\%%m" -v || exit /b 1
)
