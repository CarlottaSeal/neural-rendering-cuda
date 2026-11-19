@echo off
REM Build + install a rasterizer into .venv as `diff_gaussian_rasterization`.
REM   scripts\build_rasterizer.bat upstream            -> third_party official submodule
REM   scripts\build_rasterizer.bat opt "DEFINES..."    -> p1-3dgs\rasterizer with RAST_DEFINES
REM e.g. scripts\build_rasterizer.bat opt "OPT_SHMEM_FEATURES=1 OPT_WARP_REDUCE_GRADS=1"
setlocal
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set DISTUTILS_USE_SDK=1
set "CL=/Zc:preprocessor %CL%"
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=12.0
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if /I "%~1"=="upstream" (
  set "SRC=%ROOT%\third_party\gaussian-splatting\submodules\diff-gaussian-rasterization"
  set RAST_DEFINES=
) else (
  set "SRC=%ROOT%\p1-3dgs\rasterizer"
  set "RAST_DEFINES=%~2"
)
REM stale objects would ignore changed defines
if exist "%SRC%\build" rmdir /s /q "%SRC%\build"
echo === %~1 [%RAST_DEFINES%] from %SRC% ===
uv pip install --python "%PY%" --no-build-isolation --reinstall --no-cache "%SRC%" || exit /b 1
