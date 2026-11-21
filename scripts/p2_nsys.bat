@echo off
REM nsys timeline of a P2 script, with the same VS env as p2_run.bat so the JIT extension resolves cl/link.
REM   scripts\p2_nsys.bat outputs\prof\p2_ref_cudaenc p2-ingp\profile_step.py --nvtx_only --backend cuda
setlocal
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set DISTUTILS_USE_SDK=1
set "CL=/Zc:preprocessor %CL%"
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=12.0
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "OUT=%~1"
shift
set ARGS=
:collect
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
REM build the JIT extension outside nsys, then load the cached build inside it
"%PY%" -c "import sys; sys.path.insert(0, r'%ROOT%\p2-ingp'); from ingp.cuda_ext import ext; ext()" || exit /b 1
set INGP_PREBUILT=1
"%ProgramFiles%\NVIDIA Corporation\Nsight Systems 2026.1.1\target-windows-x64\nsys.exe" profile -t cuda,nvtx --force-overwrite true -o "%OUT%" "%PY%" %ARGS%
